"""iCloud Mail (IMAP) read-only access.

Connects to imap.mail.me.com:993 with the same Apple ID + app-specific
password used for CalDAV. Uses imap-tools (cleaner than imaplib) for
search and message body extraction.

Tool surface (registered in mother/llm/tools.py):
  • summarize_inbox   — recent unread, return list of {from, subject, snippet}
  • search_email      — keyword search across subject/from/body
  • read_email        — full body of one message by uid

Design choices:
  • Connection is cached across calls (module-level, lock-guarded) so
    each tool call skips the ~300ms TLS+login handshake. The cached
    connection is validated cheaply before use and re-logged-in once
    if iCloud has idled it out. All sockets carry a 15s timeout so a
    dead network can't hang the worker thread forever.
  • Read-only: SEEN flag is preserved (we use BODY.PEEK semantics).
    No moves, no deletes, no marks.
  • Sender names are stripped of email addresses for TTS — "Charles
    Williams" instead of "Charles Williams <foo@bar.com>".
  • Body text is HTML-stripped (bs4) and clipped to ~600 chars per
    message in summaries. Full body is available via read_email(uid).

NEVER auto-send. Drafting / sending requires SMTP, additional
guardrails, and is deliberately out of scope here.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger("mother.tools.icloud_email")

ICLOUD_IMAP_HOST = "imap.mail.me.com"
ICLOUD_IMAP_PORT = 993
# Socket timeout for every IMAP operation. Without this a dead network
# blocks the worker thread forever (imaplib's default is no timeout).
ICLOUD_IMAP_TIMEOUT_S = 15


def _get_creds() -> tuple[Optional[str], Optional[str]]:
    user = os.environ.get("ICLOUD_USERNAME") or os.environ.get("ICLOUD_USER")
    pw = os.environ.get("ICLOUD_APP_PASSWORD") or os.environ.get("ICLOUD_PASSWORD")
    return user, pw


# ─────────────────────── data shapes ───────────────────────────────


@dataclass
class EmailSummary:
    uid: str
    from_name: str
    from_addr: str
    subject: str
    snippet: str
    date: datetime
    seen: bool


def _strip_email(name_or_addr: str) -> str:
    """Pull a clean human name out of 'Name <addr@host>' formats."""
    if not name_or_addr:
        return ""
    s = name_or_addr.strip()
    # Strip trailing <addr>
    s = re.sub(r"\s*<[^>]+>\s*$", "", s)
    # Strip surrounding quotes
    s = s.strip('"').strip("'").strip()
    if not s:
        # Fall back to local part of email
        m = re.match(r"([^@]+)@", name_or_addr)
        if m:
            return m.group(1)
    return s


def _html_to_plain(html: str) -> str:
    """Cheap HTML strip for summaries. Defers heavy lifting to bs4
    if available; falls back to regex otherwise."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _connect():
    """Open a fresh IMAP connection and log in. Prefer `_mailbox()`,
    which reuses a cached connection across tool calls."""
    user, pw = _get_creds()
    if not user or not pw:
        raise RuntimeError(
            "iCloud credentials not set — add ICLOUD_USERNAME and "
            "ICLOUD_APP_PASSWORD to .env"
        )
    from imap_tools import MailBox
    mailbox = MailBox(
        ICLOUD_IMAP_HOST, ICLOUD_IMAP_PORT, timeout=ICLOUD_IMAP_TIMEOUT_S,
    )
    mailbox.login(user, pw, initial_folder="INBOX")
    return mailbox


# Cached logged-in MailBox shared across tool calls, so each voice query
# skips the ~300ms+ TLS handshake + LOGIN round-trips. IMAP connections
# are stateful and not safe for concurrent use, so the lock is held for
# the whole operation, serializing email tool calls (fine — they're rare
# and user-driven).
_mb_lock = threading.Lock()
_mb_cached = None  # type: Optional[object]


@contextmanager
def _mailbox():
    """Yield a live, logged-in MailBox.

    Reuses the cached connection when possible: validates it with a
    cheap `folder.set('INBOX')` (which also resets folder state for the
    caller); on any failure — iCloud idled the socket, network blip —
    discards and re-logs-in once. Unlike `with MailBox().login(...)`,
    this does NOT log out on exit; the connection stays warm for the
    next call. If the operation itself raises, the connection state is
    suspect, so it is discarded and the exception propagates.
    """
    global _mb_cached
    with _mb_lock:
        mb = _mb_cached
        if mb is not None:
            try:
                mb.folder.set("INBOX")
            except Exception:
                try:
                    mb.logout()
                except Exception:
                    pass
                mb = None
                _mb_cached = None
        if mb is None:
            mb = _connect()
            _mb_cached = mb
        try:
            yield mb
        except Exception:
            _mb_cached = None
            try:
                mb.logout()
            except Exception:
                pass
            raise


# ─────────────────────── core fetch ────────────────────────────────


def fetch_recent(
    hours: int = 24, max_messages: int = 20, *, dedicated: bool = False,
) -> List[EmailSummary]:
    """Pull recent messages (any seen state) from INBOX within `hours`.

    dedicated=True opens its OWN connection (logout on exit) instead of
    the cached one. Background jobs (world-state brief) must use this:
    the cached connection's lock is held for the whole IMAP operation,
    so a background refresh on it would make a user's email question
    queue behind ambient work for up to the socket timeout. The ~300ms
    TLS+LOGIN cost is invisible off the hot path.
    """
    from imap_tools import AND
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).date()
    messages: List[EmailSummary] = []

    def _fetch(mailbox) -> None:
        # mark_seen=False keeps the SEEN flag untouched. reverse=True
        # gets newest first; bulk=True batches FETCHes for speed.
        for msg in mailbox.fetch(
            criteria=AND(date_gte=cutoff),
            mark_seen=False,
            reverse=True,
            limit=max_messages,
            bulk=True,
        ):
            try:
                snippet = msg.text or _html_to_plain(msg.html or "")
                snippet = snippet[:400]
                messages.append(EmailSummary(
                    uid=str(msg.uid),
                    from_name=_strip_email(msg.from_),
                    from_addr=msg.from_,
                    subject=(msg.subject or "(no subject)").strip(),
                    snippet=snippet,
                    date=msg.date,
                    seen=("\\Seen" in (msg.flags or [])),
                ))
            except Exception as e:
                logger.debug("[icloud_email] msg parse failed: %s", e)

    if dedicated:
        mailbox = _connect()
        try:
            _fetch(mailbox)
        finally:
            try:
                mailbox.logout()
            except Exception:
                pass
    else:
        with _mailbox() as mailbox:
            _fetch(mailbox)
    return messages


def search_messages(
    query: str,
    max_messages: int = 10,
    days_back: int = 30,
) -> List[EmailSummary]:
    """Subject-only search via IMAP, then optional body filter via
    fetched text. iCloud's IMAP supports SUBJECT and FROM searches but
    not full-text body search reliably, so we cast a wider net by
    combining a date-bounded fetch with client-side body matching."""
    from imap_tools import AND, OR
    q = (query or "").strip()
    if not q:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()
    results: List[EmailSummary] = []
    with _mailbox() as mailbox:
        # Server-side filter on subject OR from — fast.
        for msg in mailbox.fetch(
            criteria=AND(
                OR(subject=q, from_=q),
                date_gte=cutoff,
            ),
            mark_seen=False,
            reverse=True,
            limit=max_messages,
            bulk=True,
        ):
            try:
                snippet = msg.text or _html_to_plain(msg.html or "")
                results.append(EmailSummary(
                    uid=str(msg.uid),
                    from_name=_strip_email(msg.from_),
                    from_addr=msg.from_,
                    subject=(msg.subject or "(no subject)").strip(),
                    snippet=snippet[:400],
                    date=msg.date,
                    seen=("\\Seen" in (msg.flags or [])),
                ))
            except Exception:
                continue
        # Client-side body match for misses, only if we got <half results
        # and the user might have meant a body word. Skipped if first
        # pass already returned plenty.
        if len(results) < max_messages // 2:
            seen_uids = {r.uid for r in results}
            for msg in mailbox.fetch(
                criteria=AND(date_gte=cutoff),
                mark_seen=False,
                reverse=True,
                limit=max_messages * 5,
                bulk=True,
            ):
                if str(msg.uid) in seen_uids:
                    continue
                body = msg.text or _html_to_plain(msg.html or "")
                if q.lower() in body.lower():
                    results.append(EmailSummary(
                        uid=str(msg.uid),
                        from_name=_strip_email(msg.from_),
                        from_addr=msg.from_,
                        subject=(msg.subject or "(no subject)").strip(),
                        snippet=body[:400],
                        date=msg.date,
                        seen=("\\Seen" in (msg.flags or [])),
                    ))
                    if len(results) >= max_messages:
                        break
    return results


def read_full(uid: str) -> Optional[dict]:
    """Get the full body of one message by UID. Returns
    {from, subject, date, body} or None if not found."""
    from imap_tools import AND
    with _mailbox() as mailbox:
        for msg in mailbox.fetch(
            criteria=AND(uid=uid),
            mark_seen=False,
            limit=1,
            bulk=True,
        ):
            body = msg.text or _html_to_plain(msg.html or "")
            return {
                "from": _strip_email(msg.from_),
                "from_addr": msg.from_,
                "subject": msg.subject or "(no subject)",
                "date": msg.date.isoformat() if msg.date else "",
                "body": body,
            }
    return None


# ─────────────────────── tool entrypoints ──────────────────────────


def summarize_inbox(args: dict) -> str:
    """Tool: summarize recent inbox. Args: hours (default 24), max (default 20),
    dedicated (background callers only — own connection, no shared lock)."""
    hours = int(args.get("hours", 24) or 24)
    max_messages = int(args.get("max", 20) or 20)
    dedicated = bool(args.get("dedicated", False))
    try:
        msgs = fetch_recent(
            hours=hours, max_messages=max_messages, dedicated=dedicated,
        )
    except Exception as e:
        return f"Email unavailable: {e}"
    if not msgs:
        return f"No mail in the last {hours} hours."
    unread = [m for m in msgs if not m.seen]
    parts = [f"{len(msgs)} message(s) in the last {hours}h"]
    if unread:
        parts.append(f"{len(unread)} unread")
    head = "; ".join(parts) + ":\n"
    lines: list[str] = []
    for m in msgs[:8]:
        flag = "" if m.seen else "* "
        lines.append(
            f"  {flag}{m.from_name} — {m.subject[:90]}"
        )
    overflow = "" if len(msgs) <= 8 else f"\n  ... plus {len(msgs) - 8} more."
    return head + "\n".join(lines) + overflow


def search_email(args: dict) -> str:
    """Tool: search inbox by subject/from/body keyword."""
    query = (args.get("query") or "").strip()
    if not query:
        return "No query provided."
    max_messages = int(args.get("max", 10) or 10)
    try:
        msgs = search_messages(query, max_messages=max_messages)
    except Exception as e:
        return f"Email search failed: {e}"
    if not msgs:
        return f"No messages match {query!r}."
    head = f"{len(msgs)} match(es) for {query!r}:\n"
    lines = []
    for m in msgs[:8]:
        # Cross-platform date formatting only — %-d is glibc-only and
        # raises ValueError on Windows, which made search_email crash
        # exactly when it FOUND matches.
        date_str = ""
        try:
            if m.date and hasattr(m.date, "strftime"):
                date_str = m.date.strftime("%b %d").replace(" 0", " ")
        except Exception:
            pass
        lines.append(
            f"  uid={m.uid} {date_str} — {m.from_name} — {m.subject[:80]}"
        )
    return head + "\n".join(lines)


def read_email(args: dict) -> str:
    """Tool: read full body of one message by UID."""
    uid = (args.get("uid") or "").strip()
    if not uid:
        return "No uid provided. Use search_email or summarize_inbox to find a uid first."
    try:
        msg = read_full(uid)
    except Exception as e:
        return f"Email read failed: {e}"
    if msg is None:
        return f"No message found with uid {uid}."
    body = msg["body"]
    # Cap so it doesn't blow the LLM context budget.
    if len(body) > 3000:
        body = body[:3000] + "\n... (truncated; ask for more if you need it)"
    return (
        f"From: {msg['from']}\n"
        f"Subject: {msg['subject']}\n"
        f"Date: {msg['date']}\n\n"
        f"{body}"
    )
