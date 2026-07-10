"""iCloud Mail (IMAP) read-only access.

Connects to imap.mail.me.com:993 with the same Apple ID + app-specific
password used for CalDAV. Uses imap-tools (cleaner than imaplib) for
search and message body extraction.

Tool surface (registered in mother/llm/tools.py):
  • summarize_inbox   — recent unread, return list of {from, subject, snippet}
  • search_email      — keyword search across subject/from/body
  • read_email        — full body of one message by uid

Design choices:
  • Connection is per-call, NOT cached. iCloud aggressively idles
    long-lived IMAP connections and re-handshaking takes ~300ms which
    is acceptable for an on-demand voice query. A future optimization
    is a connection pool with idle keepalive, but it's not worth the
    complexity yet.
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger("mother.tools.icloud_email")

ICLOUD_IMAP_HOST = "imap.mail.me.com"
ICLOUD_IMAP_PORT = 993


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
    """Open a fresh IMAP connection. Caller is responsible for closing
    via the context manager `with` statement."""
    user, pw = _get_creds()
    if not user or not pw:
        raise RuntimeError(
            "iCloud credentials not set — add ICLOUD_USERNAME and "
            "ICLOUD_APP_PASSWORD to .env"
        )
    from imap_tools import MailBox
    mailbox = MailBox(ICLOUD_IMAP_HOST, ICLOUD_IMAP_PORT)
    mailbox.login(user, pw, initial_folder="INBOX")
    return mailbox


# ─────────────────────── core fetch ────────────────────────────────


def fetch_recent(hours: int = 24, max_messages: int = 20) -> List[EmailSummary]:
    """Pull recent messages (any seen state) from INBOX within `hours`."""
    from imap_tools import AND
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).date()
    messages: List[EmailSummary] = []
    with _connect() as mailbox:
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
    with _connect() as mailbox:
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
    with _connect() as mailbox:
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
    """Tool: summarize recent inbox. Args: hours (default 24), max (default 20)."""
    hours = int(args.get("hours", 24) or 24)
    max_messages = int(args.get("max", 20) or 20)
    try:
        msgs = fetch_recent(hours=hours, max_messages=max_messages)
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
