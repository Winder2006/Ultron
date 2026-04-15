# MOTHER AI Security Policy

- Least privilege: never run as admin/root.
- Sandbox: all writes under `~/AI_Workspace/` only.
- No arbitrary shell: only whitelisted commands via `src/security/command_policy.py`.
- Audit: every action logged to `logs/commands.log` (UTC).
- eDEX UI runs locally only; do not expose ports or reverse proxies.
- Windows: inbound firewall rule blocks `edex-ui.exe`.
- macOS/Linux: ensure no public listeners; use host firewall if enabled.
- Rotate logs periodically; redact secrets.

## Command Gate
- Only registered commands are allowed.
- Risky commands require explicit confirmation.
- Rate-limited (10/min) with cool-off on repeated failures.

## Network Guard
- On startup, assert no non-loopback listeners.
- If offenders are detected, warn and wait for user confirmation before proceeding.
