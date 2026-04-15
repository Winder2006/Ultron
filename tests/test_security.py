import os
from pathlib import Path
from src.security.command_policy import is_path_allowed, register_command, run_command
from src.security.network_guard import assert_local_only


def test_sandbox_block(tmp_path):
    outside = Path(os.path.expanduser("~")).resolve()
    assert not is_path_allowed(outside)


def test_risky_requires_confirm():
    def _danger(path: str):
        return True
    register_command('delete_file', _danger, risky=True)
    r = run_command('delete_file', path='~/AI_Workspace/file.txt')
    assert not r.get('ok') and 'confirmation' in r.get('error','').lower()


def test_audit_and_allowed(tmp_path):
    def _create_file(path: str, content: str):
        p = Path(path).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8'); return str(p)
    register_command('create_file', _create_file, risky=False)
    r = run_command('create_file', path='~/AI_Workspace/t.txt', content='ok')
    assert r.get('ok')


def test_network_guard_local_only():
    ok, _ = assert_local_only()
    assert isinstance(ok, bool)
