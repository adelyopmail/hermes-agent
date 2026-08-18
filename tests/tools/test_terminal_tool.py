"""Regression tests for sudo detection and sudo password handling."""

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "once per session" in description


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"




def test_terminal_execution_is_serialized_per_task():
    """Two concurrent terminal calls for one task must not overlap."""
    import concurrent.futures
    import threading
    import time

    class FakeEnvironment:
        def __init__(self):
            self.active = 0
            self.overlap = False
            self.guard = threading.Lock()

        def execute(self, command, **_kwargs):
            with self.guard:
                self.active += 1
                self.overlap = self.overlap or self.active > 1
            time.sleep(0.03)
            with self.guard:
                self.active -= 1
            return {"output": command, "returncode": 0}

    env = FakeEnvironment()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda command: terminal_tool._execute_with_task_lock(env, "task-id", command), ["one", "two"]))

    assert env.overlap is False


def test_managed_multica_read_command_is_narrowly_classified():
    assert terminal_tool._is_managed_multica_read_command("multica issue get issue-id --output json") is True
    assert terminal_tool._is_managed_multica_read_command("multica issue comment list issue-id --roots-only --summary --compact --output json") is True
    assert terminal_tool._is_managed_multica_read_command("multica issue update issue-id --title x") is False


def test_managed_multica_command_uses_direct_executor(monkeypatch):
    """Task-scoped Multica API calls bypass the wedged shell wrapper only."""
    class FakeEnvironment:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("LocalEnvironment wrapper should not run")

    expected = {"output": "ok", "returncode": 0}
    monkeypatch.setattr(terminal_tool, "_execute_managed_multica_direct", lambda *_a, **_k: expected)

    result = terminal_tool._execute_with_task_lock(
        FakeEnvironment(),
        "task-id",
        "multica issue get issue-id --output json",
    )

    assert result == expected


def test_multica_task_context_is_used_when_tool_task_id_is_missing(monkeypatch):
    """ACP tool workers must not collapse a managed task onto default."""
    monkeypatch.setenv("MULTICA_TASK_ID", "task-7529622e")

    assert terminal_tool._resolve_container_task_id(None) == "task-7529622e"
    assert terminal_tool._resolve_container_task_id("default") == "task-7529622e"
    assert terminal_tool._resolve_container_task_id("acp-session-id") == "task-7529622e"


def test_non_task_without_tool_id_still_uses_default(monkeypatch):
    monkeypatch.delenv("MULTICA_TASK_ID", raising=False)

    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_validate_workdir_allows_unicode_filesystem_paths():
    assert terminal_tool._validate_workdir(
        "/Users/alice/Documents/Obs_Hermes_Data/项目-projects/客户拜访"
    ) is None
    assert terminal_tool._validate_workdir("/tmp/テスト") is None
    assert terminal_tool._validate_workdir("/home/jürgen/über projekt") is None


def test_validate_workdir_still_blocks_metachars_in_unicode_paths():
    # Widening to Unicode letters must not open the injection boundary:
    # shell metacharacters and control chars stay rejected even when mixed
    # with non-ASCII path segments.
    assert terminal_tool._validate_workdir("/tmp/テスト; rm -rf /")
    assert terminal_tool._validate_workdir("/tmp/项目$(whoami)")
    assert terminal_tool._validate_workdir("/tmp/über`id`")
    assert terminal_tool._validate_workdir("/tmp/テスト\nwhoami")
    assert terminal_tool._validate_workdir("/tmp/项目|cat /etc/passwd")
    assert terminal_tool._validate_workdir("/tmp/ü\x00ber")


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
