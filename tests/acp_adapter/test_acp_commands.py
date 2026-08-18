import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock
from unittest.mock import AsyncMock, MagicMock, patch

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self._supports_active_turn_redirect = True
        self.steers = []
        self.redirects = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def redirect(self, text):
        self.redirects.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        if kwargs:
            self.updates.append((kwargs.get("session_id"), kwargs.get("update")))
        else:
            self.updates.append((args[0], args[1]))

    async def request_permission(self, *args, **kwargs):
        return SimpleNamespace(outcome="allow")


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def make_agent_and_state():
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    captured = {}
    sentinel_db = NoopDb()

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m", "provider": "p"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "p",
                "api_mode": "chat_completions",
                "base_url": "u",
                "api_key": "k",
                "command": None,
                "args": [],
            },
        ),
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, CapturingAgent)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["provider"] == "p"
    assert captured["session_id"] == "acp-session"


def test_qualified_model_selects_its_provider(monkeypatch):
    """A provider/model id from Multica must override the ACP config provider."""
    captured = {}

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def resolve_runtime_provider(**kwargs):
        captured["requested_provider"] = kwargs.get("requested")
        return {
            "provider": kwargs.get("requested") or "openai-codex",
            "api_mode": "chat_completions",
            "base_url": "https://opencode.ai/zen/go/v1",
            "api_key": "key",
            "command": None,
            "args": [],
        }

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "gpt-5.6-luna", "provider": "openai-codex"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )

    manager = SessionManager(db=NoopDb())
    agent = manager._make_agent(
        session_id="acp-session",
        cwd=".",
        model="opencode-go/deepseek-v4-flash",
    )

    assert isinstance(agent, CapturingAgent)
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["requested_provider"] == "opencode-go"
    assert captured["provider"] == "opencode-go"


@pytest.mark.asyncio
async def test_new_session_forwards_model_and_provider(monkeypatch):
    """ACP session/new must pass Multica's model/provider to SessionManager."""
    captured = {}
    state = SimpleNamespace(
        session_id="acp-session",
        cwd=".",
        agent=SimpleNamespace(model="deepseek-v4-flash", provider="opencode-go"),
    )

    class RecordingManager:
        def create_session(self, **kwargs):
            captured.update(kwargs)
            return state

    acp_agent = HermesACPAgent(session_manager=RecordingManager())
    monkeypatch.setattr(acp_agent, "_register_session_mcp_servers", AsyncMock())
    monkeypatch.setattr(acp_agent, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(acp_agent, "_schedule_available_commands_update", lambda *_args: None)
    monkeypatch.setattr(acp_agent, "_schedule_usage_update", lambda *_args: None)
    monkeypatch.setattr(acp_agent, "_build_model_state", lambda *_args: None)
    monkeypatch.setattr(acp_agent, "_session_modes", lambda *_args: None)
    monkeypatch.setattr(acp_agent, "_provenance_meta", lambda *_args: None)

    await acp_agent.new_session(
        cwd=".",
        model="opencode-go/deepseek-v4-flash",
        provider="opencode-go",
    )

    assert captured["cwd"] == "."
    assert captured["model"] == "opencode-go/deepseek-v4-flash"
    assert captured["requested_provider"] == "opencode-go"
    assert captured["base_url"] is None
    assert captured["api_mode"] is None


def test_qualified_model_selection_overrides_current_provider():
    """ACP model switching must honor provider/model ids from Multica."""
    assert HermesACPAgent._resolve_model_selection(
        "opencode-go/deepseek-v4-flash",
        "openai-codex",
    ) == ("opencode-go", "deepseek-v4-flash")


@pytest.mark.asyncio
async def test_acp_steer_slash_command_injects_into_running_agent():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer prefer the simpler fix")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == ["prefer the simpler fix"]
    assert fake.runs == []








@pytest.mark.asyncio
async def test_acp_cancel_publishes_hard_stop_while_holding_runtime_lock():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "original request"
    observed = {}

    def interrupt():
        acquired = state.runtime_lock.acquire(blocking=False)
        observed["lock_held"] = not acquired
        if acquired:
            state.runtime_lock.release()

    fake.interrupt = interrupt

    await acp_agent.cancel(state.session_id)

    assert observed["lock_held"] is True
    assert state.cancel_event.is_set()
    assert state.interrupted_prompt_text == "original request"






