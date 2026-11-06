"""resume 호환(task13 방식 b): 지문 기록, 프롬프트가 바뀐 resume 의 옛 생각 빼기(요청 사본만), 안내 1회. 모델은 대본이다."""
from __future__ import annotations

import io
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from deepagents.profiles.harness import harness_profiles
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from code_agent import agent as agent_module
from code_agent import prompts
from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE
from code_agent.render import Renderer
from code_agent.resume_compat import (
    PLACEHOLDER,
    Fingerprint,
    MemoryPromptStore,
    PromptRecord,
    SqlitePromptStore,
    drop_reasoning,
    strip_reasoning,
)
from code_agent.settings import Settings

ENV_A = "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n- 있음: uv 0.13.7\n"
ENV_B = "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n- 있음: uv 0.13.7, node 22.11.0\n"
ENV_C = "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n- 있음: uv 0.13.8, node 22.11.0\n"


def said(text: str, thought: str = "옛 생각") -> AIMessage:
    """OpenRouter 모양: 생각이 reasoning_content 와 reasoning_details(서명 포함)에 있다."""
    return AIMessage(content=text, additional_kwargs={
        "reasoning_content": thought,
        "reasoning_details": [{"type": "reasoning.text", "text": thought, "signature": "sig", "index": 0}]},
        response_metadata={"model_provider": "openrouter", "finish_reason": "stop"})


def thinking(message: BaseMessage) -> bool:
    if not isinstance(message, AIMessage):
        return False
    blocks = message.content if isinstance(message.content, list) else []
    return (any(k in message.additional_kwargs for k in ("reasoning_content", "reasoning_details"))
            or any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in blocks))


class Thread:
    """같은 체크포인터와 지문 저장소로 agent 를 다시 만들면 --resume 과 같다."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.saver = InMemorySaver()
        self.store = MemoryPromptStore()

    def agent(self, steps: list, *, env_text: str = ENV_A, store=None):
        model = ScriptedChatModel(steps=steps)
        agent = build_agent(Settings(data_dir=self.tmp_path / "data"), self.workspace, self.saver, model=model,
                            env_text=env_text, prompt_store=self.store if store is None else store)
        return agent, model

    def turn(self, agent, text: str, *, thread: str = "t", decide=None) -> list[dict]:
        """한 턴을 돌리고 prompt_migrated 이벤트들을 돌려준다."""
        parts: list = []
        run_turn(agent, {"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": thread}},
                 decide=decide or (lambda requests: [APPROVE for _ in requests]), on_part=parts.append)
        return [d for _, mode, d in parts if mode == "custom" and d.get("type") == "prompt_migrated"]


def test_fingerprint_stable(tmp_path: Path) -> None:
    t = Thread(tmp_path)
    for thread in ("a", "b"):
        agent, _ = t.agent([said("끝")])
        t.turn(agent, "하이", thread=thread)
    a, b = t.store.get("a"), t.store.get("b")
    assert a is not None and b is not None and a.fingerprint == b.fingerprint


def test_fingerprint_changes_on_prompt_or_tool_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 하네스 프로필 등록은 프로세스 전역이고 같은 키에 병합된다. 여기서 더한 override 가 다른 시험에 남지 않게 되돌린다
    monkeypatch.setattr(harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES))
    t = Thread(tmp_path)

    def record(thread: str) -> Fingerprint:
        agent, _ = t.agent([said("끝")])
        t.turn(agent, "하이", thread=thread)
        found = t.store.get(thread)
        assert found is not None
        return found.fingerprint

    base = record("a")
    monkeypatch.setattr(prompts, "CODING_PROMPT", prompts.CODING_PROMPT + "- 새 규칙\n")
    edited_prompt = record("b")
    assert edited_prompt.system_sha != base.system_sha and edited_prompt.tools_sha == base.tools_sha
    monkeypatch.setattr(agent_module, "TOOL_DESCRIPTIONS", {**agent_module.TOOL_DESCRIPTIONS, "read_file": "Reads."})
    edited_tool = record("c")
    assert edited_tool.tools_sha != base.tools_sha


def test_first_call_records(tmp_path: Path) -> None:
    t = Thread(tmp_path)
    agent, _ = t.agent([said("끝")])
    assert t.turn(agent, "하이") == []
    record = t.store.get("t")
    assert record is not None and record.migrated_at is None


def test_resume_same_prompt_no_strip(tmp_path: Path) -> None:
    t = Thread(tmp_path)
    first, model1 = t.agent([said("첫 답")])
    t.turn(first, "하나")
    resumed, model2 = t.agent([said("둘째 답", "새 생각")])
    assert t.turn(resumed, "둘") == []
    request = model2.calls[0]
    assert request[:len(model1.calls[0])] == model1.calls[0]  # 덧붙이기 관계 그대로
    assert thinking(request[2])  # 옛 생각을 그대로 보낸다(지문이 같다)


def test_resume_changed_prompt_strips_old_thinking(tmp_path: Path) -> None:
    t = Thread(tmp_path)
    first, _ = t.agent([said("첫 답")])
    t.turn(first, "하나")
    step = tool_call("ls", {"path": "/"})
    step.additional_kwargs.update(reasoning_content="새 생각")
    resumed, model = t.agent([step, said("끝", "또 새 생각")], env_text=ENV_B)  # 그사이 node 를 설치했다
    events = t.turn(resumed, "둘")
    assert events == [{"type": "prompt_migrated", "reason": "prompt", "stripped": 1}]  # 안내는 한 번
    record = t.store.get("t")
    assert record is not None and record.migrated_at == 3  # [하나, 첫 답, 둘]
    old, new = model.calls
    assert not any(thinking(m) for m in old)  # 요청 사본에서 옛 생각을 뺐다
    assert new[:len(old)] == old  # 이전 뒤로는 덧붙이기만 한다
    assert thinking(new[len(old)])  # 이전 뒤에 생긴 응답의 생각은 그대로 보낸다
    state = resumed.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    assert thinking(state[1])  # state(체크포인트)는 바꾸지 않는다


def test_old_thread_without_record_is_migrated(tmp_path: Path) -> None:
    """지문 기록이 생기기 전의 thread(표에 행이 없음)를 이어 가면 그때의 system 을 모르므로 이전한다."""
    t = Thread(tmp_path)
    first, _ = t.agent([said("첫 답")])
    t.turn(first, "하나")
    resumed, model = t.agent([said("끝")], store=MemoryPromptStore())
    assert t.turn(resumed, "둘") == [{"type": "prompt_migrated", "reason": "prompt", "stripped": 1}]
    assert not any(thinking(m) for m in model.calls[0])


def test_model_only_change_no_strip(tmp_path: Path) -> None:
    """모델만 바뀌면 공급자가 옛 생각을 버리므로(400 아님) 빼지 않고 알리기만 한다."""
    t = Thread(tmp_path)
    first, _ = t.agent([said("첫 답")])
    t.turn(first, "하나")
    old = t.store.get("t")
    assert old is not None
    t.store.put("t", PromptRecord(Fingerprint(old.fingerprint.system_sha, old.fingerprint.tools_sha, "other-model")))
    resumed, model = t.agent([said("끝")])
    assert t.turn(resumed, "둘") == [{"type": "prompt_migrated", "reason": "model", "stripped": 0}]
    assert thinking(model.calls[0][2])


def test_interrupted_tool_round_then_resume(tmp_path: Path) -> None:
    """승인 중 끊긴 도구 라운드(PatchToolCalls 가 결과를 채움) 뒤에도 생각을 빼고, 도구 호출은 그대로 둔다."""
    t = Thread(tmp_path)
    step = tool_call("execute", {"command": "uv run pytest"})
    step.additional_kwargs.update(reasoning_content="돌려 보자")
    first, _ = t.agent([step])

    def interrupt(requests: list[dict]) -> list[dict]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        t.turn(first, "시험 돌려", decide=interrupt)
    resumed, model = t.agent([said("끝")], env_text=ENV_B)
    t.turn(resumed, "다른 거 하자")
    request = model.calls[0]
    [cut] = [m for m in request if isinstance(m, AIMessage)]
    assert not thinking(cut) and cut.tool_calls[0]["name"] == "execute"
    assert any(isinstance(m, ToolMessage) and "did not complete" in m.text for m in request)
    thought_only = AIMessage(content=[{"type": "thinking", "thinking": "생각뿐", "signature": "s"}])
    assert drop_reasoning(thought_only).content == PLACEHOLDER  # Anthropic 은 빈 assistant 를 거부한다


def test_request_shapes_both_providers() -> None:
    pytest.importorskip("langchain_openrouter")
    from langchain_anthropic.chat_models import _format_messages
    from langchain_openrouter import ChatOpenRouter

    anthropic = AIMessage(content=[{"type": "thinking", "thinking": "생각", "signature": "s"},
                                   {"type": "text", "text": "답"}],
                          response_metadata={"model_provider": "anthropic"})
    messages = [HumanMessage("하나"), drop_reasoning(said("답")), HumanMessage("둘")]
    dicts, _ = ChatOpenRouter(model="anthropic/claude-opus-5.5", api_key="DUMMY")._create_message_dicts(messages, None)
    [assistant] = [d for d in dicts if d["role"] == "assistant"]
    assert not {"reasoning", "reasoning_details"} & assistant.keys() and assistant["content"] == "답"
    _, formatted = _format_messages([HumanMessage("하나"), drop_reasoning(anthropic), HumanMessage("둘")],
                                    model="claude-opus-5-5")
    [assistant] = [m for m in formatted if m["role"] == "assistant"]
    blocks = assistant["content"] if isinstance(assistant["content"], list) else [{"type": "text"}]
    assert blocks and all(b["type"] != "thinking" for b in blocks)


def test_strip_windows_follow_ids_after_summary() -> None:
    """요약이 앞쪽을 요약 메시지로 바꿔 요청 위치가 state 와 달라도, 구간은 state 인덱스로 주고 메시지는 id 로 찾는다.

    task9 의 요약 창도 같은 함수에 구간을 더한다. 구간 밖(이전 뒤)의 생각과 원래 메시지는 그대로다.
    """
    state = [HumanMessage("하나", id="h1"), said("답1").model_copy(update={"id": "a1"}), HumanMessage("둘", id="h2"),
             said("답2").model_copy(update={"id": "a2"}), HumanMessage("셋", id="h3"),
             said("답3").model_copy(update={"id": "a3"}), HumanMessage("넷", id="h4")]
    summarized = [HumanMessage("요약: 하나·둘", id="summary"), *state[3:]]  # state[0:3] 이 요약됐다
    messages, changed = strip_reasoning(summarized, state, [(0, 4)])
    assert changed == 1 and [m.id for m in messages] == [m.id for m in summarized]
    assert not thinking(messages[1]) and thinking(messages[3])  # a2 는 뺐고 구간 밖 a3 는 그대로다
    both, changed = strip_reasoning(summarized, state, [(0, 4), (5, 6)])  # 구간 두 개(이전 창과 요약 창 같은 것)
    assert changed == 2 and not any(thinking(m) for m in both)
    assert all(thinking(m) for m in state if isinstance(m, AIMessage))  # state 는 그대로
    assert strip_reasoning(summarized, state, [(4, 5)]) == (summarized, 0)  # 구간에 AI 메시지가 없으면 그대로


def test_no_notice_without_earlier_answer(tmp_path: Path) -> None:
    """기록은 있지만 앞선 응답이 없는 thread(첫 호출이 실패)는 프롬프트가 바뀌어도 뺄 생각이 없어 알리지 않는다."""
    t = Thread(tmp_path)

    def fail(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("공급자 오류")

    first, _ = t.agent([fail])
    with pytest.raises(RuntimeError):
        t.turn(first, "하나")
    resumed, _ = t.agent([said("끝")], env_text=ENV_B)
    assert t.turn(resumed, "둘") == []
    record = t.store.get("t")
    assert record is not None and record.migrated_at is None


def test_sqlite_store_separate_table(tmp_path: Path) -> None:
    """지문은 표를 따로 둔다. 옛 프로세스의 code_agent_threads 3열 INSERT 가 그대로 되고, 다른 연결에서도 읽힌다."""
    from code_agent.cli import Sessions

    db = tmp_path / "sessions.db"
    with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
        Sessions(conn)
        store = SqlitePromptStore(conn)
        record = PromptRecord(Fingerprint("s", "t", "m"), migrated_at=3)
        store.put("t1", record)
        conn.execute("INSERT INTO code_agent_threads VALUES (?, ?, ?)", ("old", "C:/ws", "2026-11-06"))
        assert [r[1] for r in conn.execute("PRAGMA table_info(code_agent_threads)")] == [
            "thread_id", "workspace", "created_at"]
        assert store.get("old") is None
    with closing(sqlite3.connect(db)) as other:
        assert SqlitePromptStore(other).get("t1") == record


def test_cli_store_shares_checkpointer_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """지문은 그래프 실행 중에 쓰고 SqliteSaver 는 체크포인트를 백그라운드에서 같은 연결로 저장하므로 같은 잠금을 쓴다."""
    from code_agent import cli

    seen: dict = {}
    monkeypatch.setattr(cli, "build_agent", lambda settings, workspace, saver, **kwargs: seen.update(saver=saver,
                                                                                                      **kwargs))
    with closing(sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False)) as conn:
        cli.session_agent(Settings(data_dir=tmp_path), tmp_path, cli.Sessions(conn), "")
    assert seen["prompt_store"].lock is seen["saver"].lock


def test_second_upgrade_moves_migrated_at(tmp_path: Path) -> None:
    t = Thread(tmp_path)
    for env_text, text in ((ENV_A, "하나"), (ENV_B, "둘")):
        agent, _ = t.agent([said(f"{text} 답")], env_text=env_text)
        t.turn(agent, text)
    agent, model = t.agent([said("셋 답")], env_text=ENV_C)
    assert t.turn(agent, "셋") == [{"type": "prompt_migrated", "reason": "prompt", "stripped": 2}]
    record = t.store.get("t")
    assert record is not None and record.migrated_at == 5  # [하나, 답, 둘, 답, 셋]
    assert not any(thinking(m) for m in model.calls[0])


def test_cli_notice_once_and_langfuse_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """지문 기록이 없는 옛 thread 를 CLI 로 이어 가면 노란 안내가 세션에 한 번 나오고, turn metadata 에 prompt_migrated 가 남는다."""
    from test_cli import Script, make_app

    from code_agent import cli
    from code_agent.tracing import LangfuseTurn
    from code_agent.usage import Usage

    app, _, _, _ = make_app(tmp_path, [said("첫 답")], "하나", "/exit")  # 지문 저장소 없이 만든 agent(옛 코드)
    app.run()
    model = ScriptedChatModel(steps=[said("둘째"), said("셋째")])
    monkeypatch.setattr(cli, "build_agent", lambda *args, **kwargs: build_agent(*args, model=model, **kwargs))
    out = io.StringIO()
    cli.App(app.settings, app.workspace, "manual", app.sessions, app.thread_id, Console(file=out, width=200),
            Script("둘", "셋", "/exit"), agent=cli.session_agent(app.settings, app.workspace, app.sessions, "")).run()
    assert out.getvalue().count("이 세션은 이전 버전의 프롬프트로 시작돼") == 1  # 두 턴 중 첫 턴에만
    assert not any(thinking(m) for m in model.calls[1][:3])  # 둘째 턴에도 옛 생각은 계속 뺀다

    class Root:
        def __init__(self) -> None:
            self.fields: dict = {}

        def update(self, **fields) -> None:
            self.fields = fields

    for event, expected in (({"reason": "prompt", "stripped": 1}, True), ({"reason": "model", "stripped": 0}, None)):
        root = Root()
        turn = LangfuseTurn(root, handler=None)
        turn.on_part(((), "custom", {"type": "prompt_migrated", **event}))
        turn.end(output="끝", usage=Usage())
        assert root.fields["metadata"].get("prompt_migrated") is expected
    rendered = io.StringIO()
    Renderer(Console(file=rendered)).on_part(((), "custom", {"type": "prompt_migrated", "reason": "model"}))
    assert "다른 모델로 시작됐다" in rendered.getvalue()
