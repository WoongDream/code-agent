"""말없이 멈추지 않기: 잘린 응답의 자동 복구(턴당 1번), 턴 끝 경고, 예산 상한 도달 표시. 모델은 대본이다."""
from __future__ import annotations

from pathlib import Path

import pytest
from fakes import tool_call, tool_calls
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from test_agent import Harness
from test_cli import make_app
from test_tracing import FakeTracer

from code_agent import agent as agent_module
from code_agent.approval import APPROVE
from code_agent.stops import (
    HELD_ERROR,
    PLACEHOLDER_CONTEXT,
    PLACEHOLDER_LENGTH,
    RECOVERY_PROMPT,
    RECOVERY_PROMPT_CUT,
    RECOVERY_PROMPT_ERROR,
    STOP_CONTEXT,
    STOP_ERROR,
    STOP_LENGTH,
    TurnStops,
    hold_cut_call,
    is_recovery,
    is_truncated,
    stop_reason,
    strip_stopped,
)
from code_agent.tracing import LangfuseTracer
from code_agent.usage import Usage
from code_agent.worklog import WorkLog, session_context

FULL = "응답이 출력 한도(64,000토큰)에 걸려 잘렸다."
THINKING_ONLY = "생각만 하다 한도를 다 써서 답변도 도구 호출도 없다."


def truncated(text: str = "", *, out: int = 64000, reason: tuple[str, str] = ("finish_reason", "length"),
              calls: list | None = None) -> AIMessage:
    """az-quiz 세션의 첫 응답 모양: 출력 한도까지 생각만 하고 텍스트·도구 호출이 없다(OpenRouter)."""
    return AIMessage(content=text, tool_calls=calls or [], additional_kwargs={"reasoning_content": "퀴즈 150개를…"},
                     usage_metadata={"input_tokens": 100, "output_tokens": out, "total_tokens": 100 + out,
                                     "output_token_details": {"reasoning": out}},
                     response_metadata=dict([reason]))


def lower_budget(monkeypatch: pytest.MonkeyPatch, **limits: int) -> None:
    real = agent_module.budget_middleware
    monkeypatch.setattr(agent_module, "budget_middleware", lambda: real(**limits))


@pytest.mark.parametrize(("meta", "expected"), [
    ({"finish_reason": "length"}, True),  # OpenRouter·OpenAI 계열
    ({"stop_reason": "max_tokens"}, True),  # Anthropic
    ({"finish_reason": "stop"}, False),
    ({"finish_reason": "tool_calls"}, False),
    ({}, False),
])
def test_is_truncated(meta: dict, expected: bool) -> None:
    assert is_truncated(AIMessage(content="", response_metadata=meta)) is expected
    assert not is_truncated(ToolMessage(content="", tool_call_id="c", response_metadata=meta))


def test_thinking_only_truncation_is_retried_with_appended_notice(tmp_path: Path) -> None:
    step = truncated()
    h = Harness(tmp_path, [step, AIMessage(content="quizzes.js 를 나눠 썼다")])
    h.run("퀴즈 150개 만들어")
    first, retry = h.model.calls
    # 앞쪽 기록은 그대로다. 바뀌는 것은 방금 나온 잘린 응답(마지막 메시지)뿐이고 그 뒤에 안내가 붙는다
    assert retry[:len(first)] == first
    cut, notice = retry[len(first):]
    assert isinstance(cut, AIMessage) and cut.id == step.id  # 같은 id 로 제자리 교체
    assert "reasoning_content" not in cut.additional_kwargs and "reasoning_details" not in cut.additional_kwargs
    assert cut.text == PLACEHOLDER_LENGTH
    assert cut.response_metadata["finish_reason"] == "length"
    assert cut.response_metadata["code_agent_stripped"] == {"reasoning_chars": len("퀴즈 150개를…"),
                                                            "dropped_tool_calls": []}
    assert isinstance(notice, HumanMessage) and notice.text == RECOVERY_PROMPT and is_recovery(notice)
    final = h.agent.get_state({"configurable": {"thread_id": "t"}}).values["messages"][-1]
    assert final.text == "quizzes.js 를 나눠 썼다"


def test_retry_is_once_per_turn(tmp_path: Path) -> None:
    """두 번째도 잘리면 턴을 끝낸다(대본이 2단계라 세 번째 호출이 오면 ScriptExhausted). 다음 턴은 다시 1번 받는다."""
    h = Harness(tmp_path, [truncated(), truncated()])
    h.run("만들어")
    assert len(h.model.calls) == 2
    h.model.steps += [truncated(), AIMessage(content="끝")]
    h.run("계속해줘")
    assert len(h.model.calls) == 4
    assert sum(is_recovery(m) for m in h.model.calls[-1]) == 2


def test_cut_only_tool_call_is_held_then_retried(tmp_path: Path) -> None:
    """잘린 응답의 유일한 호출은 실행하지 않는다. 남는 유효 호출이 없으니 정리하고 다시 부른다."""
    h = Harness(tmp_path, [truncated(calls=[{"name": "ls", "args": {"path": "/"}, "id": "c1", "type": "tool_call"}]),
                           AIMessage(content="끝")])
    h.run()
    request = h.model.calls[-1]
    assert len(h.model.calls) == 2 and is_recovery(request[-1])
    assert not any(isinstance(m, ToolMessage) for m in request)  # ls 는 실행되지 않았다
    assert "ls 호출 인자가 잘려 실행하지 않았다" in request[-2].text


def test_retry_counts_against_model_call_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """복구는 모델 호출 상한이 이번 호출을 센 뒤에 돌고, 루프 입구로 돌아가 상한 검사를 거친다."""
    lower_budget(monkeypatch, model_calls=1)
    h = Harness(tmp_path, [truncated()])
    h.run()
    assert len(h.model.calls) == 1
    final = h.agent.get_state({"configurable": {"thread_id": "t"}}).values["messages"][-1]
    assert final.text.startswith("Model call limits exceeded")


def test_cli_warns_when_retry_is_truncated_again(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [truncated(), truncated()], "퀴즈 만들어", "/exit", tracer=tracer)
    app.run()
    text = out.getvalue()
    assert "안내를 덧붙여 한 번 다시 요청한다" in text  # 다시 부르는 순간에도 알린다
    warning = f"{FULL} 자동으로 한 번 다시 요청했지만 또 잘렸다. {THINKING_ONLY} '계속해줘'로 이어 갈 수 있다."
    assert warning in text
    assert text.index(warning) < text.index("모델 2회")  # 사용량 줄보다 먼저
    assert tracer.turns[0].ended["warnings"] == [warning]


def test_cli_notes_recovered_truncation(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [truncated(), AIMessage(content="다 썼다")], "만들어", "/exit", tracer=tracer)
    app.run()
    text = out.getvalue()
    assert "다 썼다" in text
    assert "응답 1개가 출력 한도(64,000토큰)에 걸려 잘렸다. 자동으로 한 번 다시 요청해 이어 갔다." in text
    assert tracer.turns[0].ended["output"] == "다 썼다"


def test_partial_answer_cut_by_anthropic_max_tokens(tmp_path: Path) -> None:
    steps = [truncated("앞부분", out=1234, reason=("stop_reason", "max_tokens")),
             truncated("또 앞부분", out=1234, reason=("stop_reason", "max_tokens"))]
    app, model, out, _ = make_app(tmp_path, steps, "설명해", "/exit")
    app.run()
    assert "응답이 출력 한도(1,234토큰)에 걸려 잘렸다. 자동으로 한 번 다시 요청했지만 또 잘렸다. 답변이 중간에 끊겼다." in out.getvalue()
    assert model.calls[1][-1].text == RECOVERY_PROMPT_CUT  # 답변이 끊긴 경우는 이어서 마무리하라는 안내


def test_normal_turn_has_no_warning(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [AIMessage(content="안녕", response_metadata={"finish_reason": "stop"})],
                              "하이", "/exit", tracer=tracer)
    app.run()
    assert "출력 한도" not in out.getvalue()
    assert tracer.turns[0].ended["warnings"] == []


@pytest.mark.parametrize(("limits", "steps", "message"), [
    ({"model_calls": 1}, [tool_call("ls", {"path": "/"}), AIMessage(content="끝")],
     "Model call limits exceeded: run limit (1/1)"),
    ({"tool_calls": 1}, [tool_calls(("ls", {"path": "/"}), ("ls", {"path": "/"}))],
     "Tool call limit reached: run limit exceeded (2/1 calls)."),
])
def test_budget_stop_is_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limits: dict, steps: list,
                              message: str) -> None:
    """예산 미들웨어의 종료 메시지는 model 노드가 아니라 미들웨어 노드에서 나온다. 전에는 화면에 보이지 않았다."""
    lower_budget(monkeypatch, **limits)
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, steps, "go", "/exit", mode="yolo", tracer=tracer)
    app.run()
    text = out.getvalue()
    assert message in text
    budget = f"예산 상한에 걸려 턴을 멈췄다({message.rstrip('.')}). '계속해줘'로 이어 갈 수 있다."
    assert budget in text
    assert tracer.turns[0].ended["output"] == message  # 작업 기록의 최종 답변도 이 메시지다
    assert tracer.turns[0].ended["warnings"] == [budget]


def test_turn_stops_ignores_subagents() -> None:
    stops = TurnStops()
    stops.on_part((("tools:1",), "updates", {"model": {"messages": [truncated()]}}))
    stops.on_part(((), "messages", (truncated(), {"langgraph_node": "model"})))
    assert stops.warnings() == []


def test_session_context_keeps_recovery_in_same_turn(tmp_path: Path) -> None:
    previous = [HumanMessage("퀴즈 만들어"), truncated(),
                HumanMessage(RECOVERY_PROMPT, name="truncation-recovery"),
                AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/q.js", "content": "x"},
                                                   "id": "w", "type": "tool_call"}]),
                ToolMessage(content="ok", tool_call_id="w"), AIMessage(content="됐다")]
    assert session_context(previous, None).splitlines() == [
        "- 같은 세션의 이전 턴 1개:", "  1. 요청: 퀴즈 만들어 · 도구 1개 · 바꾼 파일: /q.js"]


def test_worklog_warning_section_follows_answer() -> None:
    text = WorkLog().render(answer="", changes=[], diff="", warnings=[f"{FULL} {THINKING_ONLY}"])
    assert text.startswith(f"## 최종 답변\n(없음)\n\n## 경고\n- {FULL} {THINKING_ONLY}\n\n## 실행한 도구")


def test_real_sdk_marks_turn_as_warning(tmp_path: Path) -> None:
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from langfuse import LangfuseOtelSpanAttributes as A
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test-stops", secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
                      span_exporter=exporter)
    app, _, _, _ = make_app(tmp_path, [truncated(), truncated()], "만들어", "/exit",
                            tracer=LangfuseTracer(client, public_key="pk-lf-test-stops"))
    app.run()
    [root] = [s for s in exporter.get_finished_spans() if s.name == "turn"]
    attrs = dict(root.attributes)
    assert attrs[A.OBSERVATION_LEVEL] == "WARNING"
    assert FULL in attrs[A.OBSERVATION_STATUS_MESSAGE]
    assert f"## 경고\n- {FULL}" in attrs[A.OBSERVATION_OUTPUT]


def test_recovery_runs_after_budget_count_and_reenters_through_budget_check(tmp_path: Path) -> None:
    """실제 그래프 간선으로 순서를 고정한다. deepagents 가 사용자 미들웨어를 끼우는 자리가 바뀌면 여기서 잡힌다."""
    h = Harness(tmp_path, [])
    builder = h.agent.builder

    def next_after_model(node: str) -> str | None:
        targets = [end for start, end in builder.edges if start == node]
        targets += [end for spec in builder.branches.get(node, {}).values() for end in spec.ends]
        return next((t for t in targets if t.endswith(".after_model")), None)

    chain, node = [], "model"
    while (node := next_after_model(node)) is not None:
        chain.append(node)
    # after_model 은 역순으로 돈다. 복구가 루프 출구(마지막)이고 모델 호출 상한의 집계가 그보다 먼저다
    assert chain[-1] == "TruncationRecoveryMiddleware.after_model"
    assert "ModelCallLimitMiddleware.after_model" in chain[:-1]
    # 출구에서 jump_to "model" 은 루프 입구로 간다. 그 입구가 모델 호출 상한 검사다(도구 뒤에 돌아가는 곳과 같다)
    [exit_branch] = builder.branches["TruncationRecoveryMiddleware.after_model"].values()
    [tools_branch] = builder.branches["tools"].values()
    assert set(tools_branch.ends) == {"ModelCallLimitMiddleware.before_model"}
    assert {"tools", "ModelCallLimitMiddleware.before_model"} <= set(exit_branch.ends)


def test_approved_text_with_tool_call_is_shown_once(tmp_path: Path) -> None:
    """HITL after_model 은 승인 뒤 같은 응답을 다시 내보낸다. 예산 메시지로 오인해 노랗게 다시 찍으면 안 된다."""
    step = AIMessage(content="파일을 만든다", tool_calls=[
        {"name": "write_file", "args": {"file_path": "/a.txt", "content": "A"}, "id": "w1", "type": "tool_call"}])
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [step, AIMessage(content="끝")], "만들어", "y", "/exit", tracer=tracer)
    app.run()
    assert out.getvalue().count("파일을 만든다") == 1
    assert tracer.turns[0].ended["output"] == "끝"


def interrupted_then_next_turn(tmp_path: Path, first: list) -> tuple[str, list, FakeTracer]:
    """first 단계 뒤 execute 승인 프롬프트에서 Ctrl+C 로 끊고, 다음 턴을 돌린다. (두 번째 턴 화면, 두 번째 턴 모델 입력)."""
    execute = AIMessage(content="명령을 돌린다", tool_calls=[
        {"name": "execute", "args": {"command": "uv run pytest"}, "id": "x1", "type": "tool_call"}])
    tracer = FakeTracer()
    app, model, out, _ = make_app(tmp_path, [*first, execute, AIMessage(content="새 답")],
                                  "먼저", "다음", KeyboardInterrupt(), "또", "/exit", tracer=tracer)
    (app.workspace / "a.txt").write_text("UNIQUE-CONTENT", encoding="utf-8")
    app.run()
    text = out.getvalue()
    return text[text.index("중단했다"):], model.calls[-1], tracer


def test_history_rewrite_after_interrupted_approval_is_not_redrawn(tmp_path: Path) -> None:
    """승인 중 끊긴 턴 다음 턴에 PatchToolCalls 가 기록 전체를 다시 내보내도 옛 답변·도구 결과를 다시 찍지 않는다."""
    first = [tool_call("read_file", {"file_path": "/a.txt"}), AIMessage(content="옛 답변")]
    after, history, tracer = interrupted_then_next_turn(tmp_path, first)
    # 시나리오 확인: 끊긴 호출에 PatchToolCalls 가 결과를 채워 기록을 다시 썼다
    assert any(isinstance(m, ToolMessage) and "did not complete" in m.text for m in history)
    assert "새 답" in after
    for old in ("옛 답변", "명령을 돌린다", "UNIQUE-CONTENT"):
        assert old not in after
    assert tracer.turns[-1].ended["output"] == "새 답"


def test_old_recovery_notice_is_not_redrawn_or_recounted(tmp_path: Path) -> None:
    after, history, tracer = interrupted_then_next_turn(tmp_path, [AIMessage(content="옛 답변"), truncated()])
    assert sum(is_recovery(m) for m in history) == 1  # 옛 안내가 기록에 있다
    assert "안내를 덧붙여 한 번 다시 요청한다" not in after
    stops = TurnStops()
    stops.on_part(((), "updates", {"PatchToolCallsMiddleware.before_agent": {"messages": history}}))
    assert stops.recoveries == 0
    assert tracer.turns[-1].ended["warnings"] == []


# --- 멈춘 응답 정리(K1·K2·K3)와 정지 사유 3분류 ---

def stopped(text: str = "", reason: tuple[str, str] = ("finish_reason", "length"), *, calls: list | None = None,
            invalid: list | None = None) -> AIMessage:
    """OpenRouter 모양의 멈춘 응답. reasoning_content 와 reasoning_details 에 같은 생각이 있다."""
    message = truncated(text, reason=reason, calls=calls)
    message.additional_kwargs["reasoning_details"] = [{"type": "reasoning.text", "text": "퀴즈 150개를…",
                                                       "signature": "sig", "index": 0}]
    message.invalid_tool_calls = invalid or []
    return message


def half_written(call_id: str = "w1") -> dict:
    return {"name": "write_file", "args": '{"file_path": "/quizzes.js", "content": "const q = [', "id": call_id,
            "error": None, "type": "invalid_tool_call"}


@pytest.mark.parametrize(("meta", "expected"), [
    ({"finish_reason": "length"}, "length"), ({"stop_reason": "max_tokens"}, "max_tokens"),
    ({"finish_reason": "error"}, "error"), ({"stop_reason": "model_context_window_exceeded"},
                                            "model_context_window_exceeded"),
    ({"finish_reason": "stop"}, None), ({"stop_reason": "refusal"}, None), ({"finish_reason": "content_filter"}, None),
])
def test_stop_sets(meta: dict, expected: str | None) -> None:
    assert stop_reason(AIMessage(content="", response_metadata=meta)) == expected
    assert sum(expected in s for s in (STOP_LENGTH, STOP_ERROR, STOP_CONTEXT)) == (expected is not None)


def test_partial_text_kept_with_marker(tmp_path: Path) -> None:
    h = Harness(tmp_path, [stopped("앞부분 설명"), AIMessage(content="끝")])
    h.run()
    cut, notice = h.model.calls[1][-2:]
    assert cut.text == "앞부분 설명\n\n[출력 한도로 여기서 잘림]"
    assert notice.text == RECOVERY_PROMPT_CUT


def test_anthropic_thinking_blocks_removed() -> None:
    """서명이 있는 thinking 도, 잘려 서명이 빈 thinking(K1)도, redacted_thinking 도 뺀다. 텍스트 블록만 남긴다."""
    blocks = [{"type": "thinking", "thinking": "생각 A", "signature": "sig"},
              {"type": "thinking", "thinking": "생각 B", "signature": ""},
              {"type": "redacted_thinking", "data": "xx"}, {"type": "text", "text": "부분"}]
    meta = {"stop_reason": "max_tokens", "model_provider": "anthropic"}
    cut = strip_stopped(AIMessage(content=blocks, response_metadata=meta, id="a1"))
    assert cut is not None and cut.id == "a1"
    assert cut.content == "부분\n\n[출력 한도로 여기서 잘림]"
    assert cut.response_metadata["code_agent_stripped"]["reasoning_chars"] == len("생각 A") + len("생각 B")
    thinking_only = strip_stopped(AIMessage(content=blocks[:2], response_metadata=meta))
    assert thinking_only is not None and thinking_only.content == PLACEHOLDER_LENGTH


def test_second_truncation_stripped_without_retry(tmp_path: Path) -> None:
    """복구 뒤에도 잘리면 다시 부르지 않고 정리만 한다. 다음 턴 요청에는 잘린 생각이 하나도 실리지 않는다."""
    h = Harness(tmp_path, [stopped(), stopped(), AIMessage(content="이어서 했다")])
    h.run("만들어")
    assert len(h.model.calls) == 2
    h.run("계속해줘")
    request = h.model.calls[-1]
    assert not any(key in m.additional_kwargs for m in request for key in ("reasoning_content", "reasoning_details"))
    assert [m.text for m in request if isinstance(m, AIMessage)] == [PLACEHOLDER_LENGTH, PLACEHOLDER_LENGTH]


@pytest.mark.parametrize("finish", ["tool_calls", "stop"])
def test_normal_finish_tool_calls_untouched(tmp_path: Path, finish: str) -> None:
    """멈추지 않은(정상) 응답의 도구 호출은 그대로 실행하고 메시지도 바꾸지 않는다(회귀 방지)."""
    step = AIMessage(content="", response_metadata={"finish_reason": finish},
                     additional_kwargs={"reasoning_content": "생각"}, tool_calls=[
                         {"name": "write_file", "args": {"file_path": "/a.txt", "content": "A"}, "id": "w",
                          "type": "tool_call"}])
    h = Harness(tmp_path, [step, AIMessage(content="끝")], [APPROVE])
    h.run()
    assert (h.workspace / "a.txt").read_text(encoding="utf-8") == "A"
    [kept] = [m for m in h.model.calls[-1] if isinstance(m, AIMessage)]
    assert kept == step


def test_stripped_message_not_rendered(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [stopped(), AIMessage(content="끝")], "만들어", "/exit", tracer=tracer)
    app.run()
    assert "기록에서 뺐다" not in out.getvalue()
    assert tracer.turns[0].ended["output"] == "끝"


def test_backfill_usage_preserved() -> None:
    """정리해도 사용량은 그대로라 sessions.db·소급 채점의 집계가 바뀌지 않는다."""
    original = stopped()
    cut = strip_stopped(original)
    assert cut is not None and cut.usage_metadata == original.usage_metadata
    before, after = Usage(), Usage()
    before.add_message(original)
    after.add_message(cut)
    assert after == before and after.reasoning == 64000


def test_request_dicts_both_providers() -> None:
    """정리한 응답이 실제 요청 본문에서 생각·도구 호출 없는 텍스트 하나가 되는지(두 공급자)."""
    pytest.importorskip("langchain_openrouter")
    from langchain_anthropic.chat_models import _format_messages
    from langchain_openrouter import ChatOpenRouter

    cut = strip_stopped(stopped(invalid=[half_written()]))
    assert cut is not None
    retry = [HumanMessage("만들어"), cut, HumanMessage(RECOVERY_PROMPT, name="truncation-recovery")]
    dicts, _ = ChatOpenRouter(model="anthropic/claude-opus-5.5", api_key="k")._create_message_dicts(retry, None)
    [assistant] = [d for d in dicts if d["role"] == "assistant"]
    assert not {"reasoning", "reasoning_details", "tool_calls"} & assistant.keys()
    assert "write_file 호출 인자가 잘려 실행하지 않았다" in assistant["content"]
    _, formatted = _format_messages(retry, model="claude-opus-5-5")
    [assistant] = [m for m in formatted if m["role"] == "assistant"]
    content = assistant["content"]
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    assert [b["type"] for b in blocks] == ["text"]


def test_invalid_only_call_dropped(tmp_path: Path) -> None:
    """도구 인자를 쓰다 잘리고 유효한 호출이 없으면 잘린 호출을 뺀다. 가짜 결과도 붙지 않는다(K2)."""
    h = Harness(tmp_path, [stopped(invalid=[half_written()]), AIMessage(content="끝")])
    h.run()
    request = h.model.calls[1]
    cut = request[-2]
    assert isinstance(cut, AIMessage) and cut.invalid_tool_calls == [] and cut.tool_calls == []
    assert "write_file 호출 인자가 잘려 실행하지 않았다" in cut.text
    assert cut.response_metadata["code_agent_stripped"]["dropped_tool_calls"] == ["write_file"]
    assert not any(isinstance(m, ToolMessage) for m in request)


def test_mixed_calls_sanitized(tmp_path: Path) -> None:
    """유효한 호출이 남으면 메시지는 두고 잘린 호출의 인자만 "{}" 로. langchain 이 '인자가 잘렸다' 결과를 붙인다.

    마지막 유효 호출(c3, 쓰다가 멈춘 것)은 실행하지 않고, 그 앞의 ls(c1)만 실행한다.
    """
    step = stopped(calls=[{"name": "ls", "args": {"path": "/"}, "id": "c1", "type": "tool_call"},
                          {"name": "ls", "args": {"path": "/sub"}, "id": "c3", "type": "tool_call"}],
                   invalid=[half_written("w2")])
    h = Harness(tmp_path, [step, AIMessage(content="끝")])
    h.run()
    request = h.model.calls[1]
    [ai] = [m for m in request if isinstance(m, AIMessage)]
    assert ai.id == step.id and [c["id"] for c in ai.tool_calls] == ["c1"]
    assert ai.additional_kwargs["reasoning_content"] == "퀴즈 150개를…"  # 생각은 그대로
    assert [(c["id"], c["args"]) for c in ai.invalid_tool_calls] == [("w2", "{}"), ("c3", "{}")]
    results = {m.tool_call_id: m.text for m in request if isinstance(m, ToolMessage)}
    assert set(results) == {"c1", "w2", "c3"}
    assert "malformed or truncated" in results["w2"] and "malformed or truncated" in results["c3"]


def test_hold_cut_call() -> None:
    calls = [{"name": "read_file", "args": {"file_path": "/a"}, "id": "r", "type": "tool_call"},
             {"name": "write_file", "args": {"file_path": "/b", "content": "1\n2"}, "id": "w", "type": "tool_call"}]
    held = hold_cut_call(stopped(calls=calls))
    assert held is not None and [c["id"] for c in held.tool_calls] == ["r"]
    assert [(c["id"], c["args"]) for c in held.invalid_tool_calls] == [("w", "{}")]
    assert held.response_metadata["code_agent_held_call"] == "write_file"
    normal = AIMessage(content="", tool_calls=calls, response_metadata={"finish_reason": "tool_calls"})
    assert hold_cut_call(normal) is None and hold_cut_call(stopped()) is None


def cut_call(name: str, args: dict, *, before: list | None = None) -> AIMessage:
    """출력 한도로 멈췄지만 langchain 이 잘린 인자를 고쳐 유효한 호출이 된 응답(실측 모양). before 는 그 앞의 완결된 호출."""
    return stopped(calls=[*(before or []), {"name": name, "args": args, "id": "cut", "type": "tool_call"}])


def test_cut_write_file_is_not_asked_in_manual_mode(tmp_path: Path) -> None:
    """잘린 write_file 은 승인을 묻지도 실행하지도 않는다. 모델이 다시 쓴 호출만 묻는다."""
    steps = [cut_call("write_file", {"file_path": "/numbers.txt", "content": "1\n2\n3\n4"}),
             tool_call("write_file", {"file_path": "/numbers.txt", "content": "1\n2\n3\n4\n5\n"}), AIMessage(content="끝")]
    app, _, out, script = make_app(tmp_path, steps, "만들어", "y", "/exit")
    app.run()
    assert script.prompts.count("[y] 승인 / [n] 거부 › ") == 1
    assert (app.workspace / "numbers.txt").read_text(encoding="utf-8") == "1\n2\n3\n4\n5\n"
    assert out.getvalue().count("승인 요청: write_file") == 1
    assert f"write_file 호출은 {HELD_ERROR}." in out.getvalue()


@pytest.mark.parametrize(("name", "args"), [
    ("write_file", {"file_path": "/new.txt", "content": "절반"}),
    ("edit_file", {"file_path": "/app.py", "old_string": "x = 1", "new_string": "x = 2 if"}),
])
def test_cut_file_call_changes_nothing_in_auto_edit(tmp_path: Path, name: str, args: dict) -> None:
    app, _, _, _ = make_app(tmp_path, [cut_call(name, args), AIMessage(content="끝")], "고쳐", "/exit",
                            mode="auto-edit")
    (app.workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    app.run()
    assert not (app.workspace / "new.txt").exists()
    assert (app.workspace / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_parallel_complete_call_runs_and_cut_call_is_blocked(tmp_path: Path) -> None:
    read = {"name": "read_file", "args": {"file_path": "/a.txt"}, "id": "r1", "type": "tool_call"}
    h = Harness(tmp_path, [cut_call("write_file", {"file_path": "/b.txt", "content": "절"}, before=[read]),
                           AIMessage(content="끝")])
    (h.workspace / "a.txt").write_text("A-CONTENT", encoding="utf-8")
    h.run()
    results = {m.tool_call_id: m.text for m in h.model.calls[-1] if isinstance(m, ToolMessage)}
    assert "A-CONTENT" in results["r1"]
    assert "malformed or truncated" in results["cut"]
    assert not (h.workspace / "b.txt").exists()
    assert h.requests == []  # write_file 의 승인도 묻지 않았다


def test_cut_execute_never_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from code_agent.backend import WindowsShellBackend

    ran: list[str] = []
    real = WindowsShellBackend.execute

    def counting(self, command: str, *args, **kwargs):
        ran.append(command)
        return real(self, command, *args, **kwargs)

    monkeypatch.setattr(WindowsShellBackend, "execute", counting)
    app, _, _, _ = make_app(tmp_path, [cut_call("execute", {"command": "rm -rf build && echo x > made.txt"}),
                                       AIMessage(content="끝")], "정리해", "/exit", mode="yolo")
    app.run()
    assert ran == [] and not (app.workspace / "made.txt").exists()


def test_openrouter_error_chunk_finishes_silently() -> None:
    """choices 가 있는 오류 청크는 예외 없이 finish_reason="error" 로 끝난다. 그래서 STOP_ERROR 로 다룬다."""
    pytest.importorskip("langchain_openrouter")
    from langchain_openrouter import ChatOpenRouter

    class Chunk:
        def __init__(self, data: dict) -> None:
            self.data = data

        def model_dump(self, by_alias: bool = False) -> dict:
            return self.data

    class Chat:
        def send(self, **params):
            yield Chunk({"id": "gen-1", "model": "anthropic/claude-opus-5.5",
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "하던 "}}]})
            yield Chunk({"id": "gen-1", "error": {"message": "upstream", "code": 502},
                         "choices": [{"index": 0, "delta": {"content": "중"}, "finish_reason": "error"}]})

    class Client:
        chat = Chat()

    model = ChatOpenRouter(model="anthropic/claude-opus-5.5", api_key="k", streaming=True)
    model.client = Client()
    result = model.invoke("hi")
    assert result.text == "하던 중" and stop_reason(result) == "error"


def test_error_finish_recovered_once(tmp_path: Path) -> None:
    error = ("finish_reason", "error")
    h = Harness(tmp_path, [stopped("하던 중", reason=error), AIMessage(content="끝")])
    h.run()
    cut, notice = h.model.calls[1][-2:]
    assert cut.text == "하던 중\n\n[공급자 오류로 여기서 끊김]" and notice.text == RECOVERY_PROMPT_ERROR
    # 출력 한도와 복구 1회를 함께 쓴다: 잘린 뒤 오류가 나면 더 부르지 않는다
    (tmp_path / "b").mkdir()
    h = Harness(tmp_path / "b", [stopped(), stopped(reason=error)])
    h.run()
    assert len(h.model.calls) == 2
    (tmp_path / "c").mkdir()
    app, _, out, _ = make_app(tmp_path / "c", [stopped(reason=error), stopped(reason=error)], "해줘", "/exit")
    app.run()
    text = out.getvalue()
    assert "응답이 공급자 오류로 중단됐다. 안내를 덧붙여 한 번 다시 요청한다." in text
    assert ("응답이 공급자 오류로 중단됐다. 자동으로 한 번 다시 요청했지만 또 멈췄다. 답변도 도구 호출도 없다. "
            "'계속해줘'로 이어 갈 수 있다.") in text


def test_context_window_not_retried(tmp_path: Path) -> None:
    context = ("stop_reason", "model_context_window_exceeded")
    tracer = FakeTracer()
    app, model, out, _ = make_app(tmp_path, [stopped(reason=context)], "해줘", "/exit", tracer=tracer)
    app.run()
    assert len(model.calls) == 1
    final = app.agent.get_state(app.config).values["messages"][-1]
    assert final.text == PLACEHOLDER_CONTEXT and "reasoning_content" not in final.additional_kwargs
    warning = "응답이 컨텍스트 창이 가득 차 중단됐다. /new 로 새로 시작하거나 요약한 뒤 이어 가라."
    assert warning in out.getvalue() and tracer.turns[0].ended["warnings"] == [warning]
