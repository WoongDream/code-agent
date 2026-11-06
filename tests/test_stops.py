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
from code_agent.stops import RECOVERY_PROMPT, TurnStops, is_recovery, is_truncated
from code_agent.tracing import LangfuseTracer
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
    h = Harness(tmp_path, [truncated(), AIMessage(content="quizzes.js 를 나눠 썼다")])
    h.run("퀴즈 150개 만들어")
    first, retry = h.model.calls
    # 기록은 고치지 않고 덧붙이기만 한다: 잘린 응답(생각 포함)이 그대로 있고 그 뒤에 안내가 붙는다
    assert retry[:len(first)] == first
    cut, notice = retry[len(first):]
    assert isinstance(cut, AIMessage) and cut.additional_kwargs["reasoning_content"] == "퀴즈 150개를…"
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


def test_truncated_response_with_tool_calls_is_not_retried(tmp_path: Path) -> None:
    h = Harness(tmp_path, [truncated(calls=[{"name": "ls", "args": {"path": "/"}, "id": "c1", "type": "tool_call"}]),
                           AIMessage(content="끝")])
    h.run()
    assert len(h.model.calls) == 2
    assert not any(is_recovery(m) for m in h.model.calls[-1])


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
    assert "결과물을 바로 쓰라고 안내하고 한 번 다시 요청한다" in text  # 다시 부르는 순간에도 알린다
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
    app, _, out, _ = make_app(tmp_path, steps, "설명해", "/exit")
    app.run()
    assert "응답이 출력 한도(1,234토큰)에 걸려 잘렸다. 자동으로 한 번 다시 요청했지만 또 잘렸다. 답변이 중간에 끊겼다." in out.getvalue()


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
