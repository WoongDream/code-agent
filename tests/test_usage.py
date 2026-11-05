"""사용량 집계: 메시지 합산, 요약 문구, 콜백으로 에이전트 실행 전체를 세는지."""
from __future__ import annotations

from pathlib import Path

from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE
from code_agent.settings import Settings
from code_agent.usage import Usage, UsageCallback


def ai(text: str = "", *, inp: int, out: int, cache_read: int = 0, cache_write: int = 0, reasoning: int = 0,
       cost: float | None = None, tool_calls: list | None = None) -> AIMessage:
    return AIMessage(
        content=text, tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
                        "output_token_details": {"reasoning": reasoning}},
        response_metadata={} if cost is None else {"cost": cost})


def test_add_message_sums_details() -> None:
    usage = Usage()
    usage.add_message(ai(inp=1000, out=50, cache_write=900, reasoning=10, cost=0.01))
    usage.add_message(ai(inp=1200, out=70, cache_read=900, cost=0.002))
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 2200, 120)
    assert (usage.cache_read, usage.cache_write, usage.reasoning) == (900, 900, 10)
    assert round(usage.cost, 6) == 0.012
    assert usage.summary() == "모델 2회 · 입력 2,200 (캐시 읽기 900 · 쓰기 900) · 출력 120 · $0.0120"


def test_summary_without_cache_or_cost() -> None:
    usage = Usage()
    usage.add_message(AIMessage(content="usage 없음"))
    usage.add_message(ai(inp=10, out=2))
    assert usage.summary() == "모델 2회 · 입력 10 · 출력 2"


def test_partial_cost_is_marked() -> None:
    usage = Usage()
    usage.add_message(ai(inp=10, out=2, cost=0.5))
    usage.add_message(ai(inp=10, out=2))
    assert usage.summary().endswith("$0.5000 (1/2회분)")


def test_usage_add() -> None:
    total = Usage(calls=1, input_tokens=5, cost=0.1) + Usage(calls=2, input_tokens=7, cost=0.2, priced_calls=1)
    assert (total.calls, total.input_tokens, round(total.cost, 6), total.priced_calls) == (3, 12, 0.3, 1)


def test_callback_counts_every_model_call_in_a_turn(tmp_path: Path) -> None:
    """도구 호출 응답과 재개 후 마지막 응답, 두 번의 모델 호출을 모두 센다."""
    (tmp_path / "ws").mkdir()
    call = {"name": "execute", "args": {"command": "echo hi"}, "id": "call-1", "type": "tool_call"}
    model = ScriptedChatModel(steps=[ai(inp=3000, out=40, cost=0.02, tool_calls=[call]),
                                     ai("끝", inp=3100, out=10, cache_read=2900, cost=0.003)])
    agent = build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", InMemorySaver(), model=model)
    meter = UsageCallback()
    run_turn(agent, {"messages": [{"role": "user", "content": "go"}]},
             {"configurable": {"thread_id": "t"}, "callbacks": [meter]},
             decide=lambda requests: [APPROVE for _ in requests], on_part=lambda part: None)
    assert (meter.usage.calls, meter.usage.input_tokens, meter.usage.cache_read) == (2, 6100, 2900)
    assert round(meter.usage.cost, 6) == 0.023
