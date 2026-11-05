"""OpenRouter 캐시: 요청 본문에 session_id 와(Anthropic 이면) cache_control 이 실리는지. SDK 클라이언트만 가짜로 바꾼다."""
from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE
from code_agent.caching import is_anthropic_on_openrouter
from code_agent.models import make_model
from code_agent.settings import Settings
from code_agent.usage import UsageCallback

pytest.importorskip("langchain_openrouter")


class _Chunk:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, by_alias: bool = False) -> dict:
        return self.data


class FakeOpenRouter:
    """openrouter SDK 의 chat.send 대역. 받은 인자를 남기고 스트리밍 청크 세 개(본문, 끝, usage)를 돌려준다."""

    def __init__(self, model: str) -> None:
        self.model, self.sent = model, []
        self.chat = self

    def send(self, messages: list[dict], **params) -> list[_Chunk]:
        self.sent.append(params)
        head = {"id": "gen-1", "model": self.model}
        return [_Chunk({**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "안녕"},
                                              "finish_reason": None}]}),
                _Chunk({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                _Chunk({**head, "choices": [], "usage": {
                    "prompt_tokens": 5000, "completion_tokens": 3, "total_tokens": 5003, "cost": 0.0025,
                    "prompt_tokens_details": {"cached_tokens": 4800, "cache_write_tokens": 0}}})]


def run_once(tmp_path: Path, model_id: str) -> tuple[FakeOpenRouter, UsageCallback]:
    (tmp_path / "ws").mkdir()
    settings = Settings(model=f"openrouter:{model_id}", openrouter_api_key="k", data_dir=tmp_path / "data")
    model = make_model(settings)
    fake = FakeOpenRouter(model_id)
    model.client = fake
    agent = build_agent(settings, tmp_path / "ws", InMemorySaver(), model=model)
    meter = UsageCallback()
    run_turn(agent, {"messages": [{"role": "user", "content": "hi"}]},
             {"configurable": {"thread_id": "t1"}, "callbacks": [meter]},
             decide=lambda requests: [APPROVE for _ in requests], on_part=lambda part: None)
    return fake, meter


def test_anthropic_model_gets_cache_control_and_session(tmp_path: Path) -> None:
    fake, meter = run_once(tmp_path, "anthropic/claude-opus-5.5")
    [params] = fake.sent
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["session_id"] == "code-agent-t1"
    assert params["reasoning"] == {"effort": "high"}
    # 스트리밍 usage 청크의 캐시 읽기와 비용이 사용량 집계까지 이어진다
    assert (meter.usage.calls, meter.usage.input_tokens, meter.usage.cache_read) == (1, 5000, 4800)
    assert meter.usage.cost == pytest.approx(0.0025)


def test_other_model_gets_session_only(tmp_path: Path) -> None:
    fake, _ = run_once(tmp_path, "openai/gpt-5.6-luna")
    [params] = fake.sent
    assert params["session_id"] == "code-agent-t1"
    assert "cache_control" not in params


@pytest.mark.parametrize(("name", "expected"), [
    ("anthropic/claude-opus-5.5", True), ("~anthropic/claude-fable-latest", True),
    ("anthropic/claude-opus-5.5:batch", True), ("openai/gpt-5.6-luna", False), ("google/gemini-3", False),
])
def test_is_anthropic_on_openrouter(name: str, expected: bool) -> None:
    assert is_anthropic_on_openrouter(name) is expected
