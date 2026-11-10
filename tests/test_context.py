"""컨텍스트 한도: 모델 프로필 채우기, CODE_AGENT_CONTEXT_TOKENS, 요약 자리 대체(인자 잘림 끔),
기록이 덧붙이기만 되는지.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from deepagents.middleware import SummarizationMiddleware
from deepagents.middleware.summarization import compute_summarization_defaults
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage
from test_agent import Harness

from code_agent import settings as settings_module
from code_agent.approval import APPROVE
from code_agent.models import MAX_TOKENS, make_model, with_context_limit
from code_agent.settings import DEFAULT_CONTEXT_TOKENS, Settings, load_settings


def test_openrouter_opus55_gets_context_limit() -> None:
    """langchain-openrouter 0.2.8 데이터에 5.5 계열이 없어 프로필이 비었다.
    그러면 deepagents 기본값이 메시지 20개부터 도구 인자를 잘랐다."""
    pytest.importorskip("langchain_openrouter")
    model = make_model(Settings(model="openrouter:anthropic/claude-opus-5.5",
                                openrouter_api_key="k"))
    assert model.profile == {"max_input_tokens": DEFAULT_CONTEXT_TOKENS,
                             "max_output_tokens": MAX_TOKENS}
    assert compute_summarization_defaults(model)["trigger"] == ("fraction", 0.85)


def test_known_profile_keeps_other_keys() -> None:
    pytest.importorskip("langchain_openrouter")
    from langchain_openrouter import ChatOpenRouter

    raw = ChatOpenRouter(model="anthropic/claude-opus-5", api_key="k").profile or {}
    assert raw["max_input_tokens"] == 1_000_000
    model = make_model(Settings(model="openrouter:anthropic/claude-opus-5",
                                openrouter_api_key="k"))
    assert model.profile == {**raw, "max_input_tokens": DEFAULT_CONTEXT_TOKENS}


def test_anthropic_profile_capped() -> None:
    model = make_model(Settings(anthropic_api_key="sk-test"))  # Anthropic 직결 Opus 5.5 는 1M 창
    assert (model.profile or {})["max_input_tokens"] == DEFAULT_CONTEXT_TOKENS
    wide = make_model(Settings(anthropic_api_key="sk-test", context_tokens=1_000_000))
    assert (wide.profile or {})["max_input_tokens"] == 1_000_000
    # 알려진 창이 설정보다 작으면 작은 쪽. 원본은 바꾸지 않는다
    small = ScriptedChatModel(steps=[], profile={"max_input_tokens": 100_000})
    assert (with_context_limit(small, 200_000).profile or {})["max_input_tokens"] == 100_000
    assert small.profile == {"max_input_tokens": 100_000}


@pytest.mark.parametrize(("raw", "expected"), [
    (None, DEFAULT_CONTEXT_TOKENS), ("150000", 150_000), ("150,000", 150_000), ("20000", 20_000),
    ("1000000", 1_000_000), ("abc", None), ("1000", None), ("2000000", None), ("-5", None),
    ("2²", None), ("１５００００", None),  # isdigit 이지만 int 가 안 되거나 ASCII 가 아니다
])
def test_context_tokens_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str | None,
                                expected: int | None) -> None:
    monkeypatch.setattr(settings_module, "REPO_ROOT", tmp_path)  # 저장소 .env 를 읽지 않는다
    if raw is None:
        monkeypatch.delenv("CODE_AGENT_CONTEXT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("CODE_AGENT_CONTEXT_TOKENS", raw)
    if expected is None:
        with pytest.raises(ValueError, match="CODE_AGENT_CONTEXT_TOKENS 는 20,000~1,000,000"):
            load_settings(tmp_path)
    else:
        assert load_settings(tmp_path).context_tokens == expected


def test_summarization_slot_replaced_without_arg_truncation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepagents.graph

    seen: list[list] = []
    real = deepagents.graph.create_agent

    def spy(*args, **kwargs):
        seen.append(list(kwargs["middleware"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(deepagents.graph, "create_agent", spy)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    from langgraph.checkpoint.memory import InMemorySaver

    from code_agent.agent import build_agent

    build_agent(Settings(data_dir=tmp_path / "data", context_tokens=150_000), workspace,
                InMemorySaver(), model=ScriptedChatModel(steps=[]))
    [summarizers] = [[m for m in stack if m.name == "SummarizationMiddleware"]
                     for stack in seen[-1:]]
    [ours] = summarizers  # core 자리를 대신했다(둘이 아니다)
    assert isinstance(ours, SummarizationMiddleware)
    assert ours._truncate_args_trigger is None  # 도구 인자 잘림을 끈다
    assert ((ours._lc_helper.trigger, ours._lc_helper.keep)
            == (("tokens", 127_500), ("tokens", 15_000)))
    # 기본값 4,000 이면 버릴 기록의 끝 4K 만 요약된다
    assert ours._lc_helper.trim_tokens_to_summarize is None
    names = [m.name for m in seen[-1]]
    # core 자리 그대로
    assert names.index("SummarizationMiddleware") < names.index("TruncationRecoveryMiddleware")


def test_history_is_append_only_over_30_messages(tmp_path: Path) -> None:
    """프로필 없는 모델로 큰 write_file 을 15번 해도 앞쪽 기록이 바뀌지 않는다.

    전에는 deepagents 기본값이 메시지 20개부터 옛 write_file 인자를 잘라,
    캐시와 Opus 5.5 의 보존된 thinking 이 깨졌다.
    """
    steps = [tool_call("write_file", {"file_path": f"/f{i}.txt", "content": f"{i}" * 3000})
             for i in range(15)]
    h = Harness(tmp_path, [*steps, AIMessage(content="끝")], [APPROVE] * 15)
    assert h.model.profile is None
    h.run()
    calls = h.model.calls
    assert len(calls) == 16 and len(calls[-1]) > 30
    for before, after in pairwise(calls):
        assert after[:len(before)] == before
