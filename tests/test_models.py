"""모델 생성: 요청 본문에 effort 가 들어가고 temperature 는 빠지는지. 네트워크 호출은 없다."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from code_agent.models import make_model
from code_agent.settings import Settings


def test_anthropic_payload_sets_effort_without_temperature() -> None:
    model = make_model(Settings(anthropic_api_key="sk-test", effort="xhigh"))
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["output_config"] == {"effort": "xhigh"}
    assert "temperature" not in payload


@pytest.mark.parametrize(("effort", "sent"), [("high", "high"), ("low", "low"), ("max", "xhigh")])
def test_openrouter_passes_effort_as_reasoning(effort: str, sent: str) -> None:
    pytest.importorskip("langchain_openrouter")
    model = make_model(Settings(model="openrouter:anthropic/claude-opus-5.5", openrouter_api_key="k", effort=effort))
    params = model._default_params
    assert params["model"] == "anthropic/claude-opus-5.5"
    assert params["reasoning"] == {"effort": sent}
    assert "temperature" not in params


@pytest.mark.parametrize(("settings", "message"), [
    (Settings(model="claude-opus-5-5"), "형식"),
    (Settings(model="openai:gpt-5"), "공급자"),
    (Settings(model="anthropic:claude-opus-5-5"), "ANTHROPIC_API_KEY"),
    (Settings(model="openrouter:anthropic/claude-opus-5.5"), "OPENROUTER_API_KEY"),
])
def test_bad_model_settings(settings: Settings, message: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=message):
        make_model(settings)
