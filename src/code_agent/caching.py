"""OpenRouter 프롬프트 캐시. deepagents 는 ChatAnthropic 에만 캐시를 붙이므로 OpenRouter 경로는 여기서 켠다.

- Anthropic 모델: 요청 최상위 cache_control 하나로 자동 캐시된다. 캐시 지점이 대화 끝을 따라 이동한다.
- session_id: 첫 요청부터 같은 공급자로 라우팅돼(sticky routing) 캐시가 맞는다. 없으면 캐시 적중이 한 번 나온 뒤에야 고정된다.
근거: https://openrouter.ai/docs/guides/best-practices/prompt-caching
"""

from __future__ import annotations

from collections.abc import Callable

from deepagents._models import get_model_provider  # 0.7.32 고정 (agent.py 와 같은 이유)
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.config import get_config

CACHE_CONTROL = {"type": "ephemeral"}  # 5분 TTL. 쓰기 1.25배, 읽기 0.1배 가격


def is_anthropic_on_openrouter(model_name: str) -> bool:
    return model_name.lstrip("~").startswith("anthropic/")


class OpenRouterCacheMiddleware(AgentMiddleware):
    """OpenRouter 모델 호출에 thread 별 session_id 와(Anthropic 이면) cache_control 을 더한다."""

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        if get_model_provider(request.model) != "openrouter":
            return handler(request)
        settings = dict(request.model_settings)
        thread_id = _thread_id()
        if thread_id:
            settings["session_id"] = f"code-agent-{thread_id}"
        if is_anthropic_on_openrouter(str(getattr(request.model, "model_name", ""))):
            settings["cache_control"] = CACHE_CONTROL
        return handler(request.override(model_settings=settings))


def _thread_id() -> str | None:
    try:
        return (get_config().get("configurable") or {}).get("thread_id")
    except RuntimeError:  # 그래프 밖에서 불렸다
        return None
