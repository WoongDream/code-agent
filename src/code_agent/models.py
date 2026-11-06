"""채팅 모델 생성. 모델 이름은 "anthropic:<id>" 또는 "openrouter:<id>" 형식만 받는다."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from code_agent.settings import Settings

_API_URL = "https://api.anthropic.com"  # 셸의 ANTHROPIC_BASE_URL 로 키가 다른 곳에 가지 않게 고정한다
MAX_TOKENS = 64000  # 응답 하나의 출력 한도(thinking 포함). 항상 스트리밍하므로 HTTP 제한 시간에 걸리지 않는다
_TIMEOUT_S = 600.0
_OPENROUTER_EFFORT = {"max": "xhigh"}


def make_model(settings: Settings) -> BaseChatModel:
    provider, _, name = settings.model.partition(":")
    if not name:
        raise ValueError(f"모델은 'anthropic:<id>' 또는 'openrouter:<id>' 형식이어야 합니다: {settings.model!r}")
    if provider == "anthropic":
        return _anthropic(settings, name)
    if provider == "openrouter":
        return _openrouter(settings, name)
    raise ValueError(f"지원하지 않는 모델 공급자입니다: {provider!r}")


def _anthropic(settings: Settings, name: str) -> BaseChatModel:
    """temperature·강제 tool_choice 는 보내지 않는다(Opus 5.5 는 400). 깊이는 effort 로만 정한다.

    thinking 은 langchain-anthropic 1.7.7 이 모델 프로필에 따라 {"type": "adaptive", "display": "summarized"} 로 채운다.
    """
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 가 미설정입니다. .env.example 을 .env 로 복사해 직접 채우세요.")
    return ChatAnthropic(
        model=name, api_key=settings.anthropic_api_key, base_url=_API_URL,
        reasoning_effort=settings.effort, max_tokens=MAX_TOKENS, streaming=True,
        timeout=_TIMEOUT_S, max_retries=2,
    )


def _openrouter(settings: Settings, name: str) -> BaseChatModel:
    """ChatOpenRouter 의 timeout 단위는 밀리초다. effort 는 OpenRouter 의 reasoning.effort 로 넘긴다(max 는 없어 xhigh).

    reasoning_details 는 langchain-openrouter 가 응답에서 받아 다음 요청에 되돌려 준다(도구 루프의 thinking 유지).
    """
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as e:
        raise RuntimeError("OpenRouter 모델은 `uv sync --extra openrouter` 로 설치한 뒤 쓸 수 있습니다.") from e
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY 가 미설정입니다. .env.example 을 .env 로 복사해 직접 채우세요.")
    return ChatOpenRouter(
        model=name, api_key=settings.openrouter_api_key, reasoning={"effort": _OPENROUTER_EFFORT.get(settings.effort,
                                                                                                  settings.effort)},
        max_tokens=MAX_TOKENS, streaming=True, max_retries=2, timeout=round(_TIMEOUT_S * 1000),
    )
