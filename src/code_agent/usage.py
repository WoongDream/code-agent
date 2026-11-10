"""모델 사용량 집계. 스트림이 아니라 콜백으로 세어
서브에이전트·요약 미들웨어의 모델 호출까지 잡는다.

세션 상태의 메시지로 세면 요약이 기록을 바꾼 뒤 줄어들므로,
턴마다 sessions.db 에 따로 남긴다(cli.Sessions).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, fields
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


@dataclass
class Usage:
    """모델 호출 사용량. 더하기(+)로 합치고, 필드마다 sessions.db 의 열이 하나씩 있다.

    Attributes:
        calls: 채팅 모델 호출 수.
        input_tokens: 캐시 읽기·쓰기를 포함한 전체 입력 토큰.
        output_tokens: reasoning 을 포함한 전체 출력 토큰.
        cache_read: 그중 캐시에서 읽은 입력 토큰.
        cache_write: 그중 캐시에 쓴 입력 토큰.
        reasoning: 출력 중 생각(reasoning) 토큰.
        cost: OpenRouter 가 알려 준 비용(USD). Anthropic 직접 호출은 비용을 주지 않는다.
        priced_calls: cost 가 있던 호출 수. calls 보다 적으면 cost 는 일부 호출만의 합이다.
    """

    calls: int = 0
    input_tokens: int = 0  # 캐시 읽기·쓰기를 포함한 전체 입력
    output_tokens: int = 0  # reasoning 을 포함한 전체 출력
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    cost: float = 0.0  # OpenRouter 가 알려 준 USD. Anthropic 직접 호출은 비용을 주지 않는다
    priced_calls: int = 0

    def add_message(self, message: BaseMessage) -> None:
        """AI 메시지 하나의 usage_metadata 와 응답 메타데이터의 cost 를 더한다."""
        usage = getattr(message, "usage_metadata", None) or {}
        details_in = usage.get("input_token_details") or {}
        details_out = usage.get("output_token_details") or {}
        self.calls += 1
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        self.cache_read += details_in.get("cache_read") or 0
        self.cache_write += details_in.get("cache_creation") or 0
        self.reasoning += details_out.get("reasoning") or 0
        cost = (getattr(message, "response_metadata", None) or {}).get("cost")
        if isinstance(cost, int | float):
            self.cost += float(cost)
            self.priced_calls += 1

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{f.name: getattr(self, f.name) + getattr(other, f.name)
                        for f in fields(self)})

    def summary(self) -> str:
        """'모델 N회 · 입력 … · 출력 … · $비용' 한 줄. 턴 끝과 /cost 에 보인다."""
        cache = (f" (캐시 읽기 {self.cache_read:,} · 쓰기 {self.cache_write:,})"
                 if self.cache_read or self.cache_write else "")
        parts = [f"모델 {self.calls}회", f"입력 {self.input_tokens:,}{cache}",
                 f"출력 {self.output_tokens:,}"]
        if self.priced_calls:
            partial = ("" if self.priced_calls == self.calls
                       else f" ({self.priced_calls}/{self.calls}회분)")
            parts.append(f"${self.cost:.4f}{partial}")
        return " · ".join(parts)


class UsageCallback(BaseCallbackHandler):
    """config["callbacks"] 에 넣으면 그 실행 안의 모든 채팅 모델 호출이 끝날 때 사용량을 더한다."""

    def __init__(self) -> None:
        self.usage = Usage()
        self._lock = threading.Lock()  # 병렬 도구 호출 안의 서브에이전트가 동시에 끝날 수 있다

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """채팅 모델 호출이 끝날 때마다 응답 메시지의 사용량을 더한다."""
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is not None:
                    with self._lock:
                        self.usage.add_message(message)
