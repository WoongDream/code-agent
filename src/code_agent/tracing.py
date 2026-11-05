"""Langfuse 트레이스. 사용자 메시지 하나가 트레이스 하나다(루트 observation "turn").

- 루트 아래에 LangChain CallbackHandler 가 그래프 노드·미들웨어·모델 호출·도구 실행을 기록한다.
  승인 대기 후 재개(Command(resume=...))도 같은 핸들러를 쓰므로 같은 턴 아래에 이어진다.
- 승인 결정은 이 모듈이 event 로 남긴다(도구, 호출 요약, approve/reject, 사람이 정했는지, 거부 사유).
- 턴이 끝나면 UsageCallback 합계(OpenRouter 실제 비용 포함)를 루트 metadata 에 남긴다. 핸들러는 토큰만 넘기고
  비용은 Langfuse 가 모델 정의로 계산하므로, OpenRouter 모델 이름에는 비용이 비어 있을 수 있다.

LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 가 둘 다 있고 langfuse extra 가 설치돼 있을 때만 켜진다.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Protocol

from code_agent.approval import reject_reason
from code_agent.render import describe_call
from code_agent.settings import Settings
from code_agent.usage import Usage

TIMEOUT_S = 5  # 내보내기 요청 제한 시간. 종료 시 flush 가 이보다 오래 붙잡지 않게 한다


class TurnTrace(Protocol):
    callbacks: list

    def decision(self, action: dict, decision: dict, auto: bool) -> None: ...

    def end(self, *, output: str, usage: Usage, error: str | None = None) -> None: ...


class Tracer(Protocol):
    def turn(self, text: str, *, session_id: str, tags: list[str]) -> Any: ...  # ContextManager[TurnTrace]

    def shutdown(self) -> None: ...


def langfuse_installed() -> bool:
    return importlib.util.find_spec("langfuse") is not None


def make_tracer(settings: Settings) -> LangfuseTracer | None:
    """키가 없거나 extra 가 없으면 None. 키는 인자로만 넘겨 os.environ 을 바꾸지 않는다."""
    if not settings.langfuse_enabled or not langfuse_installed():
        return None
    from langfuse import Langfuse

    client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                      base_url=settings.langfuse_base_url, timeout=TIMEOUT_S)
    return LangfuseTracer(client, public_key=settings.langfuse_public_key)


class LangfuseTracer:
    def __init__(self, client: Any, *, public_key: str) -> None:
        self.client = client
        self.public_key = public_key

    @contextmanager
    def turn(self, text: str, *, session_id: str, tags: list[str]) -> Iterator[LangfuseTurn]:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        # session_id·trace_name 은 US-ASCII 200자 이하, tag 는 200자 이하여야 한다(넘으면 버려진다).
        # propagate_attributes 는 루트가 현재 span 이 된 뒤에 들어가야 루트에도 붙는다.
        with (self.client.start_as_current_observation(name="turn", as_type="agent", input=text) as root,
              propagate_attributes(session_id=session_id, trace_name="code-agent turn",
                                   tags=[tag[:200] for tag in tags])):
            yield LangfuseTurn(root, CallbackHandler(public_key=self.public_key))

    def shutdown(self) -> None:
        """남은 span 을 보내고 백그라운드 스레드를 끝낸다. REPL 이 끝날 때 부른다."""
        self.client.shutdown()


class LangfuseTurn:
    def __init__(self, root: Any, handler: Any) -> None:
        self.root = root
        self.callbacks = [handler]  # 턴 안의 모든 stream 호출(재개 포함)에 같은 인스턴스를 쓴다

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        approved = decision.get("type") == "approve"
        self.root.create_event(
            name=f"approval:{action['name']}",
            input={"tool": action["name"], "call": describe_call(action)},
            output={"decision": decision.get("type"), "by": "auto" if auto else "user",
                    "reason": reject_reason(decision)},
            level="DEFAULT" if approved else "WARNING",
        )

    def end(self, *, output: str, usage: Usage, error: str | None = None) -> None:
        fields: dict[str, Any] = {"output": output, "metadata": {"usage": asdict(usage)}}
        if error:
            fields.update(level="ERROR", status_message=error)
        self.root.update(**fields)
