"""Langfuse 트레이스. 사용자 메시지 하나가 트레이스 하나다(루트 observation "turn").

- 루트 아래에 LangChain CallbackHandler 가 그래프 노드·미들웨어·모델 호출·도구 실행을 기록한다.
  승인 대기 후 재개(Command(resume=...))도 같은 핸들러를 쓰므로 같은 턴 아래에 이어진다.
- 승인 결정은 이 모듈이 event 로 남긴다
  (도구, 호출 요약, approve/reject, 사람이 정했는지, 거부 사유).
- 턴이 끝나면 루트 출력에 작업 기록(최종 답변, 도구 실행과 결과 요약, 승인, 바뀐 파일, diff)을
  남긴다(worklog.py).
  LLM-as-a-Judge 평가기는 turn 하나만 읽으므로 판단 근거가 여기 있어야 한다.
- 출력 한도로 잘린 응답이나 예산 상한 도달(stops.py)은 turn 을 WARNING 수준으로 표시하고
  작업 기록에 경고 절을 둔다.
- UsageCallback 합계(OpenRouter 실제 비용 포함)는 루트 metadata 에 남긴다. 핸들러는 토큰만 넘기고
  비용은 Langfuse 가 모델 정의로 계산하므로, OpenRouter 모델 이름에는 가격 정의를 등록해야 한다.
  턴 소요 시간(duration_s)과 자동 복구 횟수(retries)도 metadata 에 둔다.
- 그래프가 get_stream_writer 로 보내는 도메인 사건({"type": 이름, ...}, 예: plan_written)은
  루트의 event(이름 = type)로 남긴다. 로컬 트레이스(localtrace.py)도 같은 사건을 남긴다.

LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 가 둘 다 있고 langfuse extra 가 설치돼 있을 때만 켜진다.
로컬 트레이스와 함께 쓸 때는 CompositeTracer 로 묶는다(combine_tracers). CompositeTracer 는 자식
트레이서 하나의 실패(열기, 알림, 끝내기, 종료)를 삼키고 나머지를 이어 간다. 트레이스 때문에 턴이나
REPL 이 멈추면 안 되기 때문이다.

트레이서는 선택 속성 두 개를 가질 수 있다(없어도 된다).
- uses_context: turn(context=…) 의 이전 턴 요약을 쓰는지. 거짓이면 App 이 그것을 만들지 않는다.
- notices(): 사용자에게 한 번 보일 알림(로컬 기록 실패 등). App 이 턴 뒤에 흐리게 찍는다.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

from code_agent.approval import reject_reason
from code_agent.render import describe_call
from code_agent.settings import Settings
from code_agent.usage import Usage
from code_agent.worklog import WorkLog, diff_snapshots, snapshot

TIMEOUT_S = 5  # 내보내기 요청 제한 시간. 종료 시 flush 가 이보다 오래 붙잡지 않게 한다


class TurnTrace(Protocol):
    """한 턴의 트레이스. cli.App.turn 이 스트림 조각·승인 결정·턴 끝을 알린다."""

    callbacks: list

    def on_part(self, part: tuple) -> None:
        """스트림 조각 하나를 받는다(작업 기록용)."""

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        """승인 결정 하나를 남긴다. auto 는 사람에게 묻지 않고 정했는지다."""

    def end(self, *, output: str, usage: Usage, error: str | None = None,
            warnings: Sequence[str] = (), retries: int = 0, interrupted: bool = False) -> None:
        """턴을 끝내고 최종 답변·사용량·오류·경고를 남긴다.

        retries 는 자동 복구로 모델을 다시 부른 횟수(stops.TurnStops.recoveries),
        interrupted 는 사용자가 Ctrl+C 로 끊었는지다(그때 error 도 함께 온다).
        """


class Tracer(Protocol):
    """트레이서. 턴마다 트레이스를 열고, REPL 이 끝날 때 남은 것을 보낸다."""

    def turn(self, text: str, *, session_id: str, tags: list[str],
             workspace: Path | None = None, context: str = "") -> Any:
        """턴 하나의 트레이스를 연다. ContextManager[TurnTrace] 를 돌려준다."""

    def shutdown(self) -> None:
        """남은 트레이스를 보내고 끝낸다."""


def langfuse_installed() -> bool:
    """langfuse extra 가 설치돼 있는지."""
    return importlib.util.find_spec("langfuse") is not None


def domain_event(data: Any) -> dict[str, Any] | None:
    """custom 스트림 조각이 도메인 사건이면 JSON 으로 쓸 수 있는 필드만 담은 사본, 아니면 None.

    계획·메모리·자기 개선 같은 기능은 get_stream_writer()({"type": 이름, ...}) 로 사건을 보내고,
    두 트레이서(Langfuse, 로컬)가 이 사본을 남긴다.
    직렬화되지 않는 필드는 뺀다(기록이 깨지지 않게).
    """
    if not isinstance(data, dict) or not isinstance(data.get("type"), str) or not data["type"]:
        return None
    event: dict[str, Any] = {}
    for key, value in data.items():
        try:
            event[str(key)] = json.loads(json.dumps(value, ensure_ascii=False))  # 깊은 사본
        except (TypeError, ValueError):
            continue
    return event


def combine_tracers(tracers: Sequence[Tracer | None]) -> CompositeTracer | None:
    """None 을 뺀 트레이서들을 CompositeTracer 하나로 묶는다. 없으면 None.

    하나여도 묶는다. 그 트레이서가 실패해도 턴과 REPL 이 멈추지 않게 하려는 것이다.
    앞의 트레이서부터 열고 알린다(cli 는 로컬을 앞에 둔다).
    """
    found = [tracer for tracer in tracers if tracer is not None]
    return CompositeTracer(found) if found else None


def tracer_notices(tracer: Any) -> list[str]:
    """트레이서가 사용자에게 한 번 보일 알림(notices()). 없거나 실패하면 빈 목록."""
    notices: Any = getattr(tracer, "notices", None)
    if not callable(notices):
        return []
    try:
        # callable() 로 좁히면 반환이 object 가 되어 pyright 가 반복을 거부한다
        return [str(notice) for notice in cast(Any, notices)()]
    except Exception:  # noqa: BLE001 - 알림 때문에 REPL 을 멈추지 않는다
        return []


def _quiet_exit(manager: AbstractContextManager[Any]) -> Callable[..., bool]:
    """자식 턴 컨텍스트를 닫는 함수. 닫다가 난 예외는 삼킨다(ExitStack.push 용)."""
    def close(exc_type: type[BaseException] | None, exc: BaseException | None,
              traceback: TracebackType | None) -> bool:
        try:
            return bool(manager.__exit__(exc_type, exc, traceback))
        except Exception:  # noqa: BLE001 - 한 트레이서의 실패로 턴을 멈추지 않는다
            return False
    return close


class CompositeTracer:
    """여러 트레이서(로컬, Langfuse)에 같은 턴을 나눠 준다.

    App 은 지금처럼 트레이서 하나만 본다. 자식 하나의 실패는 삼키고 나머지를 이어 간다.
    """

    def __init__(self, tracers: Sequence[Tracer]) -> None:
        self.tracers = list(tracers)

    @property
    def uses_context(self) -> bool:
        """이전 턴 요약을 쓰는 자식이 있는지(속성이 없는 트레이서는 쓴다고 본다)."""
        return any(getattr(tracer, "uses_context", True) for tracer in self.tracers)

    @contextmanager
    def turn(self, text: str, *, session_id: str, tags: list[str],
             workspace: Path | None = None, context: str = "") -> Iterator[CompositeTurn]:
        """모든 트레이서의 턴을 열고, 그것들에 나눠 주는 CompositeTurn 을 내준다.

        열다가 실패한 트레이서는 이번 턴에서 빠진다. 닫다가 난 예외도 삼킨다.
        """
        with ExitStack() as stack:
            turns = []
            for tracer in self.tracers:
                try:
                    manager = tracer.turn(text, session_id=session_id, tags=tags,
                                          workspace=workspace, context=context)
                    turn = manager.__enter__()
                except Exception:  # noqa: BLE001 - 이 트레이서만 빠지고 턴은 돈다
                    continue
                stack.push(_quiet_exit(manager))
                turns.append(turn)
            yield CompositeTurn(turns)

    def shutdown(self) -> None:
        """모든 트레이서를 끝낸다. 하나가 실패해도 나머지는 끝낸다."""
        for tracer in self.tracers:
            try:
                tracer.shutdown()
            except Exception:  # noqa: BLE001 - 종료 실패로 REPL 종료를 막지 않는다
                continue

    def notices(self) -> list[str]:
        """자식 트레이서들의 한 번 보일 알림(tracer_notices)."""
        return [notice for tracer in self.tracers for notice in tracer_notices(tracer)]


class CompositeTurn:
    """여러 TurnTrace 에 같은 알림을 나눠 준다. 콜백은 모두 모아 한 목록으로 내놓는다.

    한 턴의 실패(예외)는 삼키고 다음 턴에 계속 넘긴다.
    """

    def __init__(self, turns: Sequence[TurnTrace]) -> None:
        self.turns = list(turns)
        self.callbacks = [callback for turn in self.turns
                          for callback in getattr(turn, "callbacks", None) or []]

    def _each(self, notify: Callable[[TurnTrace], None]) -> None:
        for turn in self.turns:
            try:
                notify(turn)
            except Exception:  # noqa: BLE001 - 한 트레이서의 실패로 턴을 멈추지 않는다
                continue

    def on_part(self, part: tuple) -> None:
        """스트림 조각을 모든 턴에 넘긴다."""
        self._each(lambda turn: turn.on_part(part))

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        """승인 결정을 모든 턴에 넘긴다."""
        self._each(lambda turn: turn.decision(action, decision, auto))

    def end(self, *, output: str, usage: Usage, error: str | None = None,
            warnings: Sequence[str] = (), retries: int = 0, interrupted: bool = False) -> None:
        """모든 턴을 끝낸다."""
        self._each(lambda turn: turn.end(output=output, usage=usage, error=error,
                                         warnings=warnings, retries=retries,
                                         interrupted=interrupted))


def make_tracer(settings: Settings) -> LangfuseTracer | None:
    """키가 없거나 extra 가 없으면 None. 키는 인자로만 넘겨 os.environ 을 바꾸지 않는다."""
    if not settings.langfuse_enabled or not langfuse_installed():
        return None
    from langfuse import Langfuse

    client = Langfuse(public_key=settings.langfuse_public_key,
                      secret_key=settings.langfuse_secret_key,
                      base_url=settings.langfuse_base_url, timeout=TIMEOUT_S)
    return LangfuseTracer(client, public_key=settings.langfuse_public_key)


class LangfuseTracer:
    """Langfuse 클라이언트로 턴마다 루트 observation "turn" 을 만든다."""

    uses_context = True  # 이전 턴 요약을 작업 기록의 맥락 절로 남긴다(평가기용)

    def __init__(self, client: Any, *, public_key: str) -> None:
        self.client = client
        self.public_key = public_key

    @contextmanager
    def turn(self, text: str, *, session_id: str, tags: list[str],
             workspace: Path | None = None, context: str = "") -> Iterator[LangfuseTurn]:
        """루트 observation "turn" 을 열고 그 안에서 LangfuseTurn 을 내준다.

        session_id·trace_name·tags 는 턴 안의 모든 observation 에 붙는다.
        """
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        # session_id·trace_name 은 US-ASCII 200자 이하, tag 는 200자 이하여야 한다
        # (넘으면 버려진다).
        # propagate_attributes 는 루트가 현재 span 이 된 뒤에 들어가야 루트에도 붙는다.
        with (self.client.start_as_current_observation(name="turn", as_type="agent",
                                                       input=text) as root,
              propagate_attributes(session_id=session_id, trace_name="code-agent turn",
                                   tags=[tag[:200] for tag in tags])):
            yield LangfuseTurn(root, CallbackHandler(public_key=self.public_key),
                               workspace=workspace, context=context)

    def shutdown(self) -> None:
        """남은 span 을 보내고 백그라운드 스레드를 끝낸다. REPL 이 끝날 때 부른다."""
        self.client.shutdown()


class LangfuseTurn:
    """한 턴의 Langfuse 기록. 작업 기록(WorkLog)을 모아 턴 끝에 루트 출력으로 남긴다."""

    def __init__(self, root: Any, handler: Any, *, workspace: Path | None = None,
                 context: str = "") -> None:
        self.root = root
        self.callbacks = [handler]  # 턴 안의 모든 stream 호출(재개 포함)에 같은 인스턴스를 쓴다
        self.started = time.perf_counter()
        self.workspace = workspace
        self.before = snapshot(workspace) if workspace is not None else None
        # 작업 폴더와 이전 턴 요약. 평가기는 시스템 프롬프트와 이전 턴을 보지 못한다
        self.context = context
        self.worklog = WorkLog()
        # resume 때 프롬프트가 바뀌어 옛 생각을 요청에서 뺐는지(resume_compat)
        self.prompt_migrated = False

    def on_part(self, part: tuple) -> None:
        """스트림 조각을 작업 기록에 넣는다.

        도메인 사건은 루트의 event 로 남기고, resume 이전(prompt_migrated) 알림은 기억한다.
        """
        self.worklog.on_part(part)
        _ns, mode, data = part
        if mode != "custom" or (event := domain_event(data)) is None:
            return
        name = event.pop("type")
        self.root.create_event(name=name, input=event or None, level="DEFAULT")
        if name == "prompt_migrated":
            # 모델만 바뀐 것은 빼지 않는다
            self.prompt_migrated = self.prompt_migrated or event.get("reason") == "prompt"

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        """승인 결정을 작업 기록과 루트의 event(approval:<도구>)로 남긴다."""
        self.worklog.decision(action, decision, auto)
        approved = decision.get("type") == "approve"
        self.root.create_event(
            name=f"approval:{action['name']}",
            input={"tool": action["name"], "call": describe_call(action)},
            output={"decision": decision.get("type"), "by": "auto" if auto else "user",
                    "reason": reject_reason(decision)},
            level="DEFAULT" if approved else "WARNING",
        )

    def end(self, *, output: str, usage: Usage, error: str | None = None,
            warnings: Sequence[str] = (), retries: int = 0, interrupted: bool = False) -> None:
        """output 은 최종 답변이다. 루트 출력에는 판단 근거가 되는 작업 기록 전체를 남긴다.

        warnings(잘린 응답, 예산 상한 도달)가 있으면 오류가 없을 때
        WARNING 수준과 status_message 로 표시한다.
        metadata 에는 사용량과 함께 소요 시간(duration_s)과 자동 복구 횟수(retries)를 둔다.
        """
        changes: list[str] | None
        if self.workspace is None:
            changes, diff = [], ""
        elif self.before is None or (after := snapshot(self.workspace)) is None:
            changes, diff = None, ""  # 파일이 너무 많아 스냅샷을 건너뛰었다
        else:
            changes, diff = diff_snapshots(self.before, after)
        metadata: dict[str, Any] = {
            "usage": asdict(usage), "retries": retries,
            "duration_s": round(time.perf_counter() - self.started, 3)}
        if self.prompt_migrated:
            metadata["prompt_migrated"] = True
        if interrupted:
            metadata["interrupted"] = True
        fields: dict[str, Any] = {
            "output": self.worklog.render(answer=output, changes=changes, diff=diff,
                                          context=self.context, warnings=warnings),
            "metadata": metadata}
        if error:
            fields.update(level="ERROR", status_message=error)
        elif warnings:
            fields.update(level="WARNING", status_message="\n".join(warnings))
        self.root.update(**fields)
