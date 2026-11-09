"""로컬 트레이스. Langfuse 없이도 턴마다 사건과 지표를 jsonl 파일로 남긴다.

- 위치는 <data_dir>/runs/<thread_id>/ 다(기본 ~/.code-agent/runs/).
  - trace.jsonl: 사건 한 줄씩. 생길 때마다 덧붙여 써서,
    끊기거나 강제 종료된 턴도 거기까지는 남는다.
  - metrics.jsonl: 턴이 끝날 때 지표 한 줄.
- 모든 사건에 ts(UTC ISO), turn(thread 안의 턴 번호), type, t(턴 시작부터 지난 초)가 있다.
  - type: turn_start, llm, tool, approval, recovery, warning, error, turn_end.
  - 그 밖의 type 은 그래프가 get_stream_writer 로 보낸 도메인 사건이다
    (plan_written 등, tracing.domain_event).
  - 서브에이전트에서 난 사건에는 sub(스트림 네임스페이스)와 agent(이름, 알 때)가 붙는다.
- 모델·도구의 소요 시간은 LangChain 콜백의 run_id 로 잰다.
  도구의 결과(상태, exit code)는 스트림의 ToolMessage 로 정한다.
  가드가 막거나 사람이 거부한 호출은 실행되지 않아 도구 콜백이 오지 않기 때문이다.
- 도구 출력 원문은 남기지 않고 한 줄 요약(worklog.summarize_result)만 남긴다.
  .env 를 언급한 셸 명령은 그 요약도 빼고 exit code 만 남긴다. 파일 도구의 .env 는 가드가 막는다.
  요약의 마지막 출력 줄이 'NAME=…'(환경 변수 대입 모양)이면 비밀값일 수 있어 exit code 만 남긴다
  (cat .en? 처럼 '.env' 글자 없이 .env 를 읽은 명령).
- metrics 의 retries 는 메인 에이전트의 잘림 자동 복구 횟수다(stops.TurnStops.recoveries).
  서브에이전트의 복구는 sub 가 붙은 recovery 사건으로만 보인다.
  SDK 가 안에서 하는 HTTP 재시도(max_retries)는 콜백에 오지 않아 보이지 않는다.
- 기록에 실패해도(디스크, 권한 등) 턴은 멈추지 않는다. 프로세스에서 처음 난 실패는 App 이
  턴 뒤에 흐린 한 줄로 한 번 알린다(LocalTracer.notices).
- 강제 종료로 반만 쓴 줄이 있어도 이어 쓴다. 턴마다 파일에 처음 쓰기 전에 끝이 줄바꿈이 아니면
  줄바꿈부터 써서 새 사건이 그 조각에 붙어 함께 깨지지 않게 한다.

/trace 의 표(trace_view)도 여기서 만든다. 사건의 글은 모두 rich Text 로 넣는다(마크업으로
읽히면 '[/]' 같은 글이 표를 깨뜨린다).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from code_agent.approval import reject_reason
from code_agent.guards import strip_env_matches
from code_agent.render import describe_call
from code_agent.stops import recovery_notice, stop_reason
from code_agent.tracing import domain_event
from code_agent.usage import Usage
from code_agent.worklog import one_line, summarize_result

RUNS_DIR = "runs"
TRACE_FILE = "trace.jsonl"
METRICS_FILE = "metrics.jsonl"
MAX_REQUEST = 200  # turn_start 에 남기는 요청 앞부분 글자 수
BASE_KEYS = ("ts", "turn", "type", "t")  # 모든 사건에 있는 필드
_ENV_MENTION = re.compile(r"\.env", re.IGNORECASE)
_EXIT_CODE = re.compile(r"exit code (\d+)|Exit code: (\d+)", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")  # SECRET=… (.env 한 줄 모양)
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def run_dir(data_dir: Path, thread_id: str) -> Path:
    """thread 하나의 로컬 트레이스 폴더. thread_id 는 폴더 이름으로 안전한 글자만 남긴다."""
    return data_dir / RUNS_DIR / (_UNSAFE.sub("_", thread_id).strip(".") or "_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """jsonl 의 객체 줄들. 없는 파일은 빈 목록이다.

    줄은 LF 로만 나눈다. str.splitlines 는 U+2028·U+0085 같은 글자에서도 나눠, 그 글자가 든
    사건(ensure_ascii=False 라 그대로 쓴다)을 깨뜨린다.
    깨진 줄(강제 종료로 반만 쓴 줄 등)은 건너뛴다.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _encode_line(row: dict[str, Any]) -> bytes:
    """한 줄의 바이트.

    짝 없는 서로게이트(\\udc80 등)는 UTF-8 로 쓸 수 없어 \\u 이스케이프로 쓴다.
    """
    try:
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    except UnicodeEncodeError:
        return (json.dumps(row, ensure_ascii=True) + "\n").encode("ascii")


def _ends_mid_line(path: Path) -> bool:
    """파일이 줄바꿈 없이 끝나는지(강제 종료로 반만 쓴 줄). 없거나 빈 파일이면 거짓."""
    try:
        with path.open("rb") as file:
            if file.seek(0, os.SEEK_END) == 0:
                return False
            file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
    except OSError:
        return False


def append_jsonl(path: Path, row: dict[str, Any], *, repair: bool = False) -> None:
    """한 줄을 덧붙인다. 줄 끝은 OS 와 상관없이 LF 다.

    repair 면 먼저 파일이 반만 쓴 줄로 끝나는지 보고, 그러면 줄바꿈부터 쓴다.
    """
    data = _encode_line(row)
    if repair and _ends_mid_line(path):
        data = b"\n" + data
    with path.open("ab") as file:
        file.write(data)


def turn_numbers(events: Sequence[dict[str, Any]]) -> list[int]:
    """trace.jsonl 사건들에 있는 턴 번호, 오름차순.

    turn_start 가 깨져도 그 턴의 다른 사건으로 찾는다.
    """
    return sorted({e["turn"] for e in events if _is_turn(e.get("turn"))})


def _is_turn(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def last_turn(folder: Path) -> int:
    """thread 폴더에 남은 가장 큰 턴 번호(사건과 지표 모두에서). 없으면 0."""
    rows = [*read_jsonl(folder / TRACE_FILE), *read_jsonl(folder / METRICS_FILE)]
    return max((row["turn"] for row in rows if _is_turn(row.get("turn"))), default=0)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _tag(tags: Sequence[str], prefix: str) -> str:
    """App 이 붙인 태그(mode:…, model:…)의 값."""
    return next((tag[len(prefix):] for tag in tags if tag.startswith(prefix)), "")


def _call_key(call: dict) -> str:
    """승인 결정과 도구 결과를 짝짓는 열쇠(이름 + 인자). 승인 요청에는 호출 id 가 없다."""
    return json.dumps([call.get("name"), call.get("args") or {}], ensure_ascii=False,
                      sort_keys=True, default=str)


def _exit_code(message: ToolMessage) -> int | None:
    """execute 결과의 종료 코드. deepagents 가 artifact 에 남기고, 없으면 본문에서 찾는다."""
    artifact = message.artifact
    if isinstance(artifact, dict) and isinstance(artifact.get("exit_code"), int):
        return artifact["exit_code"]
    match = _EXIT_CODE.search(message.text)
    return int(next(g for g in match.groups() if g)) if match else None


def _last_output_line(message: ToolMessage) -> str:
    """execute 출력에서 종료 코드 줄을 뺀 마지막 줄(summarize_result 가 요약에 넣는 줄)."""
    lines = [line.strip() for line in message.text.splitlines()
             if line.strip() and not _EXIT_CODE.search(line)]
    return lines[-1] if lines else ""


def _summary(call: dict, message: ToolMessage, exit_code: int | None) -> str:
    """도구 결과 한 줄 요약. 출력 원문은 남기지 않는다.

    execute 요약에는 출력의 마지막 줄이 들어가는데, 그 줄이 .env 의 값일 수 있다.
    .env 를 언급한 명령이거나 마지막 줄이 'NAME=…' 모양이면 exit code 만 남긴다.
    """
    command = str((call.get("args") or {}).get("command", ""))
    if call.get("name") == "execute" and (
            _ENV_MENTION.search(command)
            or (message.status != "error" and _ASSIGNMENT.match(_last_output_line(message)))):
        return (f"exit {exit_code}" if exit_code is not None
                else "(비밀값일 수 있는 출력이라 요약을 남기지 않았다)")
    if message.name == "grep":  # 가드가 이미 지웠지만 한 번 더 거른다
        message = message.model_copy(update={"content": strip_env_matches(message.text)})
    return summarize_result(message)


def _scope(ns: Sequence[str], agent: Any = None) -> dict[str, str]:
    """서브에이전트 사건 표시(sub, agent). 메인 에이전트면 빈 dict."""
    if not ns:
        return {}
    scope = {"sub": "|".join(ns)}
    if isinstance(agent, str) and agent:
        scope["agent"] = agent
    return scope


def _error_status(error: BaseException | None) -> str:
    if error is None:
        return "ok"
    return "interrupted" if isinstance(error, KeyboardInterrupt) else "error"


def _error_line(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return one_line(f"{type(error).__name__}: {error}".rstrip(": "))


def _first_message(response: LLMResult | None) -> BaseMessage | None:
    for generations in response.generations if response is not None else []:
        for generation in generations:
            message = getattr(generation, "message", None)
            if isinstance(message, BaseMessage):
                return message
    return None


@dataclass
class _Span:
    """콜백으로 잰 모델·도구 실행 하나."""

    ts: str
    t: float
    started: float
    metadata: dict[str, Any]
    name: str = ""
    tool_call_id: str = ""
    duration_s: float | None = None
    error: BaseException | None = None

    def scope(self) -> dict[str, str]:
        """서브에이전트 안의 실행이면 sub·agent. checkpoint_ns 가 'a|b' 처럼 중첩돼 있다."""
        ns = str(self.metadata.get("langgraph_checkpoint_ns")
                 or self.metadata.get("checkpoint_ns") or "")
        if "|" not in ns:
            return {}
        return _scope((ns.rsplit("|", 1)[0],), self.metadata.get("lc_agent_name"))


class LocalTracer:
    """턴마다 <data_dir>/runs/<thread_id>/ 에 사건(trace.jsonl)과 지표(metrics.jsonl)를 남긴다.

    tracing.Tracer 프로토콜을 따른다. 턴 번호는 thread 마다 1부터 센다.
    프로세스를 다시 띄워 --resume 해도 trace.jsonl 의 마지막 turn_start 번호에서 이어 간다.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._last: dict[str, int] = {}  # thread_id → 마지막 턴 번호

    def folder(self, thread_id: str) -> Path:
        """thread 의 트레이스 폴더."""
        return run_dir(self.data_dir, thread_id)

    @contextmanager
    def turn(self, text: str, *, session_id: str, tags: list[str],
             workspace: Path | None = None, context: str = "") -> Iterator[LocalTurn]:
        """턴 번호를 정하고 turn_start 를 쓴 LocalTurn 을 내준다.

        context(이전 턴 요약)는 Langfuse 평가기용이라 쓰지 않는다.
        """
        folder = self.folder(session_id)
        last = self._last.get(session_id)
        if last is None:
            last = max(turn_numbers(read_jsonl(folder / TRACE_FILE)), default=0)
        self._last[session_id] = last + 1
        yield LocalTurn(folder, last + 1, text, thread_id=session_id, tags=tags,
                        workspace=workspace)

    def shutdown(self) -> None:
        """사건을 바로 파일에 쓰므로 보낼 것이 없다."""


class LocalTurn:
    """한 턴의 로컬 기록. 사건은 생길 때마다 trace.jsonl 에 덧붙이고, end 에서 지표 한 줄을 쓴다.

    tracing.TurnTrace 프로토콜을 따른다. callbacks 의 TimingCallback 이 모델·도구 실행 시간을 잰다.

    Args:
        folder: thread 의 트레이스 폴더(run_dir).
        number: thread 안의 턴 번호.
        text: 사용자 요청. 앞 MAX_REQUEST 자만 남긴다.
        thread_id: 세션(thread) ID.
        tags: App 이 붙인 태그. mode:…, model:… 을 turn_start 와 지표에 남긴다.
        workspace: 작업 폴더.
    """

    def __init__(self, folder: Path, number: int, text: str, *, thread_id: str,
                 tags: Sequence[str] = (), workspace: Path | None = None) -> None:
        self.folder, self.number, self.thread_id = folder, number, thread_id
        self.mode, self.model = _tag(tags, "mode:"), _tag(tags, "model:")
        self.started, self.started_at = time.perf_counter(), _now()
        self.callbacks: list[BaseCallbackHandler] = [TimingCallback(self)]
        self.tool_calls = self.tool_errors = self.rejections = 0
        # 콜백은 병렬 도구·서브에이전트의 스레드에서도 온다. emit 이 안에서 다시 잡으므로 RLock
        self._lock = threading.RLock()
        self._spans: dict[UUID, _Span] = {}  # run_id → 진행 중인 모델·도구 실행
        self._tools: dict[str, _Span] = {}  # 호출 id → 끝났지만 결과를 아직 못 본 도구 실행
        self._calls: dict[str, tuple[dict, tuple[str, ...]]] = {}  # 호출 id → (호출, 네임스페이스)
        self._done: set[str] = set()  # 결과를 기록한 호출 id
        self._rejected: list[str] = []  # 거부한 호출의 열쇠. 그 오류 결과는 '거부'로 표시한다
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # 쓰기마다 실패를 삼킨다
        self.emit("turn_start", t=0.0, ts=self.started_at,
                  request=" ".join(text.split())[:MAX_REQUEST], mode=self.mode or None,
                  model=self.model or None, thread_id=thread_id,
                  workspace=str(workspace) if workspace is not None else None)

    def elapsed(self) -> float:
        """턴 시작부터 지난 초."""
        return time.perf_counter() - self.started

    def emit(self, kind: str, *, ts: str | None = None, t: float | None = None,
             **fields: Any) -> None:
        """사건 한 줄을 trace.jsonl 에 덧붙인다. 값이 None 인 필드는 뺀다."""
        self._event(kind, fields, ts=ts, t=t)

    def _event(self, kind: str, fields: dict[str, Any], *, ts: str | None = None,
               t: float | None = None) -> None:
        event: dict[str, Any] = {"ts": ts or _now(), "turn": self.number, "type": kind,
                                 "t": round(self.elapsed() if t is None else t, 3)}
        for key, value in fields.items():
            if value is not None:
                event.setdefault(key, value)  # 도메인 사건이 기본 필드를 덮지 못하게 한다
        self._append(TRACE_FILE, event)

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with self._lock:
            try:
                append_jsonl(self.folder / name, row)
            except (OSError, TypeError, ValueError):  # 기록 실패로 턴을 멈추지 않는다
                pass

    # --- 스트림·승인·턴 끝(App.turn) ---

    def on_part(self, part: tuple) -> None:
        """도구 호출과 결과, 자동 복구, 도메인 사건을 기록한다."""
        ns, mode, data = part
        if mode == "custom":
            if (event := domain_event(data)) is not None:
                kind = event.pop("type")
                self._event(kind, {**event, **_scope(ns)})
            return
        if mode != "updates" or not isinstance(data, dict):
            return
        for node, update in data.items():
            messages = update.get("messages") if isinstance(update, dict) else None
            stopped: AIMessage | None = None  # 복구 안내 바로 앞의 정리된 응답
            for message in messages if isinstance(messages, list) else []:
                if node == "model" and isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        with self._lock:
                            self._calls[call["id"] or ""] = (dict(call), tuple(ns))
                elif isinstance(message, ToolMessage):
                    self._tool_result(message)
                elif isinstance(message, AIMessage) and stop_reason(message) is not None:
                    stopped = message
                elif recovery_notice(node, message):
                    self._recovery(stopped, ns)

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        """승인 결정을 approval 사건으로 남긴다. 거부한 호출은 그 결과를 '거부'로 표시한다."""
        if decision.get("type") != "approve":
            with self._lock:
                self.rejections += 1
                self._rejected.append(_call_key(action))
        reason = reject_reason(decision)
        self.emit("approval", tool=action.get("name"), call=one_line(describe_call(action)),
                  decision=decision.get("type"), by="auto" if auto else "user",
                  reason=one_line(reason) if reason else None)

    def end(self, *, output: str, usage: Usage, error: str | None = None,
            warnings: Sequence[str] = (), retries: int = 0, interrupted: bool = False) -> None:
        """경고·오류·turn_end 사건과 metrics.jsonl 의 지표 한 줄을 쓴다.

        status 는 interrupted(사용자 중단) > error > warning(잘림·예산 경고) > ok 순으로 정한다.
        최종 답변은 길이만 남긴다.
        """
        with self._lock:
            leftover = [(call_id, span) for call_id, span in self._tools.items()
                        if call_id not in self._done]
            self._tools.clear()
        # 실행은 끝났는데 결과가 스트림에 오지 않았다(도구 예외나 Ctrl+C 로 턴이 끝남).
        # 예외 없이 끝난 도구는 실패로 세지 않는다
        for call_id, span in leftover:
            call, ns = self._calls.get(call_id, ({"name": span.name, "args": {}}, ()))
            self._tool_event(call, ns, span, status=_error_status(span.error),
                             summary=_error_line(span.error) or "(결과를 받기 전에 턴이 끝났다)")
        for warning in warnings:
            self.emit("warning", message=one_line(warning))
        if error and not interrupted:
            self.emit("error", message=one_line(error))
        status = ("interrupted" if interrupted else "error" if error
                  else "warning" if warnings else "ok")
        duration = round(self.elapsed(), 3)
        self.emit("turn_end", status=status, duration_s=duration, answer_chars=len(output))
        self._append(METRICS_FILE, {
            "turn": self.number, "thread_id": self.thread_id, "started_at": self.started_at,
            "duration_s": duration, "status": status, "mode": self.mode, "model": self.model,
            "model_calls": usage.calls, "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors, "retries": retries, "rejections": self.rejections,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_read": usage.cache_read, "cache_write": usage.cache_write,
            "reasoning_tokens": usage.reasoning,
            # Anthropic 직접 호출은 비용을 주지 않는다. 0 과 '모름'을 구분한다
            "cost": round(usage.cost, 6) if usage.priced_calls else None})

    def _tool_result(self, message: ToolMessage) -> None:
        call_id = message.tool_call_id
        with self._lock:
            if call_id not in self._calls or call_id in self._done:
                # 이번 턴에 본 호출이 아니다(PatchToolCalls 가 기록 전체를 다시 내보낸 것 등)
                return
            self._done.add(call_id)
            call, ns = self._calls[call_id]
            span = self._tools.pop(call_id, None)
            key = _call_key(call)
            rejected = message.status == "error" and key in self._rejected
            if rejected:
                self._rejected.remove(key)
        exit_code = _exit_code(message) if call.get("name") == "execute" else None
        failed = message.status == "error" or exit_code not in (None, 0)
        status = "rejected" if rejected else "error" if failed else "ok"
        # 거부 사유는 approval 사건에 있다. 결과는 HITL 의 긴 안내문이라 되풀이하지 않는다
        summary = "거부됨" if rejected else _summary(call, message, exit_code)
        self._tool_event(call, ns, span, status=status, exit_code=exit_code, summary=summary)

    def _tool_event(self, call: dict, ns: Sequence[str], span: _Span | None, *, status: str,
                    summary: str, exit_code: int | None = None) -> None:
        self.tool_calls += 1
        if status == "error":
            self.tool_errors += 1
        agent = span.metadata.get("lc_agent_name") if span is not None else None
        fields: dict[str, Any] = {
            "name": call.get("name"), "call": one_line(describe_call(call)), "status": status,
            "duration_s": span.duration_s if span else None, "exit_code": exit_code,
            "result" if status == "ok" else "error": summary, **_scope(ns, agent)}
        self._event("tool", fields, ts=span.ts if span else None, t=span.t if span else None)

    def _recovery(self, stopped: AIMessage | None, ns: Sequence[str]) -> None:
        stripped = (stopped.response_metadata.get("code_agent_stripped") or {}
                    if stopped is not None else {})
        self._event("recovery", {
            "reason": stop_reason(stopped) if stopped is not None else None,
            "reasoning_chars": stripped.get("reasoning_chars"), **_scope(ns)})

    # --- 콜백(TimingCallback) ---

    def span_start(self, run_id: UUID, metadata: dict[str, Any] | None, *, name: str = "",
                   tool_call_id: str | None = None) -> None:
        """모델·도구 실행이 시작됐다."""
        span = _Span(ts=_now(), t=self.elapsed(), started=time.perf_counter(),
                     metadata=dict(metadata or {}), name=name, tool_call_id=tool_call_id or "")
        with self._lock:
            self._spans[run_id] = span

    def _finish(self, run_id: UUID, error: BaseException | None) -> _Span | None:
        with self._lock:
            span = self._spans.pop(run_id, None)
        if span is not None:
            span.duration_s = round(time.perf_counter() - span.started, 3)
            span.error = error
        return span

    def llm_end(self, run_id: UUID, *, response: LLMResult | None = None,
                error: BaseException | None = None) -> None:
        """모델 호출이 끝났다. 소요 시간·토큰·정지 사유를 llm 사건으로 남긴다.

        응답 글은 남기지 않는다.
        """
        span = self._finish(run_id, error)
        if span is None:
            return
        message = _first_message(response)
        meta = message.response_metadata if message is not None else {}
        usage = (message.usage_metadata if isinstance(message, AIMessage) else None) or {}
        details_in = usage.get("input_token_details") or {}
        cost = meta.get("cost")
        self._event("llm", {
            "duration_s": span.duration_s, "status": _error_status(error),
            "model": meta.get("model_name") or span.metadata.get("ls_model_name"),
            "node": span.metadata.get("langgraph_node"),
            "finish_reason": meta.get("finish_reason") or meta.get("stop_reason"),
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "cache_read": details_in.get("cache_read") or 0,
            "cache_write": details_in.get("cache_creation") or 0,
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning") or 0,
            "cost": float(cost) if isinstance(cost, int | float) else None,
            "tool_calls": len(message.tool_calls) if isinstance(message, AIMessage) else None,
            "error": _error_line(error), **span.scope()}, ts=span.ts, t=span.t)

    def tool_end(self, run_id: UUID, *, error: BaseException | None = None) -> None:
        """도구 실행이 끝났다. 결과(ToolMessage)가 스트림에 오면 그때 tool 사건으로 남긴다."""
        span = self._finish(run_id, error)
        if span is not None and span.tool_call_id:
            with self._lock:
                self._tools[span.tool_call_id] = span


class TimingCallback(BaseCallbackHandler):
    """모델·도구 실행의 시작과 끝을 run_id 로 짝지어 LocalTurn 에 알린다.

    config["callbacks"] 로 넘기면 서브에이전트·요약 미들웨어 안의 호출까지 온다.
    """

    def __init__(self, turn: LocalTurn) -> None:
        self.turn = turn

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]],
                            *, run_id: UUID, parent_run_id: UUID | None = None,
                            tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
                            **kwargs: Any) -> None:
        """채팅 모델 호출이 시작됐다."""
        self.turn.span_start(run_id, metadata)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, tags: list[str] | None = None,
                     metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """채팅이 아닌 LLM 호출이 시작됐다."""
        self.turn.span_start(run_id, metadata)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None,
                   tags: list[str] | None = None, **kwargs: Any) -> None:
        """모델 호출이 끝났다."""
        self.turn.llm_end(run_id, response=response)

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     parent_run_id: UUID | None = None, tags: list[str] | None = None,
                     **kwargs: Any) -> None:
        """모델 호출이 실패했다."""
        self.turn.llm_end(run_id, error=error)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, tags: list[str] | None = None,
                      metadata: dict[str, Any] | None = None,
                      inputs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """도구 실행이 시작됐다. tool_call_id 로 스트림의 결과와 짝짓는다."""
        self.turn.span_start(run_id, metadata, name=str((serialized or {}).get("name") or ""),
                             tool_call_id=kwargs.get("tool_call_id"))

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None,
                    **kwargs: Any) -> None:
        """도구 실행이 끝났다."""
        self.turn.tool_end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        """도구 실행이 예외로 끝났다."""
        self.turn.tool_end(run_id, error=error)


# --- /trace 출력 ---

def trace_view(folder: Path, arg: str = "") -> RenderableType | None:
    """/trace 의 출력을 만든다.

    Args:
        folder: thread 의 트레이스 폴더(run_dir).
        arg: 비면 마지막 턴, 숫자면 그 번호의 턴, "errors" 면 이 세션의 실패한 도구·오류·경고.

    Returns:
        rich 로 그릴 표(턴이면 끝에 지표 한 줄)나 안내 문구. 이 thread 의 트레이스가 없으면 None.
    """
    events = read_jsonl(folder / TRACE_FILE)
    turns = turn_numbers(events)
    if not turns:
        return None
    if arg == "errors":
        return _errors_view(events)
    if arg and not (arg.isascii() and arg.isdecimal()):
        return Text("사용법: /trace [n|errors]", style="red")
    number = int(arg) if arg else turns[-1]
    if number not in turns:
        return Text(f"턴 {number} 의 트레이스가 없다(있는 턴 {turns[0]}~{turns[-1]}).",
                    style="red")
    metrics = next((m for m in reversed(read_jsonl(folder / METRICS_FILE))
                    if m.get("turn") == number), None)
    return _turn_view(number, [e for e in events if e.get("turn") == number], metrics)


def _num(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _status(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind == "approval":
        return str(event.get("decision") or "")
    if kind in ("warning", "error"):
        return str(kind)
    return str(event.get("status") or "")


def _style(event: dict[str, Any]) -> str | None:
    """실패는 빨강, 경고·거부·중단·복구는 노랑."""
    status = _status(event)
    if status == "error":
        return "red"
    if (status in ("warning", "rejected", "reject", "interrupted")
            or event.get("type") == "recovery"):
        return "yellow"
    return None


def _content(event: dict[str, Any]) -> str:
    """표의 '내용' 칸. 사건 종류마다 사람이 읽을 한 줄로 만든다."""
    kind = event.get("type")
    sub = ""
    if event.get("sub"):
        sub = f"[sub{' ' + str(event['agent']) if event.get('agent') else ''}] "
    if kind == "turn_start":
        return f"mode {event.get('mode') or '?'} · model {event.get('model') or '?'}"
    if kind == "llm":
        node = event.get("node")
        where = f" ({node})" if node and node != "model" else ""  # 요약 미들웨어 등
        parts = [f"{event.get('model') or '모델'}{where}",
                 f"입력 {_int(event.get('input_tokens')):,}",
                 f"출력 {_int(event.get('output_tokens')):,}"]
        if _int(event.get("cache_read")):
            parts.append(f"캐시 읽기 {_int(event.get('cache_read')):,}")
        if event.get("finish_reason"):
            parts.append(str(event["finish_reason"]))
        if event.get("error"):
            parts.append(str(event["error"]))
        return sub + " · ".join(parts)
    if kind == "tool":
        return f"{sub}{event.get('call', '')} → {event.get('result') or event.get('error') or ''}"
    if kind == "approval":
        verdict = "승인" if event.get("decision") == "approve" else "거부"
        by = "자동" if event.get("by") == "auto" else "사용자"
        reason = f": {event['reason']}" if event.get("reason") else ""
        return f"{event.get('call', '')} → {verdict}({by}){reason}"
    if kind == "recovery":
        return f"{sub}멈춘 응답({event.get('reason') or '?'})을 자동으로 다시 요청"
    if kind in ("warning", "error"):
        return str(event.get("message", ""))
    if kind == "turn_end":
        return f"답변 {_int(event.get('answer_chars')):,}자"
    fields = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                       for key, value in event.items() if key not in BASE_KEYS)
    return one_line(sub + fields)  # 도메인 사건(plan_written 등)


def _duration(event: dict[str, Any]) -> str:
    value = event.get("duration_s")
    return f"{value:.2f}" if isinstance(value, int | float) else ""


def _metrics_line(metrics: dict[str, Any]) -> str:
    cache = (f" (캐시 읽기 {_int(metrics.get('cache_read')):,})"
             if _int(metrics.get("cache_read")) else "")
    parts = [f"{_num(metrics.get('duration_s')):.1f}초",
             f"모델 {_int(metrics.get('model_calls'))}회",
             f"도구 {_int(metrics.get('tool_calls'))}개(오류 {_int(metrics.get('tool_errors'))})",
             f"재시도 {_int(metrics.get('retries'))}", f"거부 {_int(metrics.get('rejections'))}",
             f"입력 {_int(metrics.get('input_tokens')):,}{cache}",
             f"출력 {_int(metrics.get('output_tokens')):,}"]
    cost = metrics.get("cost")
    if isinstance(cost, int | float):
        parts.append(f"${cost:.4f}")
    parts.append(f"상태 {metrics.get('status', '?')}")
    return " · ".join(parts)


def _turn_view(number: int, events: list[dict[str, Any]],
               metrics: dict[str, Any] | None) -> RenderableType:
    """한 턴의 타임라인 표. 실행 시작 순서(t)로 늘어놓는다(파일에는 끝난 순서로 쌓인다)."""
    events = sorted(events, key=lambda e: _num(e.get("t")))
    start = next((e for e in events if e.get("type") == "turn_start"), {})
    table = Table(title=Text(f"턴 {number} · {start.get('request', '')}"), title_justify="left")
    table.add_column("+초", justify="right")
    table.add_column("종류")
    table.add_column("내용", overflow="fold")
    table.add_column("소요(초)", justify="right")
    table.add_column("상태")
    for event in events:
        table.add_row(f"+{_num(event.get('t')):.1f}", str(event.get("type", "")),
                      Text(_content(event)), _duration(event), _status(event),
                      style=_style(event))
    footer = (Text(_metrics_line(metrics)) if metrics is not None
              else Text("지표 없음: 턴이 끝나기 전에 프로세스가 멈췄다.", style="yellow"))
    return Group(table, footer)


def _errors_view(events: list[dict[str, Any]]) -> RenderableType:
    """이 세션의 실패한 도구·모델 호출, 오류, 경고."""
    rows = [e for e in events if e.get("type") in ("warning", "error")
            or (e.get("type") in ("tool", "llm") and e.get("status") == "error")]
    if not rows:
        return Text("이 세션에는 실패한 도구·오류·경고가 없다.", style="dim")
    rows.sort(key=lambda e: (_int(e.get("turn")), _num(e.get("t"))))
    table = Table(title=Text(f"이 세션의 실패·경고 {len(rows)}건"), title_justify="left")
    table.add_column("턴", justify="right")
    table.add_column("+초", justify="right")
    table.add_column("종류")
    table.add_column("내용", overflow="fold")
    table.add_column("상태")
    for event in rows:
        table.add_row(str(event.get("turn", "")), f"+{_num(event.get('t')):.1f}",
                      str(event.get("type", "")), Text(_content(event)), _status(event),
                      style=_style(event))
    return table
