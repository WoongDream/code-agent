"""resume 호환(task13 방식 b). 프롬프트나 도구가 바뀐 뒤 옛 thread 를 이어 갈 때 옛 생각을 요청 사본에서 뺀다.

--resume 은 옛 응답의 생각(thinking)을 그대로 다시 보내지만 system 과 tools 는 지금 코드로 만든다. Opus 5.5 의 생각 서명은
그것을 만든 prefix(system, tools, 앞 메시지)에 묶여 있어 prefix 가 바뀌면 강제 계정은 400 이고, 아니면 무효인 생각이 간다.
- thread 마다 지문(system·tools 해시, 모델)을 따로 표에 둔다(cli.Sessions 처럼 기존 표에 열을 더하지 않는다).
- 지문이 다르거나, 기록이 없는 옛 thread 인데 이미 대화가 있으면 '이전'한다. 그 시점 state 에 있던 AI 메시지의 생각을
  이후 모든 요청 사본에서 뺀다(state 는 그대로). 앞쪽 생각 묶음을 지우는 것은 뒤 블록을 무효로 만들지 않고(Anthropic 가이드),
  이전 뒤로는 덧붙이기만 하는 기록이 된다. 화면에는 custom 스트림 이벤트로 한 번 알린다.
- 모델만 바뀌었으면 공급자가 옛 생각을 버리므로(400 아님) 빼지 않고 알리기만 한다.
'생각 뺄 구간'은 strip_reasoning 하나로 처리한다. task9(요약 창의 생각 빼기)도 구간 하나를 더해 이 함수를 쓰면 된다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_config, get_stream_writer

from code_agent.stops import REASONING_BLOCKS, REASONING_KWARGS

PLACEHOLDER = "(이전 세션의 생각 생략)"  # Anthropic 은 빈 assistant content 를 거부한다
Window = tuple[int, int]  # state 메시지 인덱스 구간 [start, end)


@dataclass(frozen=True)
class Fingerprint:
    system_sha: str
    tools_sha: str
    model: str

    def same_prompt(self, other: Fingerprint) -> bool:
        return (self.system_sha, self.tools_sha) == (other.system_sha, other.tools_sha)


@dataclass(frozen=True)
class PromptRecord:
    fingerprint: Fingerprint
    migrated_at: int | None = None  # state 의 이 개수보다 앞에 있던 AI 메시지는 생각을 뺀다


class PromptStore(Protocol):
    def get(self, thread_id: str) -> PromptRecord | None: ...

    def put(self, thread_id: str, record: PromptRecord) -> None: ...


class MemoryPromptStore:
    """시험과 sessions.db 없이 만든 agent 용."""

    def __init__(self) -> None:
        self.records: dict[str, PromptRecord] = {}

    def get(self, thread_id: str) -> PromptRecord | None:
        return self.records.get(thread_id)

    def put(self, thread_id: str, record: PromptRecord) -> None:
        self.records[thread_id] = record


class SqlitePromptStore:
    """sessions.db 의 code_agent_thread_prompt 표.

    모델 호출 직전(그래프 실행 중)에 읽고 쓴다. 그때 langgraph 는 앞 단계의 체크포인트를 백그라운드 스레드에서 같은 연결로
    저장할 수 있으므로(durability "async"), 체크포인터와 같은 잠금(SqliteSaver.lock)을 받아 트랜잭션이 섞이지 않게 한다.
    """

    def __init__(self, conn: sqlite3.Connection, lock: AbstractContextManager[Any] | None = None) -> None:
        self.conn = conn
        self.lock = lock if lock is not None else threading.Lock()
        with self.lock:
            conn.execute("CREATE TABLE IF NOT EXISTS code_agent_thread_prompt (thread_id TEXT PRIMARY KEY, "
                         "system_sha TEXT NOT NULL, tools_sha TEXT NOT NULL, model TEXT NOT NULL, "
                         "recorded_at TEXT NOT NULL, migrated_at INTEGER)")
            conn.commit()

    def get(self, thread_id: str) -> PromptRecord | None:
        with self.lock:
            row = self.conn.execute("SELECT system_sha, tools_sha, model, migrated_at FROM code_agent_thread_prompt "
                                    "WHERE thread_id = ?", (thread_id,)).fetchone()
        return PromptRecord(Fingerprint(*row[:3]), row[3]) if row else None

    def put(self, thread_id: str, record: PromptRecord) -> None:
        fp = record.fingerprint
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO code_agent_thread_prompt (thread_id, system_sha, tools_sha, "
                              "model, recorded_at, migrated_at) VALUES (?, ?, ?, ?, ?, ?)",
                              (thread_id, fp.system_sha, fp.tools_sha, fp.model, datetime.now(UTC).isoformat(),
                               record.migrated_at))
            self.conn.commit()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tool_schema(tool: Any) -> Any:
    try:
        return convert_to_openai_tool(tool)
    except (TypeError, ValueError):  # 공급자 고유 도구 dict 등
        return tool


def fingerprint(request: ModelRequest) -> Fingerprint:
    """요청의 system 전문, 도구 스키마(이름·설명·인자), 모델 이름. 요청 하나에 약 20KB 를 해시한다.

    모델 이름 필드는 ChatOpenRouter 가 model_name, ChatAnthropic 이 model 이다.
    """
    system = request.system_message.text if request.system_message is not None else ""
    tools = json.dumps([_tool_schema(t) for t in request.tools], sort_keys=True, ensure_ascii=False, default=str)
    model = getattr(request.model, "model_name", None) or getattr(request.model, "model", None) or ""
    return Fingerprint(_sha(system), _sha(tools), str(model))


def drop_reasoning(message: AIMessage) -> AIMessage:
    """생각을 뺀 같은 id 의 사본. 뺄 것이 없으면 그대로 돌려준다.

    - OpenRouter: additional_kwargs 의 reasoning_content, reasoning_details
    - Anthropic: content 의 thinking, redacted_thinking 블록
    남는 글도 도구 호출도 없으면 자리표시를 둔다(Anthropic 은 빈 assistant 를 거부한다).
    """
    kwargs = {k: v for k, v in message.additional_kwargs.items() if k not in REASONING_KWARGS}
    content = message.content
    if isinstance(content, list):
        content = [b for b in content if not (isinstance(b, dict) and b.get("type") in REASONING_BLOCKS)]
    if kwargs == message.additional_kwargs and content == message.content:
        return message
    if not _has_text(content) and not message.tool_calls:
        content = PLACEHOLDER
    return message.model_copy(update={"additional_kwargs": kwargs, "content": content})


def _has_text(content: str | list[Any]) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    return any((isinstance(b, str) and b.strip()) or (isinstance(b, dict) and b.get("type") == "text"
                                                       and str(b.get("text", "")).strip()) for b in content)


def strip_reasoning(messages: Sequence[BaseMessage], state: Sequence[BaseMessage],
                    windows: Iterable[Window]) -> tuple[list[BaseMessage], int]:
    """state 의 구간들([start, end))에 든 AI 메시지의 생각을 요청 사본 messages 에서 뺀다. (사본, 실제로 바뀐 개수).

    구간은 state 인덱스로 받고, messages 에서는 메시지 id 로 찾는다. 요약(core 자리)이 앞쪽을 요약 메시지로 바꿔 요청의
    위치가 state 와 달라져도 남은 메시지의 id 는 그대로라 위치를 바꿔 계산할 필요가 없다. state 와 messages 는 바꾸지 않는다.
    """
    targets = {m.id for start, end in windows for m in state[start:end] if isinstance(m, AIMessage) and m.id}
    result: list[BaseMessage] = []
    changed = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.id in targets:
            stripped = drop_reasoning(message)
            changed += stripped is not message
            message = stripped
        result.append(message)
    return result, changed


def next_record(saved: PromptRecord | None, current: Fingerprint,
                state: Sequence[BaseMessage]) -> tuple[PromptRecord, str | None]:
    """(이번 요청의 기록, 안내 사유). 사유는 'prompt'(이전해 옛 생각을 뺀다), 'model'(모델만 바뀜), None 이다.

    state 는 이번 요청 직전의 메시지다. 앞선 AI 응답이 없으면(새 thread, 첫 호출이 실패한 thread) 뺄 생각이 없어 알리지 않는다.
    """
    resumed = any(isinstance(m, AIMessage) for m in state)
    if saved is None:
        # 새 thread 의 첫 호출이거나, 지문 기록이 생기기 전의 옛 thread 를 이어 왔다(그때의 system 을 모르니 이전한다)
        return PromptRecord(current, len(state) if resumed else None), "prompt" if resumed else None
    if saved.fingerprint == current:
        return saved, None
    if resumed and not saved.fingerprint.same_prompt(current):
        return PromptRecord(current, len(state)), "prompt"  # 두 번째 이전이면 migrated_at 이 커진다
    return PromptRecord(current, saved.migrated_at), "model" if resumed else None


def _thread_id() -> str | None:
    try:
        return (get_config().get("configurable") or {}).get("thread_id")
    except RuntimeError:  # 그래프 밖에서 불렸다
        return None


def _notify(event: dict[str, Any]) -> None:
    """custom 스트림 이벤트. 그래프 밖이면 아무것도 하지 않는다(custom 모드가 아니면 writer 가 버린다)."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass


class PromptFingerprintMiddleware(AgentMiddleware):
    """thread 의 지문을 기록하고, 바뀌었으면 '이전'해 그 앞 AI 메시지의 생각을 요청 사본에서 뺀다.

    사용자 middleware 목록에서 TodoList 와 ToolArgsMiddleware 뒤에 둔다. wrap_model_call 은 앞이 바깥쪽이라, 우리 system
    절(core 와 TodoList 가 붙인 것까지)과 모델이 받는 도구 목록을 본다. 그보다 안쪽(tail)인 메모리·스킬 절은 thread 마다
    고정이라 괜찮다.
    안내 이벤트는 기록이 바뀔 때만 나가므로 thread 마다 이전 한 번에 한 번이다.
    """

    def __init__(self, store: PromptStore) -> None:
        super().__init__()
        self.store = store

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._prepare(request))

    async def awrap_model_call(self, request: ModelRequest,
                               handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(self._prepare(request))

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        thread_id = _thread_id()
        if thread_id is None:
            return request
        state: list[BaseMessage] = list(request.state.get("messages", []))
        saved = self.store.get(thread_id)
        record, event = next_record(saved, fingerprint(request), state)
        if record is not saved:
            self.store.put(thread_id, record)
        stripped = 0
        if record.migrated_at:
            messages, stripped = strip_reasoning(request.messages, state, [(0, record.migrated_at)])
            if stripped:
                request = request.override(messages=cast("list[AnyMessage]", messages))
        if event is not None:
            _notify({"type": "prompt_migrated", "reason": event, "stripped": stripped})
        return request
