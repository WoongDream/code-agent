"""메모리(AGENTS.md)와 스킬.

- 사용자 공통 메모리는 <data_dir>/memory/AGENTS.md 다(기본 ~/.code-agent/memory/).
  백엔드 라우트 /.code-agent-user/ 로 잇는다(backend.make_backend). 프로젝트 메모리는 작업 폴더의
  /AGENTS.md 다. 둘 다 없으면 만들지 않는다.
- deepagents MemoryMiddleware 는 thread 의 첫 요청 때 한 번 읽어 state(memory_contents)에 두고
  매 모델 호출의 system 끝에 붙인다. 이 값은 체크포인트에 남는다.
  - 그래서 대화 중 system 이 바이트 단위로 같다(프롬프트 캐시와 Opus 5.5 의 보존된 thinking).
  - --resume 해도 다시 읽지 않는다.
  - 대화 중 고친 AGENTS.md 는 다음 /new 부터 반영된다.
    바로 다시 읽으면 system 이 바뀌어 앞의 생각이 모두 무효가 된다.
- AgentMemoryMiddleware 는 그 MemoryMiddleware 를 같은 이름으로 대신한다. deepagents 는
  사용자 middleware 가 기본 것과 이름이 같으면 그 자리에서 바꾼다
  (graph.py _apply_custom_middleware).
  - 프로젝트 AGENTS.md 는 사람이 신뢰한 내용(MemoryTrust 의 sha256)일 때만 읽는다.
    저장소가 심은 지시가 system 에 들어가는 것을 막는다(S11).
    사용자 공통 메모리는 사용자 것이라 묻지 않는다.
  - 읽은 직후 memory_loaded 사건을, 메모리 파일 쓰기가 실제로 실행되면 memory_write 사건을
    custom 스트림으로 보낸다(트레이스와 작업 기록이 받는다).
    읽기와 같은 곳에서 내므로 thread 마다 한 번이고, --resume 하면 읽지 않으니 다시 나오지 않는다.
- 스킬은 /.code-agent-user/skills/ 와 /.agents/skills/ 의 <이름>/SKILL.md 다(deepagents
  SkillsMiddleware). 같은 이름이면 프로젝트가 이긴다. 메모리처럼 thread 마다 한 번 읽는다.
  - 폴더가 없는 것은 오류가 아니다. deepagents 는 그것을 system 의 경고 절로 넣으므로
    QuietSkillsMiddleware 가 뺀다.
- deepagents 는 그런 경고를 logging 으로도 남긴다. 처리기가 없으면 파이썬이 stderr 로 찍어
  상태 줄과 입력 화면을 깨므로 <data_dir>/logs/deepagents.log 로 보낸다(library_logs).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse
from deepagents.middleware.memory import MemoryMiddleware, MemoryState, MemoryStateUpdate
from deepagents.middleware.skills import SkillsMiddleware, SkillsState, SkillsStateUpdate
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command

from code_agent.approval import FILE_TOOLS, PROJECT_MEMORY, is_memory_file, is_user_memory
from code_agent.backend import USER_MEMORY_ROUTE

MEMORY_DIR = "memory"  # <data_dir> 아래 사용자 공통 메모리 폴더
MEMORY_FILE = "AGENTS.md"
USER_MEMORY = USER_MEMORY_ROUTE + MEMORY_FILE
# 이 순서로 system 에 붙는다. 프로젝트 것이 뒤라 사용자 공통보다 구체적인 지시가 나중에 온다
MEMORY_SOURCES = (USER_MEMORY, PROJECT_MEMORY)
USER_SKILLS = USER_MEMORY_ROUTE + "skills/"
PROJECT_SKILLS = "/.agents/skills/"
SKILL_SOURCES = (USER_SKILLS, PROJECT_SKILLS)  # 같은 이름이면 뒤쪽이 이긴다
# system 의 스킬 절에 보일 이름. 기본 이름(폴더 이름에서 만든 'Agents')은 '이 PC 의 모든 에이전트
# 도구가 같이 쓰는 것'으로 설명돼 프로젝트 폴더와 맞지 않는다
SKILL_LABELS = {USER_SKILLS: "User", PROJECT_SKILLS: "Project"}
WARN_CHARS = 8_000  # 합계가 이보다 크면 경고한다. 매 요청 system 에 들어가 캐시 읽기 비용이 는다
LOG_FILE = "deepagents.log"
_MISSING = "path_not_found"  # 스킬 폴더가 없을 때 deepagents 가 남기는 오류의 끝
REMEMBER_PROMPT = (
    "이번 대화에서 확인된, 앞으로도 지킬 규칙이나 사실만 골라 프로젝트 것은 /AGENTS.md 에, "
    "모든 프로젝트 공통은 /.code-agent-user/AGENTS.md 에 짧은 항목으로 덧붙여라. "
    "기존 항목은 지우지 않는다(파일이 없으면 write_file 로 만든다). "
    "비밀값, 일회성 사실, 코드에서 바로 알 수 있는 것은 쓰지 않는다.")
REMEMBER_DONE = "메모리는 다음 /new 부터 반영된다."


def memory_dir(data_dir: Path) -> Path:
    """사용자 공통 메모리 폴더(USER_MEMORY_ROUTE 가 가리키는 곳)."""
    return data_dir / MEMORY_DIR


def remember_prompt(memo: str = "") -> str:
    """/remember 가 보내는 사용자 메시지. memo 는 사용자가 덧붙인 기억할 것이다."""
    memo = memo.strip()
    return REMEMBER_PROMPT + (f"\n\n사용자가 기억하라고 한 것: {memo}" if memo else "")


def decode_memory(data: bytes) -> str:
    """메모리 파일 내용. UTF-8 이 아닌 바이트는 바꿔 읽고 앞의 BOM 은 뗀다.

    deepagents 는 그대로 decode 해 UTF-8 이 아니면 예외로 턴이 실패한다(이 thread 는 계속).
    """
    return data.decode("utf-8", errors="replace").removeprefix("﻿")


def content_sha256(data: bytes) -> str:
    """신뢰 기록에 쓰는 파일 내용의 sha256."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class MemoryTrust:
    """프로젝트 /AGENTS.md 를 메모리로 읽어도 되는 내용.

    App 이 새 thread 를 시작할 때(시작, /new) 사람에게 묻고 정한다(cli.App.start_memory).
    agent 와 같은 객체를 나눠 쓰고, AgentMemoryMiddleware 가 thread 의 첫 요청 때 본다.
    파일이 그사이 바뀌면 해시가 달라 읽지 않는다.

    Attributes:
        sha256: 신뢰한 내용의 해시(content_sha256). None 이면 프로젝트 AGENTS.md 를 읽지 않는다.
    """

    sha256: str | None = None

    def allows(self, data: bytes) -> bool:
        """이 내용을 읽어도 되는지."""
        return self.sha256 is not None and self.sha256 == content_sha256(data)


def _emit(event: dict[str, Any]) -> None:
    """custom 스트림 사건. 그래프 밖(단위 시험)이면 아무것도 하지 않는다."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass


def _succeeded(result: ToolMessage | Command) -> bool:
    """도구가 성공했는지. deepagents 파일 도구는 오류를 'Error: ...' 글로도 돌려준다."""
    if isinstance(result, Command):
        update = result.update if isinstance(result.update, dict) else {}
        messages = [m for m in update.get("messages") or [] if isinstance(m, ToolMessage)]
        if not messages:
            return True
        result = messages[-1]
    return result.status != "error" and not result.text.startswith("Error")


class AgentMemoryMiddleware(MemoryMiddleware):
    """신뢰한 프로젝트 AGENTS.md 만 읽고, 읽기·쓰기를 사건으로 남기는 MemoryMiddleware.

    이름이 MemoryMiddleware 라 create_deep_agent(memory=...) 가 만든 기본 것을 그 자리에서 바꾼다
    (system 의 맨 끝, Anthropic 이면 cache_control 도 기본과 같다).
    system 에 붙이는 형식(modify_request)은 그대로 쓰고 sources 순서로 붙인다. 그래서 --resume 한
    thread 는 신뢰 여부와 관계없이 그때 읽은 것을 그대로 쓴다.

    Args:
        backend: 메인 agent 와 같은 백엔드(USER_MEMORY_ROUTE 가 있어야 한다).
        sources: 읽을 메모리 경로(MEMORY_SOURCES).
        trust: 프로젝트 AGENTS.md 를 읽어도 되는 내용(MemoryTrust).
    """

    def __init__(self, *, backend: BackendProtocol, sources: Sequence[str],
                 trust: MemoryTrust) -> None:
        super().__init__(backend=backend, sources=list(sources), add_cache_control=True)
        self.trust = trust

    @property
    def name(self) -> str:
        """deepagents 기본 MemoryMiddleware 자리를 이름으로 대신한다."""
        return "MemoryMiddleware"

    def before_agent(self, state: MemoryState, runtime: Runtime,
                     config: RunnableConfig) -> MemoryStateUpdate | None:
        """thread 의 첫 요청이면 메모리를 읽어 state 에 둔다. 이미 읽었으면 하지 않는다."""
        if "memory_contents" in state:
            return None
        return self._loaded(self._backend.download_files(list(self.sources)))

    async def abefore_agent(self, state: MemoryState, runtime: Runtime,
                            config: RunnableConfig) -> MemoryStateUpdate | None:
        """before_agent 의 비동기판."""
        if "memory_contents" in state:
            return None
        return self._loaded(await self._backend.adownload_files(list(self.sources)))

    def _loaded(self, responses: Sequence[FileDownloadResponse]) -> MemoryStateUpdate:
        """읽은 결과로 memory_contents 를 만들고 memory_loaded 사건을 보낸다.

        없는 파일은 건너뛴다. 읽지 못한 파일(폴더 등)은 기본 구현처럼 예외를 내지 않고 사건에
        남긴다. 예외를 내면 이 thread 의 모든 턴이 실패한다.
        """
        contents: dict[str, str] = {}
        untrusted: list[str] = []
        errors: list[dict[str, str]] = []
        for path, response in zip(self.sources, responses, strict=True):
            if response.error is not None:
                if response.error != "file_not_found":
                    errors.append({"path": path, "error": str(response.error)})
                continue
            if response.content is None:
                continue
            if path == PROJECT_MEMORY and not self.trust.allows(response.content):
                untrusted.append(path)
                continue
            contents[path] = decode_memory(response.content)
        event: dict[str, Any] = {
            "type": "memory_loaded",
            "files": [{"path": path, "chars": len(text)} for path, text in contents.items()],
            "total_chars": sum(len(text) for text in contents.values())}
        if untrusted:
            event["untrusted"] = untrusted
        if errors:
            event["errors"] = errors
        _emit(event)
        return MemoryStateUpdate(memory_contents=contents)

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command],
                       ) -> ToolMessage | Command:
        """메모리 파일 쓰기가 실제로 실행돼 성공하면 memory_write 사건을 보낸다.

        가드·계획 차단·사람 거부로 막힌 호출은 여기까지 오지 않는다(이 middleware 가 안쪽이다).
        """
        result = handler(request)
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        path = str(args.get("file_path", ""))
        if name in FILE_TOOLS and is_memory_file(path) and _succeeded(result):
            _emit({"type": "memory_write", "path": path, "tool": name,
                   "scope": "user" if is_user_memory(path) else "project"})
        return result


def _without_missing(update: SkillsStateUpdate | None) -> SkillsStateUpdate | None:
    if update is None:
        return None
    errors = [e for e in update.get("skills_load_errors") or [] if not e.endswith(_MISSING)]
    return SkillsStateUpdate(skills_metadata=update["skills_metadata"], skills_load_errors=errors)


class QuietSkillsMiddleware(SkillsMiddleware):
    """스킬 폴더가 없는 것을 오류로 보지 않는 SkillsMiddleware.

    기본 구현은 없는 폴더마다 'Cannot load skills from ...' 를 system 의 경고 절에 넣는다.
    스킬 폴더는 대부분 없으므로 그 경고는 매번 system 을 늘리는 잡음이다. 다른 오류는 그대로 둔다.
    이름이 SkillsMiddleware 라 create_deep_agent(skills=...) 의 기본 것을 그 자리에서 바꾼다.
    """

    @property
    def name(self) -> str:
        """deepagents 기본 SkillsMiddleware 자리를 이름으로 대신한다."""
        return "SkillsMiddleware"

    def before_agent(self, state: SkillsState, runtime: Runtime,
                     config: RunnableConfig) -> SkillsStateUpdate | None:
        """thread 의 첫 요청이면 스킬 목록을 읽는다. 없는 폴더의 오류는 뺀다."""
        return _without_missing(super().before_agent(state, runtime, config))

    async def abefore_agent(self, state: SkillsState, runtime: Runtime,
                            config: RunnableConfig) -> SkillsStateUpdate | None:
        """before_agent 의 비동기판."""
        return _without_missing(await super().abefore_agent(state, runtime, config))


@contextmanager
def library_logs(data_dir: Path) -> Iterator[Path]:
    """그동안 deepagents 로거의 경고를 <data_dir>/logs/deepagents.log 로 보낸다.

    처리기가 없으면 파이썬이 WARNING 이상을 stderr 로 찍어(logging.lastResort) 상태 줄과 입력
    화면을 깬다. 스킬 폴더가 없을 때도 deepagents 가 경고를 남긴다. 끝나면 처리기를 떼고 원래대로
    되돌린다(시험이 임시 폴더 처리기를 남기지 않게).

    Yields:
        로그 파일 경로.
    """
    logger = logging.getLogger("deepagents")
    path = data_dir / "logs" / LOG_FILE
    handler: logging.Handler
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8", delay=True)
    except OSError:
        handler = logging.NullHandler()  # 로그를 못 남겨도 화면은 깨지 않게 한다
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield path
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
        handler.close()


def disk_path(source: str, *, workspace: Path, data_dir: Path) -> Path:
    """메모리·스킬 가상 경로(MEMORY_SOURCES, SKILL_SOURCES)의 실제 위치."""
    if source.startswith(USER_MEMORY_ROUTE):
        return memory_dir(data_dir) / source.removeprefix(USER_MEMORY_ROUTE)
    return workspace / source.lstrip("/")


def read_memory(path: Path) -> bytes | None:
    """메모리 파일 바이트. 없거나 읽지 못하면 None."""
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def pending_chars(*, workspace: Path, data_dir: Path, trust: MemoryTrust) -> int:
    """새 thread 의 첫 요청 때 읽힐 메모리의 글자 수(사용자 공통 + 신뢰한 프로젝트 것)."""
    total = 0
    for source in MEMORY_SOURCES:
        data = read_memory(disk_path(source, workspace=workspace, data_dir=data_dir))
        if data is not None and (source != PROJECT_MEMORY or trust.allows(data)):
            total += len(decode_memory(data))
    return total


def loaded_chars(values: Mapping[str, Any]) -> int | None:
    """thread 가 읽어 둔 메모리의 글자 수. 아직 읽지 않았으면 None."""
    contents = values.get("memory_contents")
    if not isinstance(contents, dict):
        return None
    return sum(len(str(text)) for text in contents.values())


def memory_report(values: Mapping[str, Any], *, workspace: Path, data_dir: Path,
                  trust: MemoryTrust) -> list[str]:
    """/memory 가 보일 줄들: 메모리 파일마다 실제 위치, 이 thread 가 읽은 글자 수, 그 뒤 디스크에서
    바뀌었는지와 스킬 목록.

    Args:
        values: agent.get_state(config).values. memory_contents·skills_metadata 를 본다.
        workspace: 작업 폴더.
        data_dir: 데이터 폴더(사용자 공통 메모리 위치).
        trust: 아직 읽지 않은 thread 에서 프로젝트 AGENTS.md 를 읽을지 보일 때 쓴다.

    Returns:
        rich 마크업이 없는 글 줄 목록.
    """
    contents = values.get("memory_contents")
    loaded = contents if isinstance(contents, dict) else None
    lines = [("메모리: 첫 요청 때 읽어 이 세션 동안 그대로 쓴다. "
              "고친 내용은 다음 /new 부터 반영된다.") if loaded is not None else
             "메모리: 이 세션은 아직 읽지 않았다(첫 요청 때 읽는다)."]
    for source in MEMORY_SOURCES:
        path = disk_path(source, workspace=workspace, data_dir=data_dir)
        data = read_memory(path)
        lines.append(f"- {source} → {path}")
        lines.append(f"  {_file_state(source, data, loaded, trust)}")
    total = loaded_chars(values)
    if total is None:
        total = pending_chars(workspace=workspace, data_dir=data_dir, trust=trust)
    over = f" (경고: {WARN_CHARS:,}자 초과)" if total > WARN_CHARS else ""
    lines.append(f"합계 {total:,}자{over}")
    skills = values.get("skills_metadata")
    if isinstance(skills, list) and skills:
        lines.append(f"스킬 {len(skills)}개:")
        lines += [f"- {s.get('name')}: {s.get('description')} ({s.get('path')})"
                  for s in skills if isinstance(s, dict)]
    else:
        folders = ", ".join(f"{s} → {disk_path(s, workspace=workspace, data_dir=data_dir)}"
                            for s in SKILL_SOURCES)
        lines.append(("스킬 없음" if isinstance(skills, list) else "스킬: 첫 요청 때 읽는다")
                     + f" ({folders}, <이름>/SKILL.md)")
    return lines


def _file_state(source: str, data: bytes | None, loaded: dict | None,
                trust: MemoryTrust) -> str:
    """메모리 파일 하나의 상태 한 줄."""
    disk = None if data is None else decode_memory(data)
    if loaded is None:  # 아직 읽지 않은 thread
        if disk is None:
            return "없음"
        if source == PROJECT_MEMORY and data is not None and not trust.allows(data):
            return f"디스크 {len(disk):,}자 · 신뢰하지 않아 읽지 않는다"
        return f"디스크 {len(disk):,}자 · 첫 요청 때 읽는다"
    if source not in loaded:
        if disk is None:
            return "없음"
        why = ("신뢰하지 않았거나 세션을 시작한 뒤 생겼다" if source == PROJECT_MEMORY
               else "세션을 시작한 뒤 생겼다")
        return f"읽지 않음({why}) · 디스크 {len(disk):,}자 · 다음 /new 부터 반영"
    text = str(loaded[source])
    if disk is None:
        change = "디스크에서 지워졌다(다음 /new 부터 반영)"
    elif disk == text:
        change = "디스크와 같다"
    else:
        change = f"디스크에서 바뀌었다({len(disk):,}자, 다음 /new 부터 반영)"
    return f"읽음 {len(text):,}자 · {change}"
