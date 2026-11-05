"""도구 호출 가드. permissions= 는 셸 백엔드와 함께 쓰면 NotImplementedError 라서(0단계 b) 미들웨어로 막는다."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file", "delete"})
_ENV_TEMPLATES = (".example", ".sample", ".template")


def is_env_path(path: str) -> bool:
    """.env 와 .env.* 는 막고, 내용이 비밀값이 아닌 .env.example 같은 견본은 허용한다."""
    name = PurePosixPath(path.replace("\\", "/")).name
    return name == ".env" or (name.startswith(".env.") and not name.endswith(_ENV_TEMPLATES))


def strip_env_matches(text: str) -> str:
    """grep 결과에서 .env 파일의 줄을 뺀다. 세 출력 형식(경로 목록, 'path: N', 'path:' + 들여쓴 줄)을 모두 다룬다."""
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("/"):
            path = line.split(":", 1)[0]
            skipping = is_env_path(path) and line.endswith(":")
            if is_env_path(path):
                continue
        elif skipping:
            continue
        kept.append(line)
    return "\n".join(kept) if kept else "No matches found"


class DotenvGuardMiddleware(AgentMiddleware):
    """파일 도구가 .env 를 읽거나 쓰지 못하게 하고, grep 결과에서 .env 내용을 지운다. 셸은 승인으로 막는다."""

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        path = str(args.get("file_path", ""))
        if name in PATH_TOOLS and is_env_path(path):
            return ToolMessage(content=f"Error: 권한 거부 - {path} 는 비밀값 파일이라 읽거나 쓸 수 없습니다.",
                               tool_call_id=call["id"], name=name, status="error")
        result = handler(request)
        if name == "grep" and isinstance(result, ToolMessage) and isinstance(result.content, str):
            return result.model_copy(update={"content": strip_env_matches(result.content)})
        return result
