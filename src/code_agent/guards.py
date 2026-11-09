"""도구 호출 가드. permissions= 는 셸 백엔드와 함께 쓰면 NotImplementedError 라서(0단계 b)
미들웨어로 막는다.

Windows 는 이름이 달라도 같은 파일을 연다: 대소문자, 끝 공백·점(".env "),
스트림(".env::$DATA"), 8.3 짧은 이름(ENV~1).
그래서 이름을 정규화해 판정하고,
작업 폴더의 실제 파일이 있으면 그 긴 이름(Path.resolve)으로 다시 판정한다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file", "delete"})
_ENV_TEMPLATES = (".example", ".sample", ".template")
SHORT_NAME = re.compile(r"~\d")  # 8.3 짧은 이름. ENV~1 이 .env 일 수 있다
# 개인 키·자격 증명 파일. 읽기는 막지 않고 사람에게 묻는다(사용자 결정). 공개 키(.pub)는 뺀다
_SENSITIVE = re.compile(
    r"id_(rsa|dsa|ecdsa|ed25519)(?!.*\.pub$).*|.*\.(pem|key|p12|pfx)|credentials.*\.json|.*service-account.*\.json"
    r"|\.npmrc|\.pypirc|\.netrc|\.git-credentials|secrets\..+")
SENSITIVE_READ = "개인 키·자격 증명일 수 있는 파일 읽기"


def file_name(path: str) -> str:
    """Windows 가 같은 파일로 여는 변형을 하나로 모은 이름:
    casefold, ':' 뒤 스트림 제거, 끝 공백·점 제거.
    """
    name = PurePosixPath(path.replace("\\", "/")).name
    return name.split(":", 1)[0].rstrip(" .").casefold()


def is_env_path(path: str) -> bool:
    """.env 와 .env.* 는 막고,
    내용이 비밀값이 아닌 .env.example 같은 견본은 허용한다(대소문자 무관).
    """
    name = file_name(path)
    return name == ".env" or (name.startswith(".env.") and not name.endswith(_ENV_TEMPLATES))


def is_sensitive_path(path: str) -> bool:
    """개인 키·자격 증명 파일 이름인지
    (id_rsa, *.pem, credentials*.json, .netrc, .aws/credentials 등).
    """
    parts = [p.casefold() for p in PurePosixPath(path.replace("\\", "/")).parts]
    return bool(_SENSITIVE.fullmatch(file_name(path))) or parts[-2:] == [".aws", "credentials"]


def strip_env_matches(text: str) -> str:
    """grep 결과에서 .env 파일의 줄을 뺀다.

    세 출력 형식(경로 목록, 'path: N', 'path:' + 들여쓴 줄)을 모두 다룬다.
    """
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
    """파일 도구가 .env 를 읽거나 쓰지 못하게 하고, grep 결과에서 .env 내용을 지운다.
    셸은 승인으로 막는다.

    workspace 를 주면 가상 경로를 실제 파일로 바꿔 긴 이름으로도 판정한다(8.3 이름 대책).
    대상이 없으면(새로 쓰기) 8.3 모양 이름(~숫자)은 보수적으로 막는다.
    개인 키·자격 증명 파일 읽기는 승인 프롬프트로 묻는다.
    """

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__()
        self.workspace = workspace

    def _real_name(self, path: str) -> str | None:
        """작업 폴더 안 실제 파일의 긴 이름. 없거나 작업 폴더를 모르면 None."""
        if self.workspace is None:
            return None
        target = Path(self.workspace, path.replace("\\", "/").lstrip("/"))
        try:
            return target.resolve().name if target.exists() else None
        except OSError:
            return None

    def blocks(self, path: str) -> bool:
        """.env 로 열리는 경로인지."""
        if is_env_path(path):
            return True
        real = self._real_name(path)
        if real is not None:
            return is_env_path(real)
        return self.workspace is not None and bool(SHORT_NAME.search(file_name(path)))

    def asks(self, name: str, path: str) -> bool:
        """읽기 전에 사람에게 물을 개인 키·자격 증명 파일인지."""
        if name != "read_file":
            return False
        real = self._real_name(path)
        return is_sensitive_path(path) or (real is not None and is_sensitive_path(real))

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command],
                       ) -> ToolMessage | Command:
        """.env 경로는 오류 ToolMessage 로 막고, 개인 키·자격 증명 읽기는 interrupt 로 묻는다.

        grep 결과에서는 .env 파일의 줄을 지운다.
        """
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        path = str(args.get("file_path", ""))
        if name in PATH_TOOLS and self.blocks(path):
            return ToolMessage(content=f"Error: 권한 거부 - {path} 는 비밀값 파일이라 "
                                       "읽거나 쓸 수 없습니다.",
                               tool_call_id=call["id"], name=name, status="error")
        if self.asks(name, path):
            # HITL 과 같은 모양이라 CLI 의 승인 흐름(run_turn → decide)이 그대로 처리하고,
            # 재개 값이 여기로 돌아온다
            answer = interrupt({"action_requests": [{"name": name, "args": args,
                                                     "description": SENSITIVE_READ}],
                                "review_configs": [{"action_name": name,
                                                    "allowed_decisions": ["approve", "reject"]}]})
            decision = ((answer or {}).get("decisions") or [{}])[0]
            if decision.get("type") != "approve":
                return ToolMessage(content=str(decision.get("message")
                                               or "사용자가 이 파일 읽기를 거부했습니다."),
                                   tool_call_id=call["id"], name=name, status="error")
        result = handler(request)
        if name == "grep" and isinstance(result, ToolMessage) and isinstance(result.content, str):
            return result.model_copy(update={"content": strip_env_matches(result.content)})
        return result
