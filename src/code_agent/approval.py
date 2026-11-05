"""승인 결정. 그래프는 승인 대상 도구마다 항상 interrupt 하고, 클라이언트가 모드에 따라 자동으로 정하거나 사람에게 묻는다.

결정은 approve 와 reject 만 쓴다. edit 은 고친 인자가 재검증 없이 바로 실행되기 때문이다(pjt Day08 05).
모델을 부르지 않는 순수 함수라 경계를 단위 시험으로 확인한다.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterable

MODES = ("manual", "auto-edit", "yolo")
FILE_TOOLS = frozenset({"write_file", "edit_file", "delete"})
APPROVE = {"type": "approve"}
REJECT_MESSAGE = ("사용자가 이 도구 호출을 거부했습니다. 같은 호출을 다시 시도하지 말고, "
                  "대안을 제시하거나 사용자에게 원하는 방향을 물어보세요.")

# 모드와 관계없이 자동 승인하는 읽기 전용 명령. 인터프리터·빌드 도구(python, bash, uv, node, make)는
# 넣지 않는다(dcode THREAT_MODEL T13). find(-exec, -delete), rg(--pre) 처럼 실행 옵션이 있는 도구도 뺀다.
READ_ONLY = frozenset({"ls", "pwd", "cat", "head", "tail", "wc", "which"})
READ_ONLY_GIT = frozenset({"status", "diff", "log", "show"})
_PRINTS_CONTENT = frozenset({"cat", "head", "tail"})
_SHELL_META = re.compile(r"[;&|<>`$(){}\n\r]")  # 연결·파이프·리디렉션·치환·변수 확장
_ENV_FILE = re.compile(r"(^|[/:])\.env")  # .env, sub/.env, .env.local, HEAD:.env
_DRIVE = re.compile(r"^[A-Za-z]:")


def is_read_only_command(command: str) -> bool:
    if not command.strip() or _SHELL_META.search(command):
        return False
    try:
        # 역슬래시를 / 로 바꿔 PowerShell 의 ..\ 경로도 같은 규칙으로 검사한다.
        tokens = shlex.split(command.replace("\\", "/"))
    except ValueError:
        return False
    if not tokens:
        return False
    head, args = tokens[0], tokens[1:]
    if head == "git":
        # 하위 명령이 바로 와야 한다. git -c core.fsmonitor=... status 같은 전역 옵션은 명령을 실행할 수 있다.
        if not args or args[0] not in READ_ONLY_GIT:
            return False
        args = args[1:]
        if any(a.startswith("--output") for a in args):
            return False
    elif head not in READ_ONLY:
        return False
    if head in _PRINTS_CONTENT and any(ch in a for a in args for ch in "*?["):
        return False  # 와일드카드는 .env 를 포함할 수 있다
    return all(_inside_workspace(a) for a in args)


def _inside_workspace(arg: str) -> bool:
    """작업 폴더 밖이나 .env 를 가리키는 인자는 읽기라도 사람에게 묻는다."""
    if arg.startswith(("/", "~")) or _DRIVE.match(arg) or ".." in arg.split("/"):
        return False
    return not _ENV_FILE.search(arg)


def touches_git_dir(path: str) -> bool:
    """.git/config 를 바꾸면 자동 승인되는 git status 로도 명령을 실행할 수 있다(core.fsmonitor 등)."""
    return ".git" in path.replace("\\", "/").split("/")


def auto_decision(action: dict, mode: str) -> dict | None:
    """자동으로 정할 수 있으면 결정을, 사람에게 물어야 하면 None 을 돌려준다."""
    name, args = action["name"], action.get("args") or {}
    if mode == "yolo":
        return APPROVE
    if name == "execute":
        return APPROVE if is_read_only_command(str(args.get("command", ""))) else None
    if mode == "auto-edit" and name in FILE_TOOLS and not touches_git_dir(str(args.get("file_path", ""))):
        return APPROVE
    return None


def decide_all(action_requests: Iterable[dict], mode: str, ask: Callable[[dict], dict],
               on_decision: Callable[[dict, dict, bool], None] | None = None) -> list[dict]:
    """요청 순서 그대로 결정을 만든다. HITL 미들웨어는 결정을 순서로 도구 호출과 짝짓는다.

    on_decision(action, decision, auto) 는 결정마다 불린다. auto 는 사람에게 묻지 않고 정했는지다(트레이스 기록용).
    """
    if mode not in MODES:
        raise ValueError(f"승인 모드는 {', '.join(MODES)} 중 하나여야 합니다: {mode!r}")
    decisions = []
    for action in action_requests:
        decision = auto_decision(action, mode)
        auto = decision is not None
        if decision is None:
            decision = ask(action)
        if on_decision is not None:
            on_decision(action, decision, auto)
        decisions.append(decision)
    return decisions


_REASON_PREFIX = " 사용자 의견: "


def reject(reason: str = "") -> dict:
    message = f"{REJECT_MESSAGE}{_REASON_PREFIX}{reason}" if reason else REJECT_MESSAGE
    return {"type": "reject", "message": message}


def reject_reason(decision: dict) -> str:
    """reject() 가 메시지에 붙인 사용자 의견. 없으면 빈 문자열."""
    _, _, reason = str(decision.get("message", "")).partition(_REASON_PREFIX)
    return reason
