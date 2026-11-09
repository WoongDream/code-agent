"""승인 결정. 그래프는 승인 대상 도구마다 항상 interrupt 하고,
클라이언트가 모드에 따라 자동으로 정하거나 사람에게 묻는다.

결정은 approve 와 reject 만 쓴다. edit 은 고친 인자가 재검증 없이 바로 실행되기 때문이다
(pjt Day08 05).
모델을 부르지 않는 순수 함수라 경계를 단위 시험으로 확인한다.

계획 모드(planning.py)의 상태는 PlanContext 로 받는다. 계획 승인은 모든 모드에서 사람에게 묻고,
승인된 계획 밖 파일 쓰기도 모든 모드에서 묻는다. 계획 범위 판정(normalize_path, in_plan)도
승인 결정이 쓰므로 여기 둔다. planning 이 이 모듈을 쓰고 이 모듈은 planning 을 쓰지 않는다.

메모리 파일(memory.py)도 경로 판정을 여기 둔다.
- 사용자 공통 메모리(/.code-agent-user/ 아래) 쓰기는 yolo 를 포함한 모든 모드에서 묻는다.
  모든 프로젝트의 system 에 들어가 한 번 오염되면 계속 남기 때문이다(task8 결정 3).
- 메모리 파일(사용자 공통과 프로젝트 /AGENTS.md)은 코드 변경이 아니라서 계획 승인 전 차단과
  '계획에 없는 파일' 재승인에서 뺀다.
  제 승인 규칙(사용자 공통은 항상 묻기, 프로젝트는 모드대로)은 따른다.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from code_agent.backend import USER_MEMORY_ROUTE
from code_agent.guards import SHORT_NAME, is_sensitive_path

MODES = ("manual", "auto-edit", "yolo")
PROJECT_MEMORY = "/AGENTS.md"  # 프로젝트 메모리. 작업 폴더 루트의 AGENTS.md 하나다
USER_MEMORY_REASON = ("사용자 공통 메모리: 모든 프로젝트의 시스템 프롬프트에 들어간다. "
                      "그래서 승인 모드와 관계없이 묻는다.")
FILE_TOOLS = frozenset({"write_file", "edit_file", "delete"})
APPROVE = {"type": "approve"}
REJECT_MESSAGE = ("사용자가 이 도구 호출을 거부했습니다. 같은 호출을 다시 시도하지 말고, "
                  "대안을 제시하거나 사용자에게 원하는 방향을 물어보세요.")
PLAN_APPROVAL_TOOL = "request_plan_approval"
PLAN_REVIEW_FIRST = ("plan-reviewer 검토를 먼저 받아라. task(subagent_type=\"plan-reviewer\") 로 "
                     "계획을 검토받고 지적을 반영한 뒤 request_plan_approval 을 다시 불러라.")
PLAN_REJECT_MESSAGE = ("사용자가 계획을 승인하지 않았다. 사용자 의견을 반영해 "
                       "write_plan 으로 계획을 고치고 request_plan_approval 로 다시 승인을 "
                       "요청하라. 의견이 없으면 무엇을 바꿀지 사용자에게 물어라.")
OUT_OF_PLAN = ("계획에 없는 파일이다. 승인하면 계획 범위에 더한다"
               "(계획을 끝냈으면 /plan off 로 계획을 지운다).")
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class PlanContext:
    """승인 결정에 쓰는 계획 상태. planning.plan_context 가 agent state 로 만든다.

    Attributes:
        block_message: 비어 있지 않으면 계획 승인 전이라 쓰기를 막는다. 모델에 돌려줄 이유 문구다.
        reviewed: plan-reviewer 검토를 받았는지(status reviewed·revised·approved).
        approved: 사람이 계획을 승인했는지.
        files: 계획 범위. normalize_path 로 정규화한 경로이고 폴더는 / 로 끝난다.
    """

    block_message: str = ""
    reviewed: bool = False
    approved: bool = False
    files: frozenset[str] = frozenset()

# 모드와 관계없이 자동 승인하는 읽기 전용 명령.
# 인터프리터·빌드 도구(python, bash, uv, node, make)는 넣지 않는다(dcode THREAT_MODEL T13).
# find(-exec, -delete), rg(--pre) 처럼 실행 옵션이 있는 도구도 뺀다.
READ_ONLY = frozenset({"ls", "pwd", "cat", "head", "tail", "wc", "which"})
READ_ONLY_GIT = frozenset({"status", "diff", "log", "show"})
_PRINTS_CONTENT = frozenset({"cat", "head", "tail"})
_SHELL_META = re.compile(r"[;&|<>`$(){}\n\r]")  # 연결·파이프·리디렉션·치환·변수 확장
# .env, .ENV, sub/.env, .env.local, HEAD:.env
_ENV_FILE = re.compile(r"(^|[/:])\.env", re.IGNORECASE)
_DRIVE = re.compile(r"^[A-Za-z]:")
# PowerShell 공급자 경로(FileSystem::C:\…)와 드라이브(env:, HKCU:, Cert:, Variable:, C:).
# 작업 폴더 밖으로 본다
_PROVIDER = re.compile(r"::|^[A-Za-z][\w.]*:")
# PowerShell 은 둥근 따옴표를 따옴표로, 긴 대시를 '-' 로 읽는다.
# shlex 는 글자로 보므로 판정 전에 맞춘다
_PS_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
                            "\u201c": '"', "\u201d": '"', "\u201e": '"',
                            "\u2013": "-", "\u2014": "-", "\u2015": "-"})


def _hidden_chars(command: str) -> bool:
    """제어·서식(bidi, 폭 없는 문자)·비ASCII 공백. 셸마다 다르게 읽을 수 있어 묻는다."""
    return any(not ch.isascii() and unicodedata.category(ch) in ("Cc", "Cf", "Zs", "Zl", "Zp")
               for ch in command)


def _follows(arg: str) -> bool:
    """tail -f·-F·--follow 는 셸 제한 시간(120초)까지 돌아오지 않는다.

    GNU 는 긴 옵션을 앞부분만 써도 받는다(--f, --fo=name).
    tail 의 긴 옵션 중 f 로 시작하는 것은 --follow 뿐이다.
    """
    return arg.startswith("--f") or (arg.startswith("-") and not arg.startswith("--")
                                     and any(c in arg[1:] for c in "fF"))


def _waits(arg: str) -> bool:
    """PowerShell 의 cat(Get-Content) -Wait 도 tail -f 처럼 멈춘다.

    매개변수 이름은 대소문자 없이 앞부분만 써도 된다.
    """
    return arg.lower().startswith("-wai")


def is_read_only_command(command: str) -> bool:
    """모드와 관계없이 자동 승인할 읽기 전용 명령인지.

    연결·리디렉션이 없는 READ_ONLY 명령이나 git status·diff·log·show 이고,
    인자가 모두 작업 폴더 안(.env·개인 키 제외)일 때만 참이다.
    """
    if not command.strip() or _SHELL_META.search(command) or _hidden_chars(command):
        return False
    try:
        # 역슬래시를 / 로 바꿔 PowerShell 의 ..\ 경로도 같은 규칙으로 검사한다.
        tokens = shlex.split(command.translate(_PS_QUOTES).replace("\\", "/"))
    except ValueError:
        return False
    if not tokens:
        return False
    head, args = tokens[0], tokens[1:]
    git = head == "git"
    if git:
        # 하위 명령이 바로 와야 한다.
        # git -c core.fsmonitor=... status 같은 전역 옵션은 명령을 실행할 수 있다.
        if not args or args[0] not in READ_ONLY_GIT:
            return False
        args = args[1:]
        if any(a.startswith("--output") for a in args):
            return False
    elif head not in READ_ONLY:
        return False
    if head in _PRINTS_CONTENT and any(ch in a for a in args for ch in "*?["):
        return False  # 와일드카드는 .env 를 포함할 수 있다
    if ((head == "tail" and any(_follows(a) for a in args))
            or (head == "cat" and any(_waits(a) for a in args))):
        return False
    if ((head == "wc" and any(a.startswith("--f") for a in args))
            or any("files0-from" in a for a in args)):
        # wc --files0-from=F 는 F 의 내용을 파일 이름으로 읽어 오류 메시지로 내보낸다.
        # 약어(--files0, --f)도 같다.
        # wc 의 긴 옵션 중 f 로 시작하는 것은 이것뿐이다
        return False
    return all(_inside_workspace(a, git=git) for a in args)


def _inside_workspace(arg: str, *, git: bool = False) -> bool:
    """작업 폴더 밖·.env·개인 키를 가리키는 인자는 읽기라도 사람에게 묻는다.

    --opt=값 의 값도 경로로 본다.
    """
    values = [arg, arg.split("=", 1)[1]] if arg.startswith("-") and "=" in arg else [arg]
    return all(_safe_path(v, git=git) for v in values)


def _safe_path(arg: str, *, git: bool) -> bool:
    if arg.startswith(("/", "~")) or _DRIVE.match(arg) or ".." in arg.split("/"):
        return False
    if _ENV_FILE.search(arg) or any(is_sensitive_path(part) for part in arg.split(":") if part):
        return False
    # git 인자의 HEAD~1, HEAD:path 는 리비전이라
    # 셸 경로 규칙(8.3 이름·공급자 경로)을 적용하지 않는다
    return git or not (_PROVIDER.search(arg) or SHORT_NAME.search(arg))


def touches_git_dir(path: str) -> bool:
    """.git/config 를 바꾸면 자동 승인되는 git status 로도 명령을 실행할 수 있다
    (core.fsmonitor 등).
    """
    return ".git" in path.replace("\\", "/").split("/")


def normalize_path(path: str) -> str:
    """계획 범위를 비교할 가상 경로.

    역슬래시를 / 로 바꾸고, 앞에 / 를 붙이고, 빈 조각과 '.' 을 빼고, 대소문자를 접는다(Windows 는
    대소문자를 가리지 않는다). 폴더 표시인 끝의 / 는 남긴다. 예: 'Src/./A.py' → '/src/a.py'.
    """
    text = path.strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    normal = "/" + "/".join(parts)
    if parts and text.endswith("/"):
        normal += "/"
    return normal.casefold()


def in_plan(path: str, files: Iterable[str]) -> bool:
    """path 가 계획 파일이거나, 계획 폴더(/ 로 끝남) 안이거나, 글롭(*, ?, [)과 맞는지.

    files 는 normalize_path 로 정규화한 항목이다.
    """
    target = normalize_path(path)
    for entry in files:
        if _GLOB_CHARS & set(entry):
            if fnmatch.fnmatchcase(target, entry):
                return True
        elif entry.endswith("/"):
            if target.startswith(entry) or target + "/" == entry:
                return True
        elif target == entry:
            return True
    return False


def is_user_memory(path: str) -> bool:
    """사용자 공통 메모리(/.code-agent-user/ 아래) 경로인지.

    normalize_path 로 비교해 역슬래시·'.'·빈 조각·대소문자 변형도 잡는다. 백엔드는 대소문자를
    가려 라우팅하지만 더 넓게 잡으면 한 번 더 묻는 것뿐이라 안전한 쪽이다.
    """
    normal, route = normalize_path(path), USER_MEMORY_ROUTE.casefold()
    return normal.startswith(route) or normal + "/" == route


def is_memory_file(path: str) -> bool:
    """메모리 파일(사용자 공통 메모리 폴더 또는 프로젝트 /AGENTS.md)인지."""
    return is_user_memory(path) or normalize_path(path) == PROJECT_MEMORY.casefold()


def _memory_write(name: str, args: dict) -> bool:
    return name in FILE_TOOLS and is_memory_file(str(args.get("file_path", "")))


def is_write_action(name: str, args: dict) -> bool:
    """파일 쓰기·수정·삭제이거나 읽기 전용이 아닌 셸 명령인지(계획 승인 전에 막는 호출).

    메모리 파일 쓰기는 코드 변경이 아니라서 빼고, 계획 중에도 /remember 가 된다.
    """
    if name in FILE_TOOLS:
        return not _memory_write(name, args)
    return name == "execute" and not is_read_only_command(str(args.get("command", "")))


def plan_reason(action: dict, plan: PlanContext | None) -> str:
    """사람에게 물을 때 보일 계획 관련 이유. 승인된 계획 밖 파일 쓰기면 OUT_OF_PLAN, 아니면 ''.

    메모리 파일은 계획 범위와 관계없다.
    """
    name, args = action["name"], action.get("args") or {}
    path = str(args.get("file_path", ""))
    if (plan is not None and plan.approved and name in FILE_TOOLS
            and not is_memory_file(path) and not in_plan(path, plan.files)):
        return OUT_OF_PLAN
    return ""


def ask_reason(action: dict, plan: PlanContext | None = None) -> str:
    """승인 프롬프트에 보일, 모드와 관계없이 묻는 이유.

    사용자 공통 메모리 쓰기면 USER_MEMORY_REASON, 아니면 plan_reason.
    """
    name, args = action["name"], action.get("args") or {}
    if name in FILE_TOOLS and is_user_memory(str(args.get("file_path", ""))):
        return USER_MEMORY_REASON
    return plan_reason(action, plan)


def auto_decision(action: dict, mode: str, plan: PlanContext | None = None) -> dict | None:
    """자동으로 정할 수 있으면 결정을, 사람에게 물어야 하면 None 을 돌려준다.

    plan 이 있으면 모드보다 먼저 본다.
    - 계획 승인 요청: 검토 전이면 묻지 않고 거부하고(PLAN_REVIEW_FIRST),
      아니면 모든 모드에서 묻는다.
    - 승인 전 쓰기: 어차피 PlanModeMiddleware 가 막으므로 사람에게 묻지 않고 그 이유로 거부한다.
    - 승인된 계획 밖 파일 쓰기: auto-edit·yolo 에서도 묻는다(plan_reason).
    - 메모리 파일 쓰기는 위 두 계획 규칙에서 빠진다(is_write_action, plan_reason).
    사용자 공통 메모리(/.code-agent-user/) 쓰기는 yolo 를 포함한 모든 모드에서 묻는다.

    Args:
        action: HITL 승인 요청 하나({"name", "args", ...}).
        mode: 승인 모드(MODES).
        plan: 결정 시점의 계획 상태(planning.plan_context). 계획 모드를 안 쓰면 None.

    Returns:
        자동 결정({"type": "approve"} 또는 reject) 또는 사람에게 물어야 하면 None.
    """
    name, args = action["name"], action.get("args") or {}
    if name == PLAN_APPROVAL_TOOL:
        if plan is None or not plan.reviewed:
            return {"type": "reject", "message": PLAN_REVIEW_FIRST}
        return None
    if plan is not None:
        if plan.block_message and is_write_action(name, args):
            return {"type": "reject", "message": plan.block_message}
        if plan_reason(action, plan):
            return None
    if name in FILE_TOOLS and is_user_memory(str(args.get("file_path", ""))):
        return None
    if mode == "yolo":
        return APPROVE
    if name == "execute":
        return APPROVE if is_read_only_command(str(args.get("command", ""))) else None
    if (mode == "auto-edit" and name in FILE_TOOLS
            and not touches_git_dir(str(args.get("file_path", "")))):
        return APPROVE
    return None


def decide_all(action_requests: Iterable[dict], mode: str, ask: Callable[[dict], dict],
               on_decision: Callable[[dict, dict, bool], None] | None = None, *,
               plan: PlanContext | None = None) -> list[dict]:
    """요청 순서 그대로 결정을 만든다. HITL 미들웨어는 결정을 순서로 도구 호출과 짝짓는다.

    on_decision(action, decision, auto) 는 결정마다 불린다.
    auto 는 사람에게 묻지 않고 정했는지다(트레이스 기록용).

    Args:
        action_requests: HITL interrupt 의 승인 요청. 각 항목은 {"name", "args", ...} 다.
        mode: 승인 모드(MODES: manual, auto-edit, yolo).
        ask: 자동으로 정하지 못한 요청을 사람에게 물어 결정을 돌려준다.
        on_decision: 결정마다 (action, decision, auto) 로 불린다. None 이면 부르지 않는다.
        plan: 결정 시점의 계획 상태(auto_decision). 계획 모드를 안 쓰면 None.

    Returns:
        요청과 같은 순서의 결정 목록. 각 결정은 {"type": "approve"} 또는
        {"type": "reject", "message": ...}(reject) 다.

    Raises:
        ValueError: mode 가 MODES 에 없을 때.
    """
    if mode not in MODES:
        raise ValueError(f"승인 모드는 {', '.join(MODES)} 중 하나여야 합니다: {mode!r}")
    decisions = []
    for action in action_requests:
        decision = auto_decision(action, mode, plan)
        auto = decision is not None
        if decision is None:
            decision = ask(action)
        if on_decision is not None:
            on_decision(action, decision, auto)
        decisions.append(decision)
    return decisions


_REASON_PREFIX = " 사용자 의견: "


def reject(reason: str = "", *, base: str = REJECT_MESSAGE) -> dict:
    """거부 결정. reason 이 있으면 모델에 가는 거부 메시지 끝에 사용자 의견으로 붙인다.

    base 는 거부 메시지 본문이다.
    계획 거부는 PLAN_REJECT_MESSAGE(계획을 고쳐 다시 요청하라)를 쓴다.
    """
    message = f"{base}{_REASON_PREFIX}{reason}" if reason else base
    return {"type": "reject", "message": message}


def reject_reason(decision: dict) -> str:
    """reject() 가 메시지에 붙인 사용자 의견. 없으면 빈 문자열."""
    _, _, reason = str(decision.get("message", "")).partition(_REASON_PREFIX)
    return reason
