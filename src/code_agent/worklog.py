"""한 턴의 작업 기록. LLM-as-a-Judge 가 turn observation 하나만 보고도 판단할 수 있게 근거를 텍스트로 모은다.

Langfuse 의 observation 평가기는 같은 트레이스의 다른 observation(도구 실행 등)을 읽지 못한다. 그래서 실행한 도구와
결과 요약, 승인 결정, 바뀐 파일과 diff 를 최종 답변과 함께 turn 출력에 남긴다. 길이는 평가 비용을 위해 자른다.

평가기는 시스템 프롬프트와 이전 턴도 보지 못한다. 그래서 에이전트가 턴 전에 이미 알고 있던 것(작업 폴더, 같은 세션의
이전 턴 요약)을 맥락 절로 함께 남긴다. 없으면 맞는 말도 '근거 없는 주장'으로 감점된다.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from code_agent.approval import reject_reason
from code_agent.guards import is_env_path
from code_agent.render import describe_call

MAX_STEPS = 80
MAX_CONTEXT_TURNS = 10
MAX_CONTEXT_REQUEST = 80
MAX_CONTEXT_FILES = 10
FILE_TOOLS = frozenset({"write_file", "edit_file", "delete"})
MAX_LINE = 240
MAX_DIFF_CHARS = 30_000
MAX_FILES = 5_000  # 이보다 큰 작업 폴더는 스냅샷을 건너뛴다
MAX_FILE_BYTES = 1_000_000
SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
                       ".mypy_cache", "dist", "build"})
_EXIT_CODE = re.compile(r"exit code (\d+)|Exit code: (\d+)", re.IGNORECASE)
_READ_RANGE = re.compile(r"^@@ lines (\d+)-(\d+) of (\d+) @@")  # deepagents read_file 결과의 첫 줄

Snapshot = dict[str, bytes]


def snapshot(root: Path) -> Snapshot | None:
    """작업 폴더의 파일 내용. .env 는 비밀값이라 넣지 않는다. 파일이 너무 많으면 None."""
    files: Snapshot = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if is_env_path(name):
                continue
            path = Path(current, name)
            if len(files) >= MAX_FILES:
                return None
            try:
                if path.stat().st_size <= MAX_FILE_BYTES:
                    files[path.relative_to(root).as_posix()] = path.read_bytes()
            except OSError:
                continue
    return files


def diff_snapshots(before: Snapshot, after: Snapshot) -> tuple[list[str], str]:
    """(바뀐 파일 목록, unified diff). 목록은 'A 파일', 'M 파일 (+3 -1)', 'D 파일' 형식이다."""
    changes, chunks = [], []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        old_text, new_text = _text(old), _text(new)
        if old_text is None or new_text is None:
            changes.append(f"{_status(old, new)} {name} (바이너리)")
            continue
        lines = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                          f"a/{name}", f"b/{name}", lineterm=""))
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        changes.append(f"{_status(old, new)} {name} (+{added} -{removed})")
        chunks.append("\n".join(lines))
    diff = "\n".join(chunks)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n... (diff {len(diff):,}자 중 {MAX_DIFF_CHARS:,}자까지)"
    return changes, diff


def _text(data: bytes | None) -> str | None:
    if data is None:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _status(old: bytes | None, new: bytes | None) -> str:
    return "A" if old is None else "D" if new is None else "M"


def summarize_result(message: ToolMessage) -> str:
    """도구 결과 한 줄 요약. execute 는 종료 코드와 마지막 출력 줄, read_file 은 읽은 범위만 남긴다."""
    lines = [line.strip() for line in message.text.strip().splitlines() if line.strip()]
    if message.status == "error":
        return one_line("오류: " + (lines[0] if lines else ""))
    if message.name == "execute":
        match = _EXIT_CODE.search(message.text)
        code = next((g for g in match.groups() if g), "?") if match else "?"
        body = [line for line in lines if not _EXIT_CODE.search(line)]
        return _clip(f"exit {code}" + (f" · {body[-1]}" if body else ""))
    if message.name == "read_file":
        match = _READ_RANGE.match(lines[0]) if lines else None
        return f"{match[1]}-{match[2]}줄 읽음 (전체 {match[3]}줄)" if match else f"{len(lines)}줄 읽음"
    return _clip(lines[0] if lines else "(출력 없음)")


def _clip(text: str) -> str:
    return text if len(text) <= MAX_LINE else text[: MAX_LINE - 1] + "…"


def one_line(text: str) -> str:
    """여러 줄 명령(node -e '...' 등)도 목록 한 줄에 들어가게 공백을 합치고 자른다."""
    return _clip(" ".join(text.split()))


def session_context(previous: list[BaseMessage], workspace: Path | None) -> str:
    """턴 전에 에이전트가 알고 있던 것: 시스템 프롬프트의 작업 폴더와 같은 세션의 이전 턴 요약(요청, 도구 수, 바꾼 파일)."""
    turns: list[list[BaseMessage]] = []
    for message in previous:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    lines = [f"- 작업 폴더(시스템 프롬프트로 알고 있다): {workspace}"] if workspace is not None else []
    if turns:
        start = max(0, len(turns) - MAX_CONTEXT_TURNS)
        lines.append(f"- 같은 세션의 이전 턴 {len(turns)}개" + (f" (최근 {MAX_CONTEXT_TURNS}개만)" if start else "") + ":")
        for number, turn in enumerate(turns[start:], start + 1):
            calls = [call for m in turn if isinstance(m, AIMessage) for call in m.tool_calls]
            files = sorted({str(c["args"].get("file_path", "")) for c in calls if c["name"] in FILE_TOOLS})
            shown = ", ".join(files[:MAX_CONTEXT_FILES]) + (f" 외 {len(files) - MAX_CONTEXT_FILES}개"
                                                           if len(files) > MAX_CONTEXT_FILES else "")
            # 요청 원문이 길어도 바꾼 파일 목록이 잘리지 않게 요청만 따로 줄인다
            request = " ".join(turn[0].text.split())
            request = request if len(request) <= MAX_CONTEXT_REQUEST else request[: MAX_CONTEXT_REQUEST - 1] + "…"
            lines.append(f"  {number}. 요청: {request} · 도구 {len(calls)}개" + (f" · 바꾼 파일: {shown}" if files else ""))
    return "\n".join(lines)


class WorkLog:
    """스트림 조각과 승인 결정을 받아 턴의 작업 기록을 만든다."""

    def __init__(self) -> None:
        self.steps: list[list[str]] = []  # [호출 요약, 결과 요약]
        self.approvals: list[str] = []
        self._by_call_id: dict[str, int] = {}

    def on_part(self, part: tuple[tuple[str, ...], str, Any]) -> None:
        ns, mode, data = part
        if mode != "updates":
            return
        prefix = "[sub] " if ns else ""
        for node, update in data.items():
            messages = update.get("messages") if isinstance(update, dict) else None
            for message in messages if isinstance(messages, list) else []:
                if node == "model" and isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        self._by_call_id[call["id"]] = len(self.steps)
                        self.steps.append([one_line(prefix + describe_call(call)), "(결과 없음)"])
                elif isinstance(message, ToolMessage):
                    index = self._by_call_id.get(message.tool_call_id)
                    if index is not None:
                        self.steps[index][1] = summarize_result(message)

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        verdict = "승인" if decision.get("type") == "approve" else "거부"
        reason = reject_reason(decision)
        self.approvals.append(one_line(f"{describe_call(action)} → {verdict}({'자동' if auto else '사용자'})"
                                       + (f": {reason}" if reason else "")))

    def render(self, *, answer: str, changes: list[str] | None, diff: str, context: str = "") -> str:
        shown = self.steps[:MAX_STEPS]
        steps = [f"{i}. {call} → {result}" for i, (call, result) in enumerate(shown, 1)]
        if len(self.steps) > MAX_STEPS:
            steps.append(f"... (도구 호출 {len(self.steps)}개 중 {MAX_STEPS}개까지)")
        files = ("(작업 폴더가 커서 확인하지 않았다)" if changes is None
                 else "\n".join(f"- {c}" for c in changes) or "(없음)")
        sections = [
            f"## 최종 답변\n{answer.strip() or '(없음)'}",
            f"## 실행한 도구 ({len(self.steps)}개)\n" + ("\n".join(steps) or "(없음)"),
            "## 승인 결정\n" + ("\n".join(f"- {a}" for a in self.approvals) or "(없음)"),
            f"## 바뀐 파일\n{files}",
            f"## diff\n```diff\n{diff}\n```" if diff else "## diff\n(없음)",
        ]
        if context:
            sections.append(f"## 맥락 (이 턴 전에 에이전트가 알고 있던 것)\n{context}")
        return "\n\n".join(sections)
