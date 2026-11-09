"""계획 모드: 쓰기 차단, 리뷰어 서브에이전트, 사람 승인, 범위 재승인, 상태 유지, 사건 기록.

모델은 대본이다(메인 + 리뷰어 두 개). 실제 API 를 부르지 않는다.
"""
from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from test_cli import Script

from code_agent import agent as agent_module
from code_agent import settings as settings_module
from code_agent.agent import build_agent, run_turn, summarization_middleware
from code_agent.approval import (
    APPROVE,
    AUTO_BLOCK,
    OUT_OF_PLAN,
    PLAN_REVIEW_FIRST,
    PlanContext,
    auto_decision,
    in_plan,
    normalize_path,
)
from code_agent.backend import make_backend
from code_agent.cli import App, Sessions
from code_agent.planning import (
    MAX_REVIEWS,
    PLAN_PREFIX,
    SCOPE_KEY,
    block_message,
    next_plan,
    parse_review,
    plan_context,
    plan_files,
    plan_reviewer,
    render_plan,
)
from code_agent.render import Renderer
from code_agent.settings import Settings, load_settings
from code_agent.tool_text import TODO_PROMPT
from code_agent.worklog import WorkLog

PLAN_ARGS = {
    "goal": "calc.py 에 add 함수를 더한다",
    "requirements": ["add(a, b) 는 a + b 를 돌려준다"],
    "in_scope": ["calc.py 의 add"],
    "out_of_scope": ["다른 함수"],
    "done_criteria": ["python -c 로 add(1, 2) == 3 확인"],
    "steps": [{"title": "add 구현", "files": ["/calc.py"], "test": "python -c 실행"},
              {"title": "시험 추가", "files": ["tests/"], "test": "pytest"}],
}
REVIEW_OK = AIMessage(content="VERDICT: ok\n- [low] 문서 없음 :: README 에 한 줄 더한다")
REVIEW_REVISE = AIMessage(content="VERDICT: revise\n"
                                  "- [high] 시험이 없다 :: tests/test_calc.py 를 더한다")
DONE = AIMessage(content="끝")
PLAN_PROMPT = "[y] 승인 / [n] 거부(의견) › "
TOOL_PROMPT = "[y] 승인 / [n] 거부 › "


def write_plan(**changes: Any) -> AIMessage:
    return tool_call("write_plan", {**PLAN_ARGS, **changes})


def review_call() -> AIMessage:
    return tool_call("task", {"description": "요청: calc.py 에 add 추가",
                              "subagent_type": "plan-reviewer"})


def approval_call(note: str = "") -> AIMessage:
    return tool_call("request_plan_approval", {"note": note} if note else {})


def write(path: str, content: str = "x = 1\n") -> AIMessage:
    return tool_call("write_file", {"file_path": path, "content": content})


def tool_messages(model: ScriptedChatModel, name: str | None = None) -> list[ToolMessage]:
    return [m for m in model.calls[-1]
            if isinstance(m, ToolMessage) and (name is None or m.name == name)]


class PlanHarness:
    """메인·리뷰어 대본 모델로 에이전트를 만들고 run_turn 을 돌린다.

    decide 는 모든 요청을 승인한다(헤드리스 결정 함수).
    그래서 approval.auto_decision 을 거치지 않고 PlanModeMiddleware 의 차단만 시험한다.
    """

    def __init__(self, tmp_path: Path, steps: list, reviewer_steps: list | None = None) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.model = ScriptedChatModel(steps=steps)
        self.reviewer = ScriptedChatModel(steps=reviewer_steps or [])
        self.agent = build_agent(Settings(data_dir=tmp_path / "data"), self.workspace,
                                 InMemorySaver(), model=self.model, reviewer_model=self.reviewer)
        self.requests: list[dict] = []
        self.parts: list = []
        self.config = {"configurable": {"thread_id": "t"}}

    def decide(self, action_requests: list[dict]) -> list[dict]:
        self.requests.extend(action_requests)
        return [APPROVE for _ in action_requests]

    def run(self, prompt: str = "go", *, plan_mode: bool = False) -> None:
        inp: dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
        if plan_mode:
            inp["plan_mode"] = True
        run_turn(self.agent, inp, self.config, decide=self.decide, on_part=self.parts.append)

    @property
    def values(self) -> dict:
        return self.agent.get_state(self.config).values

    def events(self, kind: str) -> list[dict]:
        return [data for _ns, mode, data in self.parts
                if mode == "custom" and isinstance(data, dict) and data.get("type") == kind]


def plan_app(tmp_path: Path, steps: list, *answers: str, reviewer_steps: list | None = None,
             mode: str = "manual", sessions: Sessions | None = None, thread_id: str | None = None,
             ) -> tuple[App, ScriptedChatModel, ScriptedChatModel, io.StringIO, Script]:
    """sessions.db(SqliteSaver)에 상태를 남기는 App.

    sessions·thread_id 를 주면 그 세션을 이어 간다(--resume).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data")
    if sessions is None:
        sessions = Sessions(sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False))
    model = ScriptedChatModel(steps=steps)
    reviewer = ScriptedChatModel(steps=reviewer_steps or [])
    agent = build_agent(settings, workspace, SqliteSaver(sessions.conn), model=model,
                        reviewer_model=reviewer)
    out, script = io.StringIO(), Script(*answers)
    app = App(settings, workspace, mode, sessions, thread_id or sessions.create(workspace),
              Console(file=out, width=120, highlight=False), script, agent=agent)
    return app, model, reviewer, out, script


def state_of(app: App) -> dict:
    return app.agent.get_state(app.config).values


# ---- 순수 함수 ----

@pytest.mark.parametrize(("path", "files", "expected"), [
    ("/calc.py", {"/calc.py"}, True),
    ("calc.py", {"/calc.py"}, True),
    ("\\Src\\.\\A.py", {"/src/a.py"}, True),  # 역슬래시, '.', 대소문자
    ("/tests/unit/test_a.py", {"/tests/"}, True),  # 폴더 항목
    ("/tests", {"/tests/"}, True),
    ("/tests2/a.py", {"/tests/"}, False),
    ("/tests/test_calc.py", {"/tests/test_*.py"}, True),  # 글롭
    ("/other.py", {"/calc.py", "/tests/"}, False),
])
def test_in_plan_matches_files_folders_and_globs(path: str, files: set[str],
                                                 expected: bool) -> None:
    assert in_plan(path, {normalize_path(f) for f in files}) is expected


def test_plan_files_union_of_steps_and_scope_changes() -> None:
    plan = next_plan(None, PLAN_ARGS)
    scope = [{"file": "/README.md", "tool": "edit_file", "version": 1},
             {"file": "/old.py", "tool": "edit_file", "version": 0}]  # 옛 버전 것은 빼다
    assert plan_files(plan, scope) == {"/calc.py", "/tests/", "/readme.md"}


@pytest.mark.parametrize(("text", "verdict", "findings"), [
    ("VERDICT: ok\n- 없음", "ok", []),
    ("**VERDICT: revise**\n- [high] 시험 없음 :: 시험을 더한다\n- [Low] 이름 → 고친다",
     "revise", [{"severity": "high", "issue": "시험 없음", "suggestion": "시험을 더한다"},
                {"severity": "low", "issue": "이름", "suggestion": "고친다"}]),
    ("검토했다. 좋아 보인다.", "unknown", []),
])
def test_parse_review(text: str, verdict: str, findings: list) -> None:
    assert parse_review(text) == (verdict, findings)


def test_next_plan_versions_and_series() -> None:
    first = next_plan(None, PLAN_ARGS)
    assert (first["version"], first["status"], first["reviews"]) == (1, "draft", [])
    reviewed = {**first, "status": "reviewed",
                "reviews": [{"version": 1, "verdict": "revise", "findings": []}]}
    second = next_plan(reviewed, PLAN_ARGS)
    assert (second["version"], second["status"], len(second["reviews"])) == (2, "revised", 1)
    # 승인된 계획을 다시 쓰면 새 계획이라 검토를 새로 센다
    third = next_plan({**second, "status": "approved"}, PLAN_ARGS)
    assert (third["version"], third["status"], third["reviews"]) == (3, "draft", [])


def test_block_message_next_steps() -> None:
    assert block_message(False, None) == ""
    assert "write_plan 으로 계획을 써라" in block_message(True, None)
    draft = next_plan(None, PLAN_ARGS)
    assert "plan-reviewer" in block_message(False, draft)
    assert "request_plan_approval" in block_message(False, {**draft, "status": "revised"})
    assert block_message(True, {**draft, "status": "approved"}) == ""


def test_plan_context_from_state() -> None:
    assert plan_context({"messages": []}) is None
    context = plan_context({"plan_mode": True})
    assert context is not None and context.block_message and not context.reviewed
    approved = plan_context({"plan": {**next_plan(None, PLAN_ARGS), "status": "approved"}})
    assert approved == PlanContext(block_message="", reviewed=True, approved=True,
                                   files=frozenset({"/calc.py", "/tests/"}))


@pytest.mark.parametrize("mode", ["manual", "auto-edit", "yolo"])
def test_auto_decision_plan_rules(mode: str) -> None:
    request = {"name": "request_plan_approval", "args": {}}
    # 검토 전 승인 요청은 묻지 않고 거부, 검토 뒤에는 모든 모드에서 묻는다
    review_first = {"type": "reject", "message": AUTO_BLOCK + PLAN_REVIEW_FIRST}
    assert auto_decision(request, mode) == review_first
    assert auto_decision(request, mode, PlanContext(reviewed=False)) == review_first
    assert auto_decision(request, mode, PlanContext(reviewed=True)) is None
    # 승인 전: 쓰기는 차단 문구로 거부, 읽기 전용 명령은 승인
    blocked = PlanContext(block_message="막힘")
    for name, args in [("write_file", {"file_path": "/a.py", "content": ""}),
                       ("execute", {"command": "echo x > a.txt"})]:
        assert auto_decision({"name": name, "args": args}, mode, blocked) == {
            "type": "reject", "message": AUTO_BLOCK + "막힘"}
    assert auto_decision({"name": "execute", "args": {"command": "ls"}}, mode, blocked) == APPROVE
    # 승인 뒤: 계획 밖 파일은 모든 모드에서 묻는다
    approved = PlanContext(reviewed=True, approved=True, files=frozenset({"/calc.py"}))
    outside = {"name": "edit_file", "args": {"file_path": "/other.py"}}
    assert auto_decision(outside, mode, approved) is None
    inside = {"name": "edit_file", "args": {"file_path": "/calc.py"}}
    assert auto_decision(inside, mode, approved) == auto_decision(inside, mode)


def test_render_plan_shows_steps_criteria_and_reviews() -> None:
    plan = {**next_plan(None, PLAN_ARGS), "status": "reviewed",
            "reviews": [{"version": 1, "verdict": "revise",
                         "findings": [{"severity": "high", "issue": "시험 없음",
                                       "suggestion": "시험을 더한다"}]}]}
    text = render_plan(plan)
    for part in ("# 계획 v1 (검토 받음)", PLAN_ARGS["goal"], "1. add 구현", "`/calc.py`",
                 "확인: pytest", "## 완료 조건", "## 검토 (1/2회)",
                 "[high] 시험 없음 → 시험을 더한다"):
        assert part in text, part


def test_prompt_has_static_plan_section(tmp_path: Path) -> None:
    h = PlanHarness(tmp_path, [DONE])
    h.run()
    system = h.model.calls[0][0].text
    assert "# 계획" in system and "plan-reviewer" in system and "request_plan_approval" in system
    assert "완료 조건마다 충족 여부와 확인 방법" in system
    # '작은 작업은 건너뛴다'는 write_todos 에만 해당한다
    assert "applies to write_todos only" in TODO_PROMPT and TODO_PROMPT in system
    assert "## 계획" not in system  # 계획 내용(요청마다 바뀜)은 system 에 넣지 않는다


# ---- 쓰기 차단 ----

def test_gate_blocks_writes_until_approved(tmp_path: Path) -> None:
    """계획 모드에서는 결정 함수가 승인해도 미들웨어가 쓰기와 쓰기 명령을 막는다.

    읽기 전용 명령은 통과시킨다.
    """
    h = PlanHarness(tmp_path, [
        write("/calc.py"),
        tool_call("edit_file", {"file_path": "/calc.py", "old_string": "a", "new_string": "b"}),
        tool_call("execute", {"command": "echo x > made.txt"}),
        tool_call("execute", {"command": "ls"}),
        write_plan(),
        write("/calc.py"),
        DONE])
    (h.workspace / "keep.txt").write_text("k", encoding="utf-8")
    h.run(plan_mode=True)
    assert not (h.workspace / "calc.py").exists() and not (h.workspace / "made.txt").exists()
    messages = tool_messages(h.model)
    blocked = [m for m in messages if m.status == "error"]
    assert [m.name for m in blocked] == ["write_file", "edit_file", "execute", "write_file"]
    assert "계획 모드인데 계획이 아직 없어" in blocked[0].text
    assert "계획(v1)이 아직 승인되지 않아" in blocked[-1].text
    assert "plan-reviewer" in blocked[-1].text
    [ls] = [m for m in messages if m.name == "execute" and m.status != "error"]
    assert "keep.txt" in ls.text


def test_manual_mode_does_not_ask_for_blocked_writes(tmp_path: Path) -> None:
    """승인 전 쓰기는 어차피 막히므로 사람에게 묻지 않고 차단 이유로 거부한다."""
    app, model, _, _, script = plan_app(tmp_path, [write_plan(), write("/calc.py"), DONE],
                                        "/plan calc.py 에 add 추가")
    app.run()
    assert TOOL_PROMPT not in script.prompts
    [rejected] = tool_messages(model, "write_file")
    assert "아직 승인되지 않아" in rejected.text
    assert not (app.workspace / "calc.py").exists()
    humans = [m.text for m in model.calls[0] if m.type == "human"]
    assert humans == [PLAN_PREFIX + "calc.py 에 add 추가"]


# ---- 리뷰어 ----

def test_subagent_cannot_read_env(tmp_path: Path) -> None:
    """리뷰어(isolated 서브에이전트)에도 .env 가드가 있다(task8 필수 조건).

    쓰기 도구와 execute 는 아예 없고, 승인 프롬프트도 생기지 않는다.
    """
    app, model, reviewer, _, script = plan_app(
        tmp_path, [write_plan(), review_call(), DONE], "/plan calc.py 에 add 추가",
        reviewer_steps=[tool_call("read_file", {"file_path": "/.env"}),
                        write("/hacked.py"),
                        tool_call("execute", {"command": "echo hi"}),
                        REVIEW_OK])
    (app.workspace / ".env").write_text("API_KEY=sk-secret-value\n", encoding="utf-8")
    app.run()
    assert set(reviewer.bound_tool_names) == {"ls", "read_file", "glob", "grep"}
    sub = [m for m in reviewer.calls[-1] if isinstance(m, ToolMessage)]
    assert "권한 거부" in sub[0].text
    assert "not a valid tool" in sub[1].text and "not a valid tool" in sub[2].text
    assert not (app.workspace / "hacked.py").exists()
    every = [*reviewer.calls, *model.calls]
    assert all("sk-secret-value" not in str(m.content) for call in every for m in call)
    assert TOOL_PROMPT not in script.prompts and PLAN_PROMPT not in script.prompts


def test_reviewer_spec_lists_its_own_middleware(tmp_path: Path) -> None:
    """isolated 서브에이전트는 메인 middleware 를 물려받지 않는다.

    그래서 spec 에 모두 있어야 한다(task8 0절).
    """
    model = ScriptedChatModel(steps=[])
    backend = make_backend(tmp_path, artifacts_dir=tmp_path / "artifacts", shell="powershell")
    summarization = summarization_middleware(model, backend, 200_000)
    spec = plan_reviewer(model, backend, tmp_path, summarization=summarization)
    names = [m.name for m in spec["middleware"]]
    assert {"FilesystemMiddleware", "DotenvGuardMiddleware", "SummarizationMiddleware",
            "OpenRouterCacheMiddleware", "TruncationRecoveryMiddleware", "PlanBriefMiddleware",
            "ModelCallLimitMiddleware", "ToolCallLimitMiddleware"} <= set(names)
    assert summarization in spec["middleware"]  # 인자 잘림을 끈 task3 요약
    assert (spec["name"], spec["tools"], spec["interrupt_on"]) == ("plan-reviewer", [], {})


def test_invalid_plan_args_return_error_to_model(tmp_path: Path) -> None:
    h = PlanHarness(tmp_path, [tool_call("write_plan", {"goal": "목표만"}), DONE])
    h.run(plan_mode=True)
    [result] = tool_messages(h.model, "write_plan")
    assert result.status == "error" and "steps" in result.text
    assert h.values.get("plan") is None


def test_review_is_recorded_and_reviewer_sees_plan(tmp_path: Path) -> None:
    h = PlanHarness(tmp_path, [write_plan(), review_call(), write_plan(goal="고친 목표"), DONE],
                    reviewer_steps=[REVIEW_REVISE])
    h.run(plan_mode=True)
    system = h.reviewer.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "# 검토할 계획" in system.text and PLAN_ARGS["goal"] in system.text
    assert "VERDICT: ok" in system.text  # 출력 형식 안내
    [result] = tool_messages(h.model, "task")
    assert "VERDICT: revise" in result.text and "검토 1/2회" in result.text
    plan = h.values["plan"]
    assert (plan["version"], plan["status"]) == (2, "revised")
    [review] = plan["reviews"]
    assert (review["version"], review["verdict"]) == (1, "revise")
    assert review["findings"] == [{"severity": "high", "issue": "시험이 없다",
                                   "suggestion": "tests/test_calc.py 를 더한다"}]


def test_review_limit_and_no_plan(tmp_path: Path) -> None:
    """계획 없이 부르거나 MAX_REVIEWS 회를 넘기면 리뷰어를 부르지 않고 오류로 답한다."""
    h = PlanHarness(tmp_path, [review_call(), write_plan(), review_call(), review_call(),
                               review_call(), DONE],
                    reviewer_steps=[REVIEW_REVISE, REVIEW_OK])
    h.run(plan_mode=True)
    results = tool_messages(h.model, "task")
    assert "검토할 계획이 없다" in results[0].text and results[0].status == "error"
    assert results[-1].status == "error" and f"{MAX_REVIEWS}회까지" in results[-1].text
    assert len(h.reviewer.calls) == 2
    assert len(h.values["plan"]["reviews"]) == MAX_REVIEWS


# ---- 승인 ----

def test_approval_without_review_is_auto_rejected(tmp_path: Path) -> None:
    app, model, _, _, script = plan_app(tmp_path, [write_plan(), approval_call(), DONE],
                                        "/plan calc.py 에 add 추가", "y")
    app.run()
    assert PLAN_PROMPT not in script.prompts
    [result] = tool_messages(model, "request_plan_approval")
    assert "plan-reviewer 검토를 먼저 받아라" in result.text and result.status == "error"
    assert state_of(app)["plan"]["status"] == "draft"


def test_approve_creates_todos_and_allows_writes(tmp_path: Path) -> None:
    app, model, _, out, script = plan_app(
        tmp_path,
        [write_plan(), review_call(), approval_call("low 지적은 이번 범위 밖"),
         write("/calc.py", "def add(a, b):\n    return a + b\n"), DONE],
        "/mode auto-edit", "/plan calc.py 에 add 추가", "y", reviewer_steps=[REVIEW_OK])
    app.run()
    assert script.prompts.count(PLAN_PROMPT) == 1 and TOOL_PROMPT not in script.prompts
    text = out.getvalue()
    for part in ("계획 v1 (검토 받음)", "add 구현", "[low] 문서 없음", "low 지적은 이번 범위 밖"):
        assert part in text, part
    values = state_of(app)
    assert values["plan"]["status"] == "approved"
    assert values["todos"] == [{"content": "add 구현", "status": "pending"},
                               {"content": "시험 추가", "status": "pending"}]
    assert (app.workspace / "calc.py").read_text(encoding="utf-8").startswith("def add")
    [approved] = tool_messages(model, "request_plan_approval")
    assert "승인했다" in approved.text and "완료 조건마다" in approved.text
    assert PLAN_ARGS["done_criteria"][0] in approved.text


def test_reject_sends_comment_to_model(tmp_path: Path) -> None:
    app, model, _, _, _ = plan_app(
        tmp_path, [write_plan(), review_call(), approval_call(), DONE],
        "/plan calc.py 에 add 추가", "n", "단계를 더 잘게 나눠라", reviewer_steps=[REVIEW_OK])
    app.run()
    [rejected] = tool_messages(model, "request_plan_approval")
    assert "계획을 승인하지 않았다" in rejected.text and "단계를 더 잘게 나눠라" in rejected.text
    assert state_of(app)["plan"]["status"] == "reviewed"


@pytest.mark.parametrize("mode", ["auto-edit", "yolo"])
def test_out_of_plan_write_asks_and_records_scope_change(tmp_path: Path, mode: str) -> None:
    """승인 뒤 계획 밖 파일은 auto-edit·yolo 에서도 묻고, 승인하면 범위에 더한다.

    계획 안 파일(폴더 항목 포함)과 범위에 더한 파일은 다시 묻지 않는다.
    """
    app, _, _, out, script = plan_app(
        tmp_path,
        [write_plan(), review_call(), approval_call(), write("/other.py"), write("/calc.py"),
         write("/tests/test_calc.py"), write("/other.py", "y = 2\n"), DONE],
        f"/mode {mode}", "/plan calc.py 에 add 추가", "y", "y", reviewer_steps=[REVIEW_OK])
    app.run()
    assert script.prompts.count(PLAN_PROMPT) == 1  # 계획 승인은 yolo 에서도 묻는다
    assert script.prompts.count(TOOL_PROMPT) == 1  # /other.py 첫 쓰기만
    text = out.getvalue()
    assert OUT_OF_PLAN[:10] in text and "계획 범위에 더했다: /other.py" in text
    values = state_of(app)
    scope = values[SCOPE_KEY]
    assert scope == [{"file": "/other.py", "tool": "write_file", "version": 1}]
    assert (app.workspace / "other.py").read_text(encoding="utf-8") == "y = 2\n"
    assert "## 범위 변경" in render_plan(values["plan"], scope)


def test_rejected_out_of_plan_write_is_not_recorded(tmp_path: Path) -> None:
    app, _, _, _, _ = plan_app(
        tmp_path, [write_plan(), review_call(), approval_call(), write("/other.py"), DONE],
        "/mode yolo", "/plan calc.py 에 add 추가", "y", "n", "", reviewer_steps=[REVIEW_OK])
    app.run()
    assert not state_of(app).get(SCOPE_KEY)
    assert not (app.workspace / "other.py").exists()


# ---- 상태 유지와 /plan ----

def test_plan_survives_resume_and_plan_off_clears(tmp_path: Path) -> None:
    app, _, _, _, _ = plan_app(tmp_path, [write_plan(), review_call(), approval_call(), DONE],
                               "/plan calc.py 에 add 추가", "y", reviewer_steps=[REVIEW_OK])
    app.run()
    # 같은 sessions.db 와 thread 로 새 App(--resume)
    resumed, _, _, out, _ = plan_app(tmp_path, [], "/plan", "/plan off", "/plan", "/plan off",
                                     sessions=app.sessions, thread_id=app.thread_id)
    assert state_of(resumed)["plan"]["status"] == "approved"
    assert plan_context(state_of(resumed)) == PlanContext(
        reviewed=True, approved=True, files=frozenset({"/calc.py", "/tests/"}))
    resumed.run()
    text = out.getvalue()
    assert "계획 v1 (승인됨)" in text and "계획 모드와 계획을 지웠다" in text
    assert "계획 없음." in text and "지울 계획이 없다" in text
    values = state_of(resumed)
    assert values.get("plan") is None and not values.get("plan_mode")
    assert plan_context(values) is None


def test_new_plan_request_drops_finished_plan(tmp_path: Path) -> None:
    app, _, _, _, _ = plan_app(tmp_path, [write_plan(), review_call(), approval_call(), DONE],
                               "/plan calc.py 에 add 추가", "y", reviewer_steps=[REVIEW_OK])
    app.run()
    inp = app._turn_input("다음 요청", plan_mode=True)
    assert inp["plan_mode"] is True and inp["plan"] is None
    assert "plan" not in app._turn_input("그냥 질문", plan_mode=False)


# ---- 사건·작업 기록·화면 ----

def test_custom_events_emitted(tmp_path: Path) -> None:
    h = PlanHarness(tmp_path, [write_plan(), review_call(), approval_call(), write("/other.py"),
                               DONE], reviewer_steps=[REVIEW_OK])  # revise 면 반영 전 승인 거부
    h.run(plan_mode=True)
    [written] = h.events("plan_written")
    assert (written["version"], written["files"]) == (1, ["/calc.py", "/tests/"])
    [reviewed] = h.events("plan_reviewed")
    assert reviewed["verdict"] == "ok" and len(reviewed["findings"]) == 1
    [approved] = h.events("plan_approval")
    assert (approved["version"], approved["steps"]) == (1, 2)
    [scope] = h.events("scope_change")
    assert (scope["file"], scope["tool"]) == ("/other.py", "write_file")


def test_worklog_plan_section() -> None:
    log = WorkLog()
    for event in [{"type": "plan_written", "version": 1, "files": ["/calc.py"], "goal": "add 추가",
                   "status": "draft"},
                  {"type": "plan_reviewed", "version": 1, "verdict": "revise",
                   "findings": [{"severity": "high", "issue": "시험 없음", "suggestion": ""}]},
                  {"type": "plan_written", "version": 2, "files": ["/calc.py", "/tests/"],
                   "goal": "add 추가", "status": "revised"},
                  {"type": "plan_approval", "version": 2, "steps": 2},
                  {"type": "scope_change", "version": 2, "file": "/README.md",
                   "tool": "edit_file"},
                  {"type": "prompt_migrated", "reason": "prompt"}]:
        log.on_part(((), "custom", event))
    text = log.render(answer="끝", changes=[], diff="")
    section = text.split("## 계획\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert section == [
        "- 상태: 승인됨 (v2)",
        "- v1 작성: add 추가 · 파일 1개 (/calc.py)",
        "- v1 검토(plan-reviewer): revise (지적 1개: high 1)",
        "- v2 작성: add 추가 · 파일 2개 (/calc.py, /tests/)",
        "- v2 사용자 승인: 단계 2개를 할 일로 넣음",
        "- 범위 변경(사용자 승인): /README.md (edit_file)"]
    assert "## 계획" not in WorkLog().render(answer="끝", changes=[], diff="")


def test_render_labels_subagent_and_hides_its_tokens() -> None:
    out = io.StringIO()
    renderer = Renderer(Console(file=out, width=120, highlight=False), status=False)
    call = review_call().tool_calls[0]
    renderer.on_part(((), "updates", {"model": {"messages": [AIMessage(content="",
                                                                       tool_calls=[call])]}}))
    ns = ("tools:6afc324d-1b83",)
    renderer.on_part((ns, "messages", (AIMessage(content="VERDICT: 비밀 아님"),
                                       {"langgraph_node": "model"})))
    assert renderer.status.label == "plan-reviewer 작업 중"  # 메인 '생각 중 N자'와 섞이지 않는다
    sub_call = tool_call("read_file", {"file_path": "/calc.py"})
    renderer.on_part((ns, "updates", {"model": {"messages": [sub_call]}}))
    text = out.getvalue()
    assert "task(plan-reviewer) 요청: calc.py 에 add 추가" in text
    assert "[plan-reviewer] ● read_file /calc.py" in text
    assert "VERDICT" not in text  # 리뷰어 토큰은 메인 답변에 섞이지 않는다
    assert renderer.last_text == ""


# ---- 설정 ----

def test_reviewer_model_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_module, "REPO_ROOT", tmp_path)  # 저장소 .env 를 읽지 않는다
    monkeypatch.setenv("CODE_AGENT_REVIEWER_MODEL", "openrouter:anthropic/claude-opus-5.5")
    assert load_settings(tmp_path).reviewer_model == "openrouter:anthropic/claude-opus-5.5"
    monkeypatch.delenv("CODE_AGENT_REVIEWER_MODEL")
    assert load_settings(tmp_path).reviewer_model == ""


def test_reviewer_model_built_from_settings(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """비어 있으면 메인 모델을 같이 쓰고, 이름이 있으면 그 이름으로 make_model 을 부른다."""
    built: list[str] = []

    def fake_make_model(settings: Settings) -> ScriptedChatModel:
        built.append(settings.model)
        return ScriptedChatModel(steps=[])

    monkeypatch.setattr(agent_module, "make_model", fake_make_model)
    (tmp_path / "ws").mkdir()
    main = ScriptedChatModel(steps=[])
    build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", None, model=main)
    assert built == []
    build_agent(Settings(data_dir=tmp_path / "data", reviewer_model="openrouter:x/rev"),
                tmp_path / "ws", None, model=main)
    assert built == ["openrouter:x/rev"]
