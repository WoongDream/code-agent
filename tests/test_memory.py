"""메모리(AGENTS.md)와 스킬: system 주입과 안정성, 사용자 공통 라우트 격리, 승인, 신뢰 확인,
/remember·/memory, 사건과 작업 기록. 모델은 대본이고 데이터 폴더는 임시 폴더다.
"""
from __future__ import annotations

import io
import logging
import os
import sqlite3
from pathlib import Path

import pytest
from fakes import ScriptedChatModel, tool_call, tool_calls
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from test_cli import Script
from test_planning import REVIEW_OK, approval_call, plan_app, review_call, write, write_plan

from code_agent.agent import build_agent, run_turn
from code_agent.approval import (
    APPROVE,
    AUTO_BLOCK,
    OUT_OF_PLAN,
    USER_MEMORY_REASON,
    PlanContext,
    ask_reason,
    auto_decision,
    is_memory_file,
    is_user_memory,
    plan_reason,
)
from code_agent.backend import make_backend
from code_agent.cli import NOT_REMEMBERED, App, Sessions
from code_agent.localtrace import TRACE_FILE, LocalTracer, read_jsonl, run_dir
from code_agent.memory import (
    LOG_FILE,
    MEMORY_SOURCES,
    REMEMBER_DONE,
    USER_MEMORY,
    WARN_CHARS,
    AgentMemoryMiddleware,
    MemoryTrust,
    content_sha256,
    library_logs,
    memory_report,
    remember_prompt,
    skills_sha256,
    trust_key,
)
from code_agent.models import make_model
from code_agent.resume_compat import MemoryPromptStore, fingerprint
from code_agent.settings import Settings
from code_agent.worklog import WorkLog

DONE = AIMessage(content="끝")
TOOL_PROMPT = "[y] 승인 / [n] 거부 › "
TRUST_PROMPT_HEAD = "이 프로젝트 AGENTS.md("


def system_of(model: ScriptedChatModel, call: int = 0) -> str:
    return model.calls[call][0].text


def humans(model: ScriptedChatModel, call: int = -1) -> list[str]:
    return [m.text for m in model.calls[call] if isinstance(m, HumanMessage)]


def write_user_memory(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data" / "memory" / "AGENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_project_memory(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ws" / "AGENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def trusted(tmp_path: Path) -> MemoryTrust:
    """지금 디스크의 프로젝트 AGENTS.md 를 신뢰한다."""
    return MemoryTrust(content_sha256((tmp_path / "ws" / "AGENTS.md").read_bytes()))


def memory_app(tmp_path: Path, steps: list, *answers: str, mode: str = "manual",
               interactive: bool = False, tracer=None, sessions: Sessions | None = None,
               thread_id: str | None = None,
               ) -> tuple[App, ScriptedChatModel, io.StringIO, Script]:
    """sessions.db 에 상태를 남기고 agent 와 신뢰(MemoryTrust)를 나눠 쓰는 App(main 과 같다).

    sessions·thread_id 를 주면 그 세션을 새 agent 로 이어 간다(--resume).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data")
    if sessions is None:
        sessions = Sessions(sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False))
    trust = MemoryTrust()
    model = ScriptedChatModel(steps=steps)
    agent = build_agent(settings, workspace, SqliteSaver(sessions.conn), model=model,
                        memory_trust=trust)
    out, script = io.StringIO(), Script(*answers)
    app = App(settings, workspace, mode, sessions, thread_id or sessions.create(workspace),
              Console(file=out, width=200, highlight=False), script, agent=agent, tracer=tracer,
              memory_trust=trust, interactive=interactive)
    return app, model, out, script


class Agent:
    """대본 모델 agent 로 thread 를 돌린다. 같은 체크포인터로 다시 만들면 --resume 과 같다."""

    def __init__(self, tmp_path: Path, trust: MemoryTrust | None = None) -> None:
        self.tmp_path = tmp_path
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(exist_ok=True)
        self.saver = InMemorySaver()
        self.store = MemoryPromptStore()
        self.trust = trust if trust is not None else MemoryTrust()

    def build(self, steps: list):
        model = ScriptedChatModel(steps=steps)
        agent = build_agent(Settings(data_dir=self.tmp_path / "data"), self.workspace,
                            self.saver, model=model, prompt_store=self.store,
                            memory_trust=self.trust)
        return agent, model

    @staticmethod
    def turn(agent, text: str, thread: str = "t", decide=None) -> list:
        parts: list = []
        run_turn(agent, {"messages": [{"role": "user", "content": text}]},
                 {"configurable": {"thread_id": thread}},
                 decide=decide or (lambda requests: [APPROVE for _ in requests]),
                 on_part=parts.append)
        return parts


def custom(parts: list, kind: str) -> list[dict]:
    return [d for _, mode, d in parts if mode == "custom" and d.get("type") == kind]


# ---- system 주입과 안정성 ----

def test_memory_loaded_into_system(tmp_path: Path) -> None:
    """첫 요청 system 끝에 사용자 공통 → 프로젝트 순서로 두 파일이 들어간다."""
    write_user_memory(tmp_path, "- USER-RULE: 답은 짧게")
    write_project_memory(tmp_path, "- PROJECT-RULE: uv run pytest")
    a = Agent(tmp_path, trusted(tmp_path))
    agent, model = a.build([DONE])
    parts = a.turn(agent, "하이")
    system = system_of(model)
    assert ("<agent_memory>\n/.code-agent-user/AGENTS.md\n\n- USER-RULE: 답은 짧게\n\n"
            "/AGENTS.md\n\n- PROJECT-RULE: uv run pytest\n\n</agent_memory>") in system
    # 메모리는 우리 프롬프트와 스킬 절 뒤, system 끝에 붙는다
    assert system.index("# 실행 환경") < system.index("## Skills System") < system.index(
        "<agent_memory>\n")
    # 메모리 위치를 알려 주는 고정 절(prompts.py)
    assert "# 메모리" in system and "/.code-agent-user/AGENTS.md`(모든 프로젝트 공통)" in system
    [event] = custom(parts, "memory_loaded")
    assert event["files"] == [{"path": "/.code-agent-user/AGENTS.md", "chars": 18},
                              {"path": "/AGENTS.md", "chars": 29}]
    assert event["total_chars"] == 47 and "untrusted" not in event


def test_memory_stable_after_edit(tmp_path: Path) -> None:
    """턴 사이에 두 파일을 고쳐도 두 번째 요청의 system 은 바이트 단위로 같다."""
    user = write_user_memory(tmp_path, "USER-V1")
    project = write_project_memory(tmp_path, "PROJECT-V1")
    a = Agent(tmp_path, trusted(tmp_path))
    agent, model = a.build([DONE, DONE])
    a.turn(agent, "하나")
    user.write_text("USER-V2", encoding="utf-8")
    project.write_text("PROJECT-V2", encoding="utf-8")
    parts = a.turn(agent, "둘")
    assert system_of(model, 1) == system_of(model, 0)
    assert "USER-V1" in system_of(model, 1) and "V2" not in system_of(model, 1)
    assert custom(parts, "memory_loaded") == []  # thread 마다 한 번만 읽는다


def test_memory_reloaded_on_new(tmp_path: Path) -> None:
    """고친 사용자 메모리는 /new 로 시작한 thread 에서 반영된다."""
    user = write_user_memory(tmp_path, "USER-V1")
    app, model, _, _ = memory_app(tmp_path, [DONE, DONE], "하나", "/new", "둘", "/exit")
    first = app.thread_id

    def edit_after_first(prompt: str) -> str:
        if prompt == "› " and model.calls:
            user.write_text("USER-V2", encoding="utf-8")
        return script(prompt)

    script, app.read_line = app.read_line, edit_after_first
    app.run()
    assert app.thread_id != first
    assert "USER-V1" in system_of(model, 0) and "USER-V2" in system_of(model, 1)


def test_memory_stable_on_resume(tmp_path: Path) -> None:
    """같은 thread 를 새 agent(새 App)로 이어도 system 이 같다. 그사이 고친 파일은 읽지 않는다."""
    user = write_user_memory(tmp_path, "USER-V1")
    app, model, _, _ = memory_app(tmp_path, [DONE], "하나", "/exit")
    app.run()
    user.write_text("USER-V2", encoding="utf-8")
    resumed, model2, _, _ = memory_app(tmp_path, [DONE], "둘", "/exit", sessions=app.sessions,
                                       thread_id=app.thread_id)
    resumed.run()
    assert system_of(model2) == system_of(model)
    assert humans(model2, 0) == ["하나", "둘"]


def test_fingerprint_covers_memory_and_skills(tmp_path: Path) -> None:
    """지문(resume_compat)은 안쪽 middleware 가 붙이는 메모리·스킬 절도 본다.

    그 절이 생기기 전의 옛 thread 를 이어 가면 system 이 바뀌므로 이전해야 하기 때문이다.
    """
    write_user_memory(tmp_path, "USER-A")
    a = Agent(tmp_path)
    agent, _ = a.build([DONE])
    a.turn(agent, "하이", thread="a")
    write_user_memory(tmp_path, "USER-B")
    agent, _ = a.build([DONE])
    a.turn(agent, "하이", thread="b")
    first, second = a.store.get("a"), a.store.get("b")
    assert first is not None and second is not None
    assert first.fingerprint.system_sha != second.fingerprint.system_sha
    assert first.fingerprint.tools_sha == second.fingerprint.tools_sha


def test_anthropic_cache_tag_keeps_text_and_fingerprint(tmp_path: Path) -> None:
    """Anthropic 직결이면 메모리 middleware 가 system 마지막 블록에 cache_control 을 붙인다
    (deepagents 의 둘째 캐시 지점). 글은 그대로라 지문(resume_compat)이 같고, 같은 state 면 매번
    같은 블록이다. 다른 모델에는 붙이지 않는다.
    """
    (tmp_path / "ws").mkdir()
    backend = make_backend(tmp_path / "ws", artifacts_dir=tmp_path / "a",
                           memory_dir=tmp_path / "m")
    memory = AgentMemoryMiddleware(backend=backend, sources=MEMORY_SOURCES, trust=MemoryTrust())
    anthropic = make_model(Settings(model="anthropic:claude-opus-5-5",
                                    anthropic_api_key="DUMMY"))
    state = {"messages": [], "memory_contents": {USER_MEMORY: "USER-RULE"}}

    def probe(model) -> ModelRequest:
        return memory.modify_request(ModelRequest(model=model, messages=[], tools=[],
                                                  system_message=SystemMessage("BASE"),
                                                  state=state))  # type: ignore[arg-type]

    tagged, again, plain = probe(anthropic), probe(anthropic), probe(ScriptedChatModel(steps=[]))
    assert tagged.system_message is not None and plain.system_message is not None
    assert tagged.system_message.content_blocks[-1].get("cache_control") == {"type": "ephemeral"}
    assert again.system_message is not None
    assert again.system_message.content == tagged.system_message.content
    assert "cache_control" not in str(plain.system_message.content)
    assert plain.system_message.text == tagged.system_message.text
    assert fingerprint(tagged).system_sha == fingerprint(plain).system_sha


def test_openrouter_system_byte_stable_with_memory(tmp_path: Path) -> None:
    """OpenRouter 로 실제로 보내는 system 이 메모리를 고친 뒤 둘째 턴에도 같고, 캐시는 요청
    최상위 cache_control(caching.py) 하나뿐이다(system 블록에 붙이지 않는다).
    """
    pytest.importorskip("langchain_openrouter")
    from test_caching import FakeOpenRouter

    class Recording(FakeOpenRouter):
        def __init__(self, model: str) -> None:
            super().__init__(model)
            self.messages: list[list[dict]] = []

        def send(self, messages: list[dict], **params):
            self.messages.append(messages)
            return super().send(messages, **params)

    user = write_user_memory(tmp_path, "USER-V1")
    (tmp_path / "ws").mkdir()
    settings = Settings(model="openrouter:anthropic/claude-opus-5.5", openrouter_api_key="k",
                        data_dir=tmp_path / "data")
    model = make_model(settings)
    fake = Recording("anthropic/claude-opus-5.5")
    model.client = fake  # type: ignore[attr-defined]
    agent = build_agent(settings, tmp_path / "ws", InMemorySaver(), model=model)
    for text in ("하나", "둘"):
        Agent.turn(agent, text)
        user.write_text("USER-V2", encoding="utf-8")
    first, second = (sent[0] for sent in fake.messages)
    assert first["role"] == "system" and first == second
    assert "USER-V1" in str(first["content"]) and "cache_control" not in str(first)
    assert all(params["cache_control"] == {"type": "ephemeral"} for params in fake.sent)


def test_memory_errors_do_not_fail_turn(tmp_path: Path) -> None:
    """읽지 못하는 메모리(폴더인 AGENTS.md)와 UTF-8 이 아닌 파일·BOM 이 턴을 멈추지 않는다."""
    (tmp_path / "data" / "memory" / "AGENTS.md").mkdir(parents=True)
    write_project_memory(tmp_path, "")
    (tmp_path / "ws" / "AGENTS.md").write_bytes(b"\xef\xbb\xbfPROJECT \xff\xfe")
    a = Agent(tmp_path, trusted(tmp_path))
    agent, model = a.build([DONE])
    [event] = custom(a.turn(agent, "하이"), "memory_loaded")
    assert event["errors"] == [{"path": "/.code-agent-user/AGENTS.md", "error": "is_directory"}]
    assert "/AGENTS.md\n\nPROJECT ��" in system_of(model)  # BOM 은 떼고 바꿔 읽는다


# ---- 스킬 ----

SKILL = """---
name: {name}
description: {description}
---

# {name}
본문
"""


def test_skills_listed_and_missing_folders_quiet(tmp_path: Path) -> None:
    """두 스킬 폴더의 SKILL.md 가 system 에 보이고 같은 이름이면 프로젝트가 이긴다.

    없는 폴더는 system 에 경고 절을 만들지 않는다.
    """
    user = tmp_path / "data" / "memory" / "skills"
    for name, description in (("shared", "USER-SHARED"), ("only-user", "USER-ONLY")):
        (user / name).mkdir(parents=True)
        (user / name / "SKILL.md").write_text(SKILL.format(name=name, description=description),
                                              encoding="utf-8")
    project = tmp_path / "ws" / ".agents" / "skills" / "shared"
    project.mkdir(parents=True)
    (project / "SKILL.md").write_text(SKILL.format(name="shared", description="PROJECT-SHARED"),
                                      encoding="utf-8")
    a = Agent(tmp_path, MemoryTrust(skills_sha256=skills_sha256(tmp_path / "ws")))
    agent, model = a.build([DONE])
    a.turn(agent, "하이")
    system = system_of(model)
    assert "**only-user**: USER-ONLY" in system and "**shared**: PROJECT-SHARED" in system
    assert "USER-SHARED" not in system
    assert "**User Skills**: `/.code-agent-user/skills/`" in system
    assert "**Project Skills**: `/.agents/skills/` (higher priority)" in system

    empty = tmp_path / "empty"
    empty.mkdir()
    b = Agent(empty)
    agent, model = b.build([DONE])
    b.turn(agent, "하이")
    assert "skill_load_warnings" not in system_of(model)
    assert "No skills available yet" in system_of(model)


def test_library_logs_go_to_file_not_stderr(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    """deepagents 경고는 상태 줄을 깨지 않게 파일로 가고, 끝나면 원래대로 되돌린다."""
    logger = logging.getLogger("deepagents.middleware.skills")
    parent = logging.getLogger("deepagents")
    before = (list(parent.handlers), parent.propagate)
    with library_logs(tmp_path) as path:
        logger.warning("Cannot load skills from '/x/': path_not_found")
    assert path == tmp_path / "logs" / LOG_FILE
    assert "path_not_found" in path.read_text(encoding="utf-8")
    assert "path_not_found" not in capsys.readouterr().err
    assert (list(parent.handlers), parent.propagate) == before


# ---- 사용자 공통 메모리 라우트 ----

def test_user_memory_route_is_isolated(tmp_path: Path) -> None:
    """/.code-agent-user/ 는 <data_dir>/memory 만 보인다.

    데이터 폴더의 sessions.db·.env 와 그 폴더 안의 .env 에 닿지 못한다.
    """
    data = tmp_path / "data"
    data.mkdir()
    (data / ".env").write_text("OPENROUTER_API_KEY=DUMMY-DATA-SECRET\n", encoding="utf-8")
    (data / "sessions.db").write_text("DUMMY-SESSIONS-DB", encoding="utf-8")
    write_user_memory(tmp_path, "USER-RULE")
    (data / "memory" / ".env").write_text("TOKEN=DUMMY-MEMORY-SECRET\n", encoding="utf-8")
    paths = ["/.code-agent-user/../sessions.db", "/.code-agent-user/../.env",
             "/.code-agent-user/.env", "/.code-agent-user/AGENTS.md"]
    a = Agent(tmp_path)
    agent, model = a.build([tool_calls(*(("read_file", {"file_path": p}) for p in paths)),
                            tool_call("ls", {"path": "/.code-agent-user/"}),
                            tool_call("grep", {"pattern": "DUMMY", "path": "/.code-agent-user/"}),
                            DONE])
    a.turn(agent, "읽어")
    results = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    sessions_db, data_env, memory_env, memory, listing, grep = results
    assert all(m.status == "error" or m.text.startswith("Error")
               for m in (sessions_db, data_env, memory_env))
    assert "USER-RULE" in memory.text
    assert "/.code-agent-user/AGENTS.md" in listing.text and "sessions.db" not in listing.text
    every = "\n".join(str(m.content) for call in model.calls for m in call)
    assert "DUMMY-DATA-SECRET" not in every and "DUMMY-SESSIONS-DB" not in every
    assert "DUMMY-MEMORY-SECRET" not in every and "DUMMY-MEMORY-SECRET" not in grep.text


def test_build_agent_requires_user_memory_route(tmp_path: Path) -> None:
    """라우트가 없는 백엔드면 /.code-agent-user/ 가 작업 폴더로 가 저장소 파일이 사용자 메모리로
    읽히므로 거부한다.
    """
    (tmp_path / "ws").mkdir()
    backend = make_backend(tmp_path / "ws", artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(ValueError, match="사용자 공통 메모리 라우트"):
        build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", None,
                    model=ScriptedChatModel(steps=[]), backend=backend)


def test_reviewer_cannot_read_env_through_memory_route(tmp_path: Path) -> None:
    """plan-reviewer 서브에이전트도 사용자 메모리 라우트로 .env 를 읽지 못한다."""
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    (data / ".env").write_text("API_KEY=DUMMY-DATA-SECRET\n", encoding="utf-8")
    (data / "memory" / ".env").write_text("API_KEY=DUMMY-MEMORY-SECRET\n", encoding="utf-8")
    app, model, reviewer, _, _ = plan_app(
        tmp_path, [write_plan(), review_call(), DONE], "/plan calc.py 에 add 추가",
        reviewer_steps=[tool_call("read_file", {"file_path": "/.code-agent-user/.env"}),
                        tool_call("read_file", {"file_path": "/.code-agent-user/../.env"}),
                        tool_call("read_file", {"file_path": "/.env"}),
                        REVIEW_OK])
    (app.workspace / ".env").write_text("API_KEY=DUMMY-WS-SECRET\n", encoding="utf-8")
    app.run()
    sub = [m for m in reviewer.calls[-1] if isinstance(m, ToolMessage)]
    assert len(sub) == 3 and all(m.status == "error" or m.text.startswith("Error") for m in sub)
    every = "\n".join(str(m.content) for call in [*reviewer.calls, *model.calls] for m in call)
    assert "DUMMY" not in every


# ---- 승인 ----

USER_PATHS = ["/.code-agent-user/AGENTS.md", "\\.code-agent-user\\AGENTS.md",
              "/./.CODE-AGENT-USER//skills/x/SKILL.md", ".code-agent-user/AGENTS.md",
              "/.code-agent-user"]


@pytest.mark.parametrize("path", USER_PATHS)
@pytest.mark.parametrize("mode", ["manual", "auto-edit", "yolo"])
@pytest.mark.parametrize("tool", ["write_file", "edit_file", "delete"])
def test_user_memory_write_always_asks(path: str, mode: str, tool: str) -> None:
    """사용자 공통 메모리 쓰기는 yolo 를 포함한 모든 모드에서 묻고 그 이유를 보인다."""
    action = {"name": tool, "args": {"file_path": path, "content": "x"}}
    assert is_user_memory(path)
    assert auto_decision(action, mode) is None
    assert ask_reason(action) == USER_MEMORY_REASON


@pytest.mark.parametrize(("mode", "expected"), [("manual", None), ("auto-edit", APPROVE),
                                                ("yolo", APPROVE)])
def test_project_memory_follows_mode(mode: str, expected: dict | None) -> None:
    """프로젝트 /AGENTS.md 는 일반 파일 규칙이다. 비슷한 이름은 메모리가 아니다."""
    action = {"name": "edit_file", "args": {"file_path": "/AGENTS.md"}}
    assert auto_decision(action, mode) == expected and ask_reason(action) == ""
    for other in ("/sub/AGENTS.md", "/.code-agent-userx/a.md", "/AGENTS.md.bak"):
        assert not is_memory_file(other)


def test_memory_files_skip_plan_gate_and_scope() -> None:
    """메모리 파일은 계획 승인 전 차단과 '계획에 없는 파일' 재승인에서 빠진다."""
    blocked = PlanContext(block_message="계획을 먼저 승인받아라")
    project = {"name": "write_file", "args": {"file_path": "/AGENTS.md", "content": "x"}}
    user = {"name": "edit_file", "args": {"file_path": "/.code-agent-user/AGENTS.md"}}
    code = {"name": "write_file", "args": {"file_path": "/src/a.py", "content": "x"}}
    assert auto_decision(project, "auto-edit", blocked) == APPROVE
    assert auto_decision(user, "yolo", blocked) is None  # 계획과 관계없이 사람에게 묻는다
    assert auto_decision(code, "auto-edit", blocked) == {
        "type": "reject", "message": AUTO_BLOCK + "계획을 먼저 승인받아라"}
    approved = PlanContext(reviewed=True, approved=True, files=frozenset({"/src/a.py"}))
    assert plan_reason(project, approved) == "" and plan_reason(user, approved) == ""
    assert auto_decision(project, "auto-edit", approved) == APPROVE
    assert ask_reason(user, approved) == USER_MEMORY_REASON
    assert plan_reason({"name": "write_file", "args": {"file_path": "/b.py"}},
                       approved) == OUT_OF_PLAN


def test_memory_write_allowed_in_plan_mode(tmp_path: Path) -> None:
    """계획 모드라 코드 쓰기가 막혀도 프로젝트 AGENTS.md 쓰기는 된다(미들웨어 차단과 같은 기준)."""
    app, model, _, _, _ = plan_app(
        tmp_path, [tool_calls(("write_file", {"file_path": "/AGENTS.md", "content": "- 규칙\n"}),
                              ("write_file", {"file_path": "/a.py", "content": "x = 1\n"})),
                   DONE],
        "/mode auto-edit", "/plan a.py 를 고쳐")
    app.run()
    assert (app.workspace / "AGENTS.md").read_text(encoding="utf-8") == "- 규칙\n"
    assert not (app.workspace / "a.py").exists()
    # HITL 은 거부한 호출의 결과를 먼저 넣는다
    results = {m.status: m for m in model.calls[-1] if isinstance(m, ToolMessage)}
    assert set(results) == {"success", "error"} and "계획" in results["error"].text


def test_memory_write_is_not_scope_change(tmp_path: Path) -> None:
    """승인된 계획이 있어도 메모리 쓰기는 다시 묻지 않고(프로젝트) 범위 변경으로 남지 않는다.

    사용자 공통 메모리는 계획 이유가 아니라 메모리 이유로 묻는다.
    """
    app, _, _, out, script = plan_app(
        tmp_path,
        [write_plan(), review_call(), approval_call(), write("/AGENTS.md", "- 규칙\n"),
         write("/.code-agent-user/AGENTS.md", "- 공통\n"), DONE],
        "/mode auto-edit", "/plan calc.py 에 add 추가", "y", "y", reviewer_steps=[REVIEW_OK])
    app.run()
    assert script.prompts.count(TOOL_PROMPT) == 1  # 사용자 공통 메모리만
    text = out.getvalue()
    assert USER_MEMORY_REASON[:12] in text and OUT_OF_PLAN[:10] not in text
    values = app.agent.get_state(app.config).values
    assert values["plan"]["status"] == "approved" and not values.get("plan_scope_changes")
    assert (app.settings.data_dir / "memory" / "AGENTS.md").read_text(encoding="utf-8") == (
        "- 공통\n")


def test_remember_prompt_and_approval(tmp_path: Path) -> None:
    """/remember 는 task8 안내(+메모)로 일반 턴을 보내고, 끝나면 다음 /new 부터라고 알린다.

    yolo 에서도 사용자 공통 메모리 쓰기는 묻는다.
    """
    def remember(messages: list[BaseMessage]) -> AIMessage:
        return tool_call("write_file", {"file_path": "/.code-agent-user/AGENTS.md",
                                        "content": "- 시험은 uv run pytest\n"})

    app, model, out, script = memory_app(tmp_path, [remember, DONE], "/mode yolo",
                                         "/remember 시험은 uv run pytest 로 돌린다", "y", "/exit")
    app.run()
    assert humans(model, 0) == [remember_prompt("시험은 uv run pytest 로 돌린다")]
    assert humans(model, 0)[0].startswith("이번 대화에서 확인된, 앞으로도 지킬 규칙이나 사실만")
    assert "사용자가 기억하라고 한 것: 시험은 uv run pytest 로 돌린다" in humans(model, 0)[0]
    assert script.prompts.count(TOOL_PROMPT) == 1
    text = out.getvalue()
    assert "사용자 공통 메모리" in text and REMEMBER_DONE in text
    assert (tmp_path / "data" / "memory" / "AGENTS.md").read_text(encoding="utf-8") == (
        "- 시험은 uv run pytest\n")
    assert remember_prompt() == remember_prompt("  ")  # 메모가 없으면 안내만


# ---- 신뢰 확인(S11) ----

def test_project_memory_trust_prompt_and_stored(tmp_path: Path) -> None:
    """처음 보는 프로젝트 AGENTS.md 는 묻고, 신뢰하면 읽고 결정을 남긴다.

    같은 내용이면 다음 실행에서 다시 묻지 않는다.
    """
    write_project_memory(tmp_path, "PROJECT-RULE")
    app, model, out, script = memory_app(tmp_path, [DONE], "y", "하이", "/exit",
                                         interactive=True)
    app.run()
    [prompt] = [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert prompt == "이 프로젝트 AGENTS.md(12자)를 메모리로 읽을까? [y/N] › "
    assert "PROJECT-RULE" in out.getvalue()  # 앞부분을 보여 주고 묻는다
    assert "PROJECT-RULE" in system_of(model)
    sha = content_sha256(b"PROJECT-RULE")
    assert app.sessions.memory_decision(app.workspace, sha) is True

    again, model2, _, script2 = memory_app(tmp_path, [DONE], "하이", "/exit",
                                           interactive=True, sessions=app.sessions)
    again.run()
    assert not [p for p in script2.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert "PROJECT-RULE" in system_of(model2)


def test_project_memory_refused(tmp_path: Path) -> None:
    """거절(그냥 Enter 포함)하면 읽지 않고, 같은 실행의 /new 에서는 다시 묻지 않는다."""
    write_project_memory(tmp_path, "PROJECT-RULE")
    app, model, out, script = memory_app(tmp_path, [DONE], "", "하이", "/new", "/exit",
                                         interactive=True)
    tracer = LocalTracer(tmp_path / "data")
    app.tracer = tracer
    first = app.thread_id
    app.run()
    assert len([p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]) == 1
    assert "PROJECT-RULE" not in system_of(model)
    assert "이번 실행에서 거절해" in out.getvalue()
    assert app.sessions.memory_decision(app.workspace, content_sha256(b"PROJECT-RULE")) is False
    [event] = [e for e in read_jsonl(run_dir(app.settings.data_dir, first) / TRACE_FILE)
               if e["type"] == "memory_loaded"]
    assert event["untrusted"] == ["/AGENTS.md"] and event["files"] == []


def test_project_memory_changed_asks_again(tmp_path: Path) -> None:
    """신뢰한 뒤 내용이 바뀌면 다시 묻는다."""
    project = write_project_memory(tmp_path, "PROJECT-V1")
    app, _, _, _ = memory_app(tmp_path, [], "y", "/exit", interactive=True)
    app.run()
    project.write_text("PROJECT-V2", encoding="utf-8")
    again, model, out, script = memory_app(tmp_path, [DONE], "y", "하이", "/exit",
                                           interactive=True, sessions=app.sessions)
    again.run()
    assert [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert "내용이 바뀌었다" in out.getvalue() and "PROJECT-V2" in system_of(model)


def test_project_memory_not_loaded_when_not_interactive(tmp_path: Path) -> None:
    """파이프·시험처럼 물을 수 없으면 신뢰하지 않은 AGENTS.md 는 읽지 않고 흐린 줄로 알린다."""
    write_project_memory(tmp_path, "PROJECT-RULE")
    app, model, out, script = memory_app(tmp_path, [DONE], "하이", "/exit")
    app.run()
    assert "PROJECT-RULE" not in system_of(model)
    assert "신뢰를 확인하지 않아 메모리로 읽지 않는다" in out.getvalue()
    assert not [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert app.sessions.memory_decision(app.workspace, content_sha256(b"PROJECT-RULE")) is None


@pytest.mark.parametrize(("interactive", "loaded"), [(True, True), (False, False)])
def test_main_shares_trust_with_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      interactive: bool, loaded: bool) -> None:
    """main 이 만든 App 과 agent 가 같은 신뢰를 본다. 대화형이면 묻고 읽는다."""
    from test_cli import ENV_A, run_main

    from code_agent import cli

    write_project_memory(tmp_path, "PROJECT-RULE")
    monkeypatch.setattr(cli, "_interactive", lambda: interactive)
    model = ScriptedChatModel(steps=[DONE])
    answers = ("y", "하이", "/exit") if interactive else ("하이", "/exit")
    assert run_main(tmp_path, monkeypatch, ["--workspace", str(tmp_path / "ws")], *answers,
                    detected=ENV_A, model=model) == 0
    assert ("PROJECT-RULE" in system_of(model)) is loaded


def test_trust_is_bound_to_content(tmp_path: Path) -> None:
    """신뢰는 내용 해시에 묶인다. 물은 뒤 첫 요청 전에 바뀐 파일은 읽지 않는다."""
    trust = MemoryTrust(content_sha256(b"A"))
    assert trust.allows(b"A") and not trust.allows(b"B") and not MemoryTrust().allows(b"A")
    write_project_memory(tmp_path, "PROJECT-V1")
    a = Agent(tmp_path, trusted(tmp_path))
    write_project_memory(tmp_path, "PROJECT-V2")  # 신뢰를 정한 뒤 첫 요청 전에 바뀌었다
    agent, model = a.build([DONE])
    [event] = custom(a.turn(agent, "하이"), "memory_loaded")
    assert event["untrusted"] == ["/AGENTS.md"] and "PROJECT-V" not in system_of(model)


def test_memory_middleware_takes_default_slot(tmp_path: Path) -> None:
    """우리 메모리 middleware 는 deepagents 기본 것과 이름이 같아 그 자리(맨 끝 쪽)를 대신한다."""
    (tmp_path / "ws").mkdir()
    backend = make_backend(tmp_path / "ws", artifacts_dir=tmp_path / "a",
                           memory_dir=tmp_path / "m")
    middleware = AgentMemoryMiddleware(backend=backend, sources=["/AGENTS.md"],
                                       trust=MemoryTrust())
    assert middleware.name == "MemoryMiddleware" and (tmp_path / "m").is_dir()


def test_trusted_memory_table_is_separate(tmp_path: Path) -> None:
    """결정은 새 표에 둔다. 기존 표(code_agent_threads)의 열은 그대로다."""
    sessions = Sessions(sqlite3.connect(tmp_path / "s.db"))
    columns = [row[1] for row in sessions.conn.execute("PRAGMA table_info(code_agent_threads)")]
    assert columns == ["thread_id", "workspace", "created_at"]
    sessions.decide_memory(tmp_path, "abc", True)
    sessions.decide_memory(tmp_path, "abc", False)  # 같은 내용을 다시 정하면 덮어쓴다
    assert sessions.memory_decision(tmp_path, "abc") is False
    assert sessions.memory_decision(tmp_path, "other") is None
    assert not sessions.trusted_before(tmp_path)


# ---- 크기 경고 ----

@pytest.mark.parametrize(("chars", "warned"), [(WARN_CHARS, False), (WARN_CHARS + 1, True)])
def test_memory_size_warning(tmp_path: Path, chars: int, warned: bool) -> None:
    """합계가 WARN_CHARS 를 넘으면 시작과 /new 때 노란 줄로 알린다."""
    write_user_memory(tmp_path, "가" * chars)
    app, _, out, _ = memory_app(tmp_path, [], "/new", "/exit")
    app.run()
    assert out.getvalue().count(f"{WARN_CHARS:,}자 초과") == (2 if warned else 0)


def test_memory_size_warning_counts_trusted_project(tmp_path: Path) -> None:
    """프로젝트 AGENTS.md 는 신뢰해 읽을 때만 합계에 든다."""
    write_user_memory(tmp_path, "가" * (WARN_CHARS // 2))
    write_project_memory(tmp_path, "나" * (WARN_CHARS // 2 + 1))
    app, _, out, _ = memory_app(tmp_path, [], "/exit")  # 대화형이 아니라 읽지 않는다
    app.run()
    assert "자 초과" not in out.getvalue()
    app, _, out, _ = memory_app(tmp_path, [], "y", "/exit", interactive=True)
    app.run()
    assert f"{WARN_CHARS + 1:,}자다" in out.getvalue()


# ---- /memory ----

def test_memory_command(tmp_path: Path) -> None:
    """/memory: 읽기 전에는 첫 요청 때 읽는다고, 읽은 뒤에는 글자 수와 디스크 변경을 보인다."""
    user = write_user_memory(tmp_path, "USER-V1")
    write_project_memory(tmp_path, "PROJECT-RULE")
    skill = tmp_path / "ws" / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL.format(name="demo", description="DEMO-SKILL"),
                                    encoding="utf-8")
    app, model, out, script = memory_app(tmp_path, [DONE], "/memory", "하이", "/exit")
    marks: list[int] = []

    def edit_then_show(prompt: str) -> str:
        answer = script(prompt)
        if answer == "/exit":
            user.write_text("USER-V2-LONGER", encoding="utf-8")
            marks.append(len(out.getvalue()))
            app.command("/memory")
        return answer

    app.read_line = edit_then_show
    app.run()
    before, after = out.getvalue()[:marks[0]], out.getvalue()[marks[0]:]
    assert "아직 읽지 않았다" in before and "스킬: 첫 요청 때 읽는다" in before
    assert "디스크 7자 · 첫 요청 때 읽는다" in before
    assert "디스크 12자 · 신뢰하지 않아 읽지 않는다" in before
    assert "읽음 7자 · 디스크에서 바뀌었다(14자, 다음 /new 부터 반영)" in after
    assert "읽지 않음(신뢰하지 않았거나 세션을 시작한 뒤 생겼다)" in after
    assert "스킬 없음" in after and "DEMO-SKILL" not in system_of(model)  # 신뢰하지 않은 스킬
    assert str(tmp_path / "data" / "memory" / "AGENTS.md") in after
    assert humans(model, 0) == ["하이"]


def test_memory_report_same_and_deleted(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    write_user_memory(tmp_path, "USER")
    values = {"memory_contents": {"/.code-agent-user/AGENTS.md": "USER",
                                  "/AGENTS.md": "PROJECT"}, "skills_metadata": []}
    lines = memory_report(values, workspace=tmp_path / "ws", data_dir=tmp_path / "data",
                          trust=MemoryTrust())
    text = "\n".join(lines)
    assert "읽음 4자 · 디스크와 같다" in text
    assert "읽음 7자 · 디스크에서 지워졌다(다음 /new 부터 반영)" in text
    assert "합계 11자" in text and "스킬 없음" in text


# ---- 사건과 작업 기록 ----

def test_memory_events_in_local_trace_and_worklog(tmp_path: Path) -> None:
    """memory_loaded·memory_write 가 로컬 트레이스에 남고 작업 기록의 '메모리' 절이 된다.

    거부한 쓰기는 실행되지 않아 memory_write 가 없다.
    """
    write_user_memory(tmp_path, "USER-RULE")
    tracer = LocalTracer(tmp_path / "data")
    app, _, _, _ = memory_app(
        tmp_path, [tool_calls(("write_file", {"file_path": "/AGENTS.md", "content": "- a\n"}),
                              ("edit_file", {"file_path": "/.code-agent-user/AGENTS.md",
                                             "old_string": "USER-RULE",
                                             "new_string": "USER-RULE\n- b"})),
                   DONE],
        "/mode auto-edit", "기억해", "n", "", "/exit", tracer=tracer)
    app.run()
    rows = read_jsonl(run_dir(app.settings.data_dir, app.thread_id) / TRACE_FILE)
    [loaded] = [e for e in rows if e["type"] == "memory_loaded"]
    assert loaded["files"] == [{"path": "/.code-agent-user/AGENTS.md", "chars": 9}]
    writes = [e for e in rows if e["type"] == "memory_write"]
    assert [(e["path"], e["tool"], e["scope"]) for e in writes] == [
        ("/AGENTS.md", "write_file", "project")]
    assert (tmp_path / "data" / "memory" / "AGENTS.md").read_text(encoding="utf-8") == (
        "USER-RULE")

    log = WorkLog()
    for event in (loaded, *writes, {"type": "memory_write", "path": "/.code-agent-user/AGENTS.md",
                                    "tool": "edit_file", "scope": "user"},
                  {"type": "memory_loaded", "files": [], "total_chars": 0,
                   "untrusted": ["/AGENTS.md"]}):
        log.on_part(((), "custom", event))
    rendered = log.render(answer="끝", changes=[], diff="")
    section = rendered.split("## 메모리\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == [
        "- 이 세션 첫 요청에서 읽음: /.code-agent-user/AGENTS.md 9자 (합계 9자)",
        "- 메모리 쓰기(프로젝트): write_file /AGENTS.md",
        "- 메모리 쓰기(사용자 공통): edit_file /.code-agent-user/AGENTS.md",
        "- 이 세션 첫 요청에서 읽음: 메모리 파일 없음 · 신뢰하지 않아 읽지 않음: /AGENTS.md"]
    assert "## 메모리" not in WorkLog().render(answer="끝", changes=[], diff="")


# ---- 리뷰 지적: 프로젝트 스킬 신뢰, 미리보기, 면제 경로, 자기 쓰기 ----

def write_project_skill(tmp_path: Path, description: str) -> Path:
    folder = tmp_path / "ws" / ".agents" / "skills" / "evil"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(SKILL.format(name="evil", description=description), encoding="utf-8")
    return path


def test_project_skill_needs_trust(tmp_path: Path) -> None:
    """신뢰하지 않은 프로젝트 스킬 설명은 system 에 들어가지 않고, 신뢰하면 들어간다(S11)."""
    write_project_skill(tmp_path, "IGNORE-PREVIOUS-INSTRUCTIONS")
    untrusted = Agent(tmp_path)
    agent, model = untrusted.build([DONE])
    untrusted.turn(agent, "하이")
    assert "IGNORE-PREVIOUS-INSTRUCTIONS" not in system_of(model)

    other = tmp_path / "other"
    (other / "ws").mkdir(parents=True)
    write_project_skill(other, "IGNORE-PREVIOUS-INSTRUCTIONS")
    ok = Agent(other, MemoryTrust(skills_sha256=skills_sha256(other / "ws")))
    agent, model = ok.build([DONE])
    ok.turn(agent, "하이")
    assert "IGNORE-PREVIOUS-INSTRUCTIONS" in system_of(model)

    # 신뢰한 뒤 스킬이 바뀌면(해시가 달라) 읽지 않는다
    write_project_skill(other, "CHANGED-AFTER-TRUST")
    again = Agent(other, ok.trust)
    agent, model = again.build([DONE])
    again.turn(agent, "하이")
    assert "CHANGED-AFTER-TRUST" not in system_of(model)


def test_trust_prompt_covers_skills(tmp_path: Path) -> None:
    """질문에 스킬 수가 들어가고, 신뢰하면 스킬도 읽으며, 스킬이 바뀌면 다시 묻는다."""
    write_project_memory(tmp_path, "PROJECT-RULE")
    skill = write_project_skill(tmp_path, "SKILL-V1")
    app, model, out, script = memory_app(tmp_path, [DONE], "y", "하이", "/exit",
                                         interactive=True)
    app.run()
    [prompt] = [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert prompt == "이 프로젝트 AGENTS.md(12자, 스킬 1개 포함)를 메모리로 읽을까? [y/N] › "
    assert "SKILL-V1" in out.getvalue() and "SKILL-V1" in system_of(model)

    skill.write_text(SKILL.format(name="evil", description="SKILL-V2"), encoding="utf-8")
    again, model2, _, script2 = memory_app(tmp_path, [DONE], "", "하이", "/exit",
                                           interactive=True, sessions=app.sessions)
    again.run()
    assert [p for p in script2.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert "SKILL-V2" not in system_of(model2) and "PROJECT-RULE" not in system_of(model2)


def test_trust_prompt_skill_only(tmp_path: Path) -> None:
    """AGENTS.md 가 없어도 프로젝트 스킬이 있으면 묻는다."""
    write_project_skill(tmp_path, "SKILL-ONLY")
    app, model, _, script = memory_app(tmp_path, [DONE], "y", "하이", "/exit", interactive=True)
    app.run()
    assert ("이 프로젝트 AGENTS.md(없음, 스킬 1개 포함)를 메모리로 읽을까? [y/N] › "
            in script.prompts)
    assert "SKILL-ONLY" in system_of(model)


def test_trust_preview_strips_controls_and_clips(tmp_path: Path) -> None:
    """신뢰 질문의 미리보기에서 터미널 제어 문자를 빼고 긴 줄·긴 파일을 줄인다."""
    body = "\n".join(["안전\x1b[2J\x1b]0;제목\x07끝", "x" * 300, *[f"줄{i}" for i in range(20)]])
    write_project_memory(tmp_path, body)
    app, _, out, _ = memory_app(tmp_path, [DONE], "", "/exit", interactive=True)
    app.run()
    text = out.getvalue()
    assert "\x1b" not in text and "\x07" not in text
    assert "안전[2J]0;제목끝" in text  # 문자만 빠지고 글은 남는다
    assert "x" * 120 + "…" in text and "x" * 121 not in text
    assert "… 나머지 7줄 생략" in text


def test_trust_skipped_for_empty_agents_md(tmp_path: Path) -> None:
    """내용이 없는 AGENTS.md 는 스킬이 없으면 묻지 않는다."""
    for content in ("", " \n\t\n"):
        write_project_memory(tmp_path, content)
        app, _, _, script = memory_app(tmp_path, [DONE], "하이", "/exit", interactive=True)
        app.run()
        assert not [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]


def test_exempt_paths_are_exact_but_asking_is_broad() -> None:
    """계획 차단·범위 면제는 정확한 이름만, 사람에게 묻는 규칙은 대소문자 변형도 잡는다."""
    blocked = PlanContext(block_message="계획을 먼저 승인받아라")
    approved = PlanContext(reviewed=True, approved=True, files=frozenset({"/src/a.py"}))
    for path in ("/.CODE-AGENT-USER/x.py", "/agents.md", "/Agents.md", "/.code-agent-user"):
        action = {"name": "write_file", "args": {"file_path": path, "content": "x"}}
        assert not is_memory_file(path)
        assert auto_decision(action, "auto-edit", blocked) == {
            "type": "reject", "message": AUTO_BLOCK + "계획을 먼저 승인받아라"}
        assert plan_reason(action, approved) == OUT_OF_PLAN
    upper = {"name": "write_file", "args": {"file_path": "/.CODE-AGENT-USER/x.py"}}
    assert auto_decision(upper, "yolo") is None and ask_reason(upper) == USER_MEMORY_REASON
    for path in ("/.code-agent-user/a.md", "\\.code-agent-user\\a.md", "/./AGENTS.md"):
        assert is_memory_file(path)


def test_own_project_memory_write_is_trusted_next_new(tmp_path: Path) -> None:
    """에이전트가 쓴 프로젝트 AGENTS.md 는 다음 /new 에서 다시 묻지 않는다."""
    def remember(messages: list[BaseMessage]) -> AIMessage:
        return tool_call("write_file", {"file_path": "/AGENTS.md", "content": "- 규칙\n"})

    app, _, _, script = memory_app(tmp_path, [remember, DONE], "/mode auto-edit", "/remember",
                                   "/new", "/exit", interactive=True)
    app.run()
    written = "- 규칙\n".encode()
    assert (tmp_path / "ws" / "AGENTS.md").read_bytes() == written
    assert not [p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]
    assert app.sessions.memory_decision(app.workspace, content_sha256(written)) is True
    assert app.memory_trust.sha256 == content_sha256(written)


def test_own_write_does_not_trust_refused_file(tmp_path: Path) -> None:
    """거절한 AGENTS.md 에 덧붙인 쓰기는 신뢰로 남기지 않는다(거절한 내용이 섞여 있다)."""
    write_project_memory(tmp_path, "REFUSED-RULE")

    def append(messages: list[BaseMessage]) -> AIMessage:
        return tool_call("write_file", {"file_path": "/AGENTS.md",
                                        "content": "REFUSED-RULE\n- 덧붙임\n"})

    app, _, _, script = memory_app(tmp_path, [append, DONE], "", "/mode auto-edit", "/remember",
                                   "/new", "n", "/exit", interactive=True)
    app.run()
    assert len([p for p in script.prompts if p.startswith(TRUST_PROMPT_HEAD)]) == 2


def test_remember_reports_when_nothing_written(tmp_path: Path) -> None:
    """/remember 는 메모리에 쓴 것이 있을 때만 '다음 /new 부터' 라고 알린다."""
    app, _, out, _ = memory_app(tmp_path, [DONE], "/remember", "/exit")
    app.run()
    assert NOT_REMEMBERED in out.getvalue() and REMEMBER_DONE not in out.getvalue()


def test_trust_decision_key_normalizes_workspace(tmp_path: Path) -> None:
    """같은 폴더를 다른 철자로 가리켜도 같은 신뢰 결정이다."""
    sessions = Sessions(sqlite3.connect(tmp_path / "s.db"))
    (tmp_path / "ws").mkdir()
    sessions.decide_memory(tmp_path / "ws", "abc", True)
    assert sessions.memory_decision(tmp_path / "ws" / ".." / "ws", "abc") is True
    assert sessions.trusted_before(Path(os.path.normcase(str(tmp_path / "ws"))))
    assert trust_key(b"x", None) == content_sha256(b"x")
    assert trust_key(b"x", "s") != trust_key(b"x", "t") and trust_key(None, None) is None
