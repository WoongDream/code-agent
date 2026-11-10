"""자기 개선 루프: 신호 수집, 개선안 파싱, 정적 검사, 채택 규칙, 승인, 기록, /improve."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from eval_scripts import scripted
from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from rich.console import Console
from test_cli import Script, make_app

from code_agent.evalrun import EvalTask, TaskResult, load_tasks
from code_agent.improve import (
    IMPROVEMENTS_FILE,
    MAX_CHARS,
    Proposal,
    accept_rule,
    collect_signals,
    evaluate,
    improve,
    memory_snapshot,
    parse_improve_args,
    parse_proposals,
    propose,
    static_check,
    target_path,
)
from code_agent.localtrace import RUNS_DIR, append_jsonl, read_jsonl
from code_agent.settings import Settings, default_data_dir, load_settings

SKILL = """---
name: run-tests-first
description: 코드를 고치기 전에 시험부터 돌려 실패를 확인할 때 쓴다.
---

# 시험 먼저

1. python -m pytest 로 실패를 본다.
"""


def ev(turn: int, kind: str, **fields) -> dict:
    return {"ts": "2026-11-09T10:00:00+00:00", "turn": turn, "type": kind, "t": 1.0, **fields}


def write_run(data_dir: Path, thread: str, turns: dict[int, tuple[dict, list[dict]]]) -> None:
    folder = data_dir / RUNS_DIR / thread
    folder.mkdir(parents=True, exist_ok=True)
    for number, (metrics, events) in turns.items():
        for event in events:
            append_jsonl(folder / "trace.jsonl", {**ev(number, "x"), **event, "turn": number})
        append_jsonl(folder / "metrics.jsonl", {
            "turn": number, "thread_id": thread,
            "started_at": f"2026-11-09T10:{number:02d}:00+00:00",
            "status": "ok", "retries": 0, "rejections": 0, **metrics})


def tool(name: str, call: str, status: str, **fields) -> dict:
    return {"type": "tool", "name": name, "call": call, "status": status, **fields}


def problem_run(data_dir: Path) -> None:
    write_run(data_dir, "thr-a", {
        1: ({"retries": 1, "status": "warning"}, [
            {"type": "warning", "message": "출력 한도에 걸려 잘렸다"},
            {"type": "approval", "tool": "execute", "call": "execute: rm -rf build",
             "decision": "reject", "by": "user", "reason": "지우지 마"},
            {"type": "approval", "tool": "write_file", "call": "write_file: a.py",
             "decision": "reject", "by": "auto", "reason": "계획 승인 전"},
            tool("read_file", "read_file /nope.py", "error", error="파일이 없다"),
            tool("read_file", "read_file /nope.py", "error", error="파일이 없다"),
            tool("execute", "execute: python -m pytest -q", "error", exit_code=1,
                 error="1 failed"),
            {"type": "plan_reviewed", "verdict": "revise", "findings": [
                {"severity": "high", "issue": "범위가 너무 넓다"},
                {"severity": "low", "issue": "오타"}]},
        ]),
        2: ({}, [
            tool("execute", "execute: python -m pytest", "error", exit_code=1, error="1 failed"),
            tool("execute", "execute: python -m pytest", "ok", exit_code=0, result="passed"),
        ]),
    })


def kinds(signals) -> list[str]:
    return sorted(s.kind for s in signals)


# --- 신호 수집 ---

def test_collect_signals_from_fake_traces(tmp_path: Path) -> None:
    problem_run(tmp_path)
    signals = collect_signals(tmp_path)
    assert kinds(signals) == ["failed_tests", "plan_review", "rejection", "repeated_error",
                              "retry", "warning"]
    by_kind = {s.kind: s for s in signals}
    assert by_kind["rejection"].thread_id == "thr-a" and by_kind["rejection"].turn == 1
    assert "지우지 마" in by_kind["rejection"].text
    assert "범위가 너무 넓다" in by_kind["plan_review"].text and "오타" not in \
        by_kind["plan_review"].text
    assert "2번" in by_kind["repeated_error"].text
    assert by_kind["failed_tests"].evidence == "thr-a:1"  # 턴 2 는 통과한 시험이 뒤따랐다


def test_collect_signals_limits_to_recent_turns_and_skips_own_runs(tmp_path: Path) -> None:
    problem_run(tmp_path)
    assert collect_signals(tmp_path, limit_turns=1) == []  # 최근 1턴(턴 2)에는 신호가 없다
    write_run(tmp_path, "improve-x", {1: ({}, [{"type": "warning", "message": "내 것"}])})
    write_run(tmp_path, "eval-y", {1: ({}, [{"type": "warning", "message": "평가"}])})
    assert "내 것" not in " ".join(s.text for s in collect_signals(tmp_path))
    assert collect_signals(tmp_path / "none") == []


def test_judge_scores_are_optional_and_never_fatal(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, langfuse_public_key="pk", langfuse_secret_key="sk")
    body = {"data": [{"sessionId": "s1", "scores": [{"name": "judge", "value": 0.2}]},
                     {"sessionId": "s2", "scores": [{"name": "judge", "value": 0.9}]}]}
    signals = collect_signals(tmp_path, settings=settings, fetch=lambda url, headers: body)
    assert [(s.kind, s.thread_id) for s in signals] == [("judge_low", "s1")]

    def boom(url: str, headers: dict) -> dict:
        raise OSError("network")

    assert collect_signals(tmp_path, settings=settings, fetch=boom) == []
    no_keys = Settings(data_dir=tmp_path)
    assert collect_signals(tmp_path, settings=no_keys, fetch=boom) == []


# --- 개선안 ---

def reply(text: str) -> AIMessage:
    return AIMessage(content=text)


GOOD = ('여기 있다.\n```json\n[{"target": "user_memory", "patch": "- 시험을 먼저 돌린다",'
        ' "evidence": ["thr-a:1"], "expected_effect": "실패 감소"}]\n```')


def test_propose_parses_fenced_json() -> None:
    model = ScriptedChatModel(steps=[reply(GOOD)])
    [proposal] = propose([], {}, model)
    assert proposal.target == "user_memory" and proposal.patch == "- 시험을 먼저 돌린다"
    assert proposal.evidence == ["thr-a:1"] and len(model.calls) == 1


def test_propose_retries_once_after_a_parse_failure() -> None:
    model = ScriptedChatModel(steps=[reply("JSON 이 아니다"), reply(GOOD)])
    assert len(propose([], {}, model)) == 1
    assert len(model.calls) == 2
    assert "읽지 못했다" in model.calls[1][-1].text


def test_propose_gives_up_after_the_retry() -> None:
    model = ScriptedChatModel(steps=[reply("아니다"), reply('```json\n{"a": 1}\n```')])
    assert propose([], {}, model) == []
    assert len(model.calls) == 2


def test_propose_schema_errors_count_as_invalid() -> None:
    bad = '```json\n[{"target": "user_skill", "patch": "x"}]\n```'  # skill_name 없음
    model = ScriptedChatModel(steps=[reply(bad), reply(bad)])
    assert propose([], {}, model) == []


def test_parse_proposals_variants() -> None:
    assert parse_proposals("```json\n[]\n```") == []
    [p] = parse_proposals('[{"target": "project_memory", "patch": "- x"}]')  # 코드 블록 없이도
    assert p.target == "project_memory" and p.evidence == []
    with pytest.raises(ValueError):
        parse_proposals('[{"target": "settings", "patch": "x"}]')
    with pytest.raises(ValueError):
        parse_proposals('[{"target": "user_memory", "patch": "  "}]')


def test_proposal_rejects_unsafe_skill_names() -> None:
    for name in ("../evil", "A b", "", "a/b", "x" * 65):
        with pytest.raises(ValidationError):
            Proposal(target="user_skill", skill_name=name, patch=SKILL)
    with pytest.raises(ValidationError):
        Proposal(target="user_memory", skill_name="x", patch="- a")


def test_target_path_whitelist(tmp_path: Path) -> None:
    root, ws = tmp_path / "memory", tmp_path / "ws"
    paths = {
        "user_memory": root / "AGENTS.md", "project_memory": ws / "AGENTS.md"}
    for target, expected in paths.items():
        assert target_path(Proposal(target=target, patch="- a"), memory_root=root,
                           workspace=ws) == expected
    skill = Proposal(target="user_skill", skill_name="run-tests-first", patch=SKILL)
    assert target_path(skill, memory_root=root, workspace=ws) == \
        root / "skills" / "run-tests-first" / "SKILL.md"
    project = Proposal(target="project_skill", skill_name="a", patch=SKILL)
    assert target_path(project, memory_root=root, workspace=ws) == \
        ws / ".agents" / "skills" / "a" / "SKILL.md"


# --- 정적 검사 ---

def mem(patch: str, target: str = "user_memory") -> Proposal:
    return Proposal(target=target, patch=patch)


@pytest.mark.parametrize("secret", [
    "- 키는 sk-ant-api03-abcdefghijklmnop12345",
    "- OPENROUTER_API_KEY=abc123",
    "- DB_TOKEN: abcdef",
    "- 값은 CANARY_b6c40db28062722f",
])
def test_static_check_rejects_secrets(secret: str) -> None:
    assert any("비밀값" in p for p in static_check(mem(secret)))


def test_static_check_size_limit_counts_existing_memory() -> None:
    snapshot = {"/.code-agent-user/AGENTS.md": "- a\n" * 1500, "/AGENTS.md": "- b\n" * 1000}
    assert sum(map(len, snapshot.values())) > MAX_CHARS - 100
    assert any("합계" in p for p in static_check(mem("- 새 항목입니다"), snapshot))
    assert static_check(mem("- 새 항목입니다"), {"/AGENTS.md": "- b\n"}) == []


def test_static_check_duplicates() -> None:
    snapshot = {"/.code-agent-user/AGENTS.md": "- 시험을 먼저  돌린다\n"}
    assert any("중복" in p for p in static_check(mem("- 시험을 먼저 돌린다"), snapshot))
    assert any("중복" in p for p in static_check(mem("- 같다\n- 같다")))
    assert static_check(mem("- 다른 항목"), snapshot) == []


def test_static_check_skill_rules() -> None:
    def skill(patch: str, name: str = "run-tests-first") -> Proposal:
        return Proposal(target="user_skill", skill_name=name, patch=patch)

    assert static_check(skill(SKILL)) == []
    assert any("BOM" in p for p in static_check(skill("﻿" + SKILL)))
    assert any("frontmatter" in p for p in static_check(skill("# 제목만")))
    assert any("name" in p for p in static_check(skill(SKILL, "other-name")))
    no_description = "\n".join(line for line in SKILL.splitlines()
                               if not line.startswith("description:"))
    assert any("description" in p for p in static_check(skill(no_description)))
    assert any("YAML" in p for p in static_check(skill("---\nname: [\n---\nbody\n")))
    assert any("넘는다" in p for p in static_check(skill(SKILL + "x" * MAX_CHARS)))


# --- 채택 규칙 ---

def res(task: str, passed: bool, cost: float | None = 1.0, tokens: int = 100) -> TaskResult:
    return TaskResult(task, passed=passed, cost=cost, tokens=tokens)


def test_accept_rule() -> None:
    base = [res("a", True), res("b", False)]
    assert accept_rule(base, [res("a", True), res("b", True)])[0]
    assert accept_rule(base, [res("a", True), res("b", False)])[0]  # 같아도 채택
    ok, why = accept_rule(base, [res("a", False), res("b", False)])
    assert not ok and "줄었다" in why
    ok, why = accept_rule(base, [res("a", False), res("b", True)])  # 통과 수는 같지만 뒤집힘
    assert not ok and "통과에서 실패" in why and "a" in why


def test_accept_rule_cost_limit_is_130_percent() -> None:
    base = [res("a", True, cost=1.0)]
    assert accept_rule(base, [res("a", True, cost=1.3)])[0]
    ok, why = accept_rule(base, [res("a", True, cost=1.31)])
    assert not ok and "1.3배" in why
    # 비용을 모르면 토큰으로 비교한다
    tok = [res("a", True, cost=None, tokens=1000)]
    assert accept_rule(tok, [res("a", True, cost=None, tokens=1300)])[0]
    assert not accept_rule(tok, [res("a", True, cost=None, tokens=1301)])[0]
    assert accept_rule(tok, [res("a", True, cost=0.5, tokens=1000)])[0]  # 한쪽만 알면 토큰


# --- 평가(대본 모델) ---

def settings_in(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def arms(baseline: str, candidate: str):
    def models(arm: str, task: EvalTask) -> ScriptedChatModel:
        return scripted(task.id, baseline if arm == "baseline" else candidate)
    return models


def test_evaluate_accepts_an_improvement(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    proposal = mem("- 고치기 전에 시험을 본다")
    outcome = evaluate(proposal, settings, load_tasks(["buggy-median"]),
                       models=arms("wrong", "golden"))
    assert outcome.evaluated and outcome.accepted
    assert [r["pass"] for r in outcome.baseline] == [False]
    assert [r["pass"] for r in outcome.candidate] == [True]
    assert not (settings.data_dir / "memory" / "AGENTS.md").exists()  # 사용자 폴더는 그대로


def test_evaluate_rejects_a_regression_and_applies_to_a_copy(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    memory = settings.data_dir / "memory"
    memory.mkdir(parents=True)
    (memory / "AGENTS.md").write_text("- 기존\n", encoding="utf-8")
    seen: list[str] = []

    def models(arm: str, task: EvalTask) -> ScriptedChatModel:
        seen.append(arm)
        return scripted(task.id, "golden" if arm == "baseline" else "wrong")

    outcome = evaluate(mem("- 새 항목"), settings, load_tasks(["buggy-median"]), models=models)
    assert seen == ["baseline", "candidate"]
    assert outcome.evaluated and not outcome.accepted and "줄었다" in outcome.reason
    assert (memory / "AGENTS.md").read_text(encoding="utf-8") == "- 기존\n"


def test_evaluate_skips_project_targets(tmp_path: Path) -> None:
    outcome = evaluate(mem("- x", "project_memory"), settings_in(tmp_path),
                       load_tasks(["buggy-median"]), models=arms("golden", "golden"))
    assert not outcome.evaluated and not outcome.accepted and "평가 대상 아님" in outcome.reason


# --- improve() 흐름 ---

def console_pair() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=140, highlight=False), out


class Asker:
    def __init__(self, *answers: bool) -> None:
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else False


def proposals_reply(*items: dict) -> AIMessage:
    return reply("```json\n" + json.dumps(list(items), ensure_ascii=False) + "\n```")


MEMORY_ITEM = {"target": "user_memory", "patch": "- 고치기 전에 시험을 먼저 돌린다",
               "evidence": ["thr-a:1"], "expected_effect": "실패 감소"}


def events_of(data_dir: Path) -> list[dict]:
    [folder] = [p for p in (data_dir / RUNS_DIR).iterdir() if p.name.startswith("improve-")]
    return read_jsonl(folder / "trace.jsonl")


def test_no_file_change_without_approval(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    console, out = console_pair()
    ask = Asker(False)
    outcomes = improve(settings, tmp_path / "ws", console, ask=ask, no_eval=True,
                       model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]))
    assert [o.decision for o in outcomes] == ["declined"]
    assert not (settings.data_dir / "memory" / "AGENTS.md").exists()
    assert ask.prompts == ["이 개선안을 반영할까? [y/N] › "]
    assert "+- 고치기 전에 시험을 먼저 돌린다" in out.getvalue() and "평가 안 함" in out.getvalue()
    [row] = read_jsonl(settings.data_dir / IMPROVEMENTS_FILE)
    assert row["decision"] == "declined" and row["proposal"]["evidence"] == ["thr-a:1"]
    assert row["evaluation"]["evaluated"] is False


def test_approved_proposal_is_written_logged_and_traced(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    console, _ = console_pair()
    outcomes = improve(settings, tmp_path / "ws", console, ask=Asker(True), no_eval=True,
                       model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]))
    assert [o.decision for o in outcomes] == ["applied"]
    path = settings.data_dir / "memory" / "AGENTS.md"
    assert path.read_text(encoding="utf-8") == "- 고치기 전에 시험을 먼저 돌린다\n"
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    [row] = read_jsonl(settings.data_dir / IMPROVEMENTS_FILE)
    assert row["decision"] == "applied" and row["path"] == str(path)
    types = [e["type"] for e in events_of(settings.data_dir)]
    assert types == ["improve_signals", "improve_proposal", "improve_eval", "improve_decision"]
    signals = events_of(settings.data_dir)[0]
    assert signals["count"] == 6 and signals["kinds"]["rejection"] == 1


def test_a_second_apply_appends_to_existing_memory(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    memory = settings.data_dir / "memory"
    memory.mkdir(parents=True)
    (memory / "AGENTS.md").write_text("- 기존", encoding="utf-8")
    improve(settings, tmp_path, console_pair()[0], ask=Asker(True), no_eval=True,
            model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]))
    assert (memory / "AGENTS.md").read_text(encoding="utf-8") == \
        "- 기존\n- 고치기 전에 시험을 먼저 돌린다\n"


def test_skill_proposal_creates_skill_folder(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    item = {"target": "user_skill", "skill_name": "run-tests-first", "patch": SKILL,
            "evidence": ["thr-a:1"], "expected_effect": "x"}
    improve(settings, tmp_path, console_pair()[0], ask=Asker(True), no_eval=True,
            model=ScriptedChatModel(steps=[proposals_reply(item)]))
    skill = settings.data_dir / "memory" / "skills" / "run-tests-first" / "SKILL.md"
    assert skill.read_text(encoding="utf-8") == SKILL
    assert "/.code-agent-user/skills/run-tests-first/SKILL.md" in memory_snapshot(
        settings.data_dir, tmp_path)


def test_static_failure_is_rejected_without_asking(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    item = {**MEMORY_ITEM, "patch": "- OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv"}
    ask = Asker(True)
    outcomes = improve(settings, tmp_path, console_pair()[0], ask=ask, no_eval=True,
                       model=ScriptedChatModel(steps=[proposals_reply(item)]))
    assert [o.decision for o in outcomes] == ["rejected_static"] and ask.prompts == []
    assert not (settings.data_dir / "memory").exists()
    assert [e["type"] for e in events_of(settings.data_dir)][-1] == "improve_decision"


def test_yes_never_applies_unevaluated_proposals(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    ask = Asker(False)  # 사람이 없다(EOF → 거부)
    items = [MEMORY_ITEM, {**MEMORY_ITEM, "target": "project_memory", "patch": "- 프로젝트 규칙"}]
    outcomes = improve(settings, tmp_path / "ws", console_pair()[0], ask=ask, yes=True,
                       no_eval=True, model=ScriptedChatModel(steps=[proposals_reply(*items)]))
    assert [o.decision for o in outcomes] == ["declined", "declined"]
    assert len(ask.prompts) == 2
    assert not (settings.data_dir / "memory").exists() and not (tmp_path / "ws").exists()


def test_project_memory_needs_human_even_with_eval_on(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    ws = tmp_path / "ws"
    ws.mkdir()
    item = {**MEMORY_ITEM, "target": "project_memory", "patch": "- 프로젝트 규칙"}
    console, out = console_pair()
    ask = Asker(True)
    outcomes = improve(settings, ws, console, ask=ask, yes=True, task_ids=["buggy-median"],
                       model=ScriptedChatModel(steps=[proposals_reply(item)]),
                       models=arms("golden", "golden"))
    assert [o.decision for o in outcomes] == ["applied"]  # 사람이 y 했다
    assert (ws / "AGENTS.md").read_text(encoding="utf-8") == "- 프로젝트 규칙\n"
    assert "평가 대상 아님" in out.getvalue()
    assert ask.prompts == ["이 개선안을 반영할까? [y/N] › "]  # --yes 인데도 물었다


def test_evaluated_and_accepted_proposal_is_applied_with_yes(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    console, out = console_pair()
    ask = Asker()
    outcomes = improve(settings, tmp_path / "ws", console, ask=ask, yes=True,
                       task_ids=["buggy-median"],
                       model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]),
                       models=arms("wrong", "golden"))
    assert [o.decision for o in outcomes] == ["applied"] and ask.prompts == []
    assert "예상 비용 약 $0.60" in out.getvalue()  # 기준 1회 + 후보 1회
    [row] = read_jsonl(settings.data_dir / IMPROVEMENTS_FILE)
    assert row["evaluation"]["accepted"] is True
    assert row["evaluation"]["baseline"][0]["pass"] is False
    assert row["evaluation"]["candidate"][0]["pass"] is True
    types = [e["type"] for e in events_of(settings.data_dir)]
    assert types == ["improve_signals", "improve_proposal", "improve_eval", "improve_decision"]


def test_eval_rejection_blocks_the_write(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    ask = Asker(True)
    outcomes = improve(settings, tmp_path / "ws", console_pair()[0], ask=ask, yes=True,
                       task_ids=["buggy-median"],
                       model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]),
                       models=arms("golden", "wrong"))
    assert [o.decision for o in outcomes] == ["rejected_eval"] and ask.prompts == []
    assert not (settings.data_dir / "memory" / "AGENTS.md").exists()


def test_eval_cost_is_confirmed_and_can_be_declined(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    problem_run(settings.data_dir)
    ask = Asker(False, False)  # 평가는 안 하고, 반영도 안 한다
    outcomes = improve(settings, tmp_path / "ws", console_pair()[0], ask=ask,
                       model=ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)]))
    assert ask.prompts[0].startswith("평가를 진행할까?")
    assert [o.decision for o in outcomes] == ["declined"]


def test_no_signals_means_no_model_call(tmp_path: Path) -> None:
    model = ScriptedChatModel(steps=[])
    out = improve(settings_in(tmp_path), tmp_path, console_pair()[0], ask=Asker(), model=model)
    assert out == [] and model.calls == []


def test_parse_improve_args() -> None:
    assert parse_improve_args("") == (False, None, None)
    assert parse_improve_args("--no-eval --tasks a,b") == (True, ["a", "b"], None)
    assert parse_improve_args("--tasks=a") == (False, ["a"], None)
    assert parse_improve_args("--bogus")[2]


# --- /improve (REPL) ---

def test_improve_command_end_to_end_without_eval(tmp_path: Path) -> None:
    app, _, out, _ = make_app(tmp_path, [], "/improve --no-eval", "y", "/exit")
    problem_run(app.settings.data_dir)
    app.improve_model = ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)])
    assert app.run() == 0
    text = out.getvalue()
    assert "신호 6개" in text and "반영했다" in text
    memory = app.settings.data_dir / "memory" / "AGENTS.md"
    assert memory.read_text(encoding="utf-8") == "- 고치기 전에 시험을 먼저 돌린다\n"
    assert [r["decision"] for r in read_jsonl(app.settings.data_dir / IMPROVEMENTS_FILE)] == \
        ["applied"]


def test_improve_command_declined_and_bad_args(tmp_path: Path) -> None:
    app, _, out, script = make_app(tmp_path, [], "/improve --bogus", "/improve --no-eval", "n",
                                   "/exit")
    problem_run(app.settings.data_dir)
    app.improve_model = ScriptedChatModel(steps=[proposals_reply(MEMORY_ITEM)])
    app.run()
    assert "알 수 없는 인자" in out.getvalue() and "반영하지 않았다" in out.getvalue()
    assert not (app.settings.data_dir / "memory" / "AGENTS.md").exists()
    assert isinstance(script, Script)


def test_improve_command_survives_a_failing_model(tmp_path: Path) -> None:
    app, _, out, _ = make_app(tmp_path, [], "/improve --no-eval", "/exit")
    problem_run(app.settings.data_dir)
    app.improve_model = ScriptedChatModel(steps=[])  # 첫 호출에서 대본 소진 오류
    assert app.run() == 0
    assert "/improve 실패" in out.getvalue()


# --- CODE_AGENT_DATA_DIR ---

def test_data_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_AGENT_DATA_DIR", str(tmp_path / "iso"))
    assert default_data_dir() == tmp_path / "iso"
    assert load_settings().data_dir == tmp_path / "iso"
    assert load_settings(tmp_path / "explicit").data_dir == tmp_path / "explicit"
    monkeypatch.setenv("CODE_AGENT_DATA_DIR", "  ")
    assert default_data_dir() == Path.home() / ".code-agent"
    monkeypatch.delenv("CODE_AGENT_DATA_DIR")
    assert default_data_dir() == Path.home() / ".code-agent"


def test_main_eval_and_improve_use_the_isolated_data_dir(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from code_agent.cli import main, parse_args
    monkeypatch.setenv("CODE_AGENT_DATA_DIR", str(tmp_path / "iso"))
    args = parse_args(["--improve", "--yes", "--no-eval", "--tasks", "a,b"])
    assert args.improve and args.yes and args.no_eval and args.tasks == "a,b" and not args.eval
    assert main(["--eval", "--tasks", "nope", "--yes"]) == 2  # 없는 과제는 실행 전에 거른다
    assert main(["--improve", "--no-eval", "-w", str(tmp_path / "missing")]) == 2
    assert (tmp_path / "iso").is_dir()
