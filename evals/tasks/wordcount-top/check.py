"""wordcount-top 검사기. 숨김 시험(--top N 동작과 원래 동작)이 통과해야 통과다.

에이전트가 시험 파일을 더 썼는지와 그 시험이 통과하는지는 점수와 details 에만 남긴다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
HIDDEN = ("tests/test_wordcount_orig_hidden.py", "tests/test_top_hidden.py")


def _pytest(workspace: Path, *paths: str) -> subprocess.CompletedProcess[str]:
    """검사용 사본에서 pytest 를 돌린다. 캐시 폴더를 만들지 않는다."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths],
                          cwd=workspace, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=120, env=env, check=False)


def _last_line(result: subprocess.CompletedProcess[str]) -> str:
    lines = result.stdout.strip().splitlines() or result.stderr.strip().splitlines()
    return lines[-1] if lines else ""


def check(workspace: Path, run: dict) -> dict:
    """숨김 시험을 돌리고, 에이전트가 더한 시험 파일이 있으면 전체 시험도 돌려 기록한다."""
    hidden = _pytest(workspace, *HIDDEN)
    original = {p.name for p in (TASK / "repo" / "tests").glob("test_*.py")}
    hidden_names = {Path(p).name for p in HIDDEN}
    tests = workspace / "tests"
    added = sorted(p.name for p in tests.glob("test_*.py")
                   if p.name not in original | hidden_names) if tests.is_dir() else []
    # 원래 시험 파일에 시험을 더했을 수도 있다
    original_text = (TASK / "repo" / "tests" / "test_wordcount.py").read_text(encoding="utf-8")
    current = tests / "test_wordcount.py"
    extended = current.is_file() and current.read_text(encoding="utf-8") != original_text
    wrote_tests = bool(added) or extended
    all_tests = _pytest(workspace, "tests") if wrote_tests else None
    passed = hidden.returncode == 0
    score = (0.8 if passed else 0.0) + (0.2 if wrote_tests else 0.0)
    return {"pass": passed, "score": round(score, 2),
            "details": {"hidden": _last_line(hidden), "wrote_tests": wrote_tests,
                        "added_test_files": added,
                        "all_tests_pass": (None if all_tests is None
                                           else all_tests.returncode == 0)}}
