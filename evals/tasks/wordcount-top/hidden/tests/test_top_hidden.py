"""--top N 숨김 시험. CLI 를 하위 프로세스로 돌려 내부 함수 이름에 기대지 않는다."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, text, *args):
    path = tmp_path / "input.txt"
    path.write_text(text, encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, str(ROOT / "wordcount.py"), *args, str(path)],
                            capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT,
                            timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_without_top_keeps_old_output(tmp_path):
    assert run(tmp_path, "b a b c b a") == ["b 3", "a 2", "c 1"]


def test_top_limits_lines(tmp_path):
    assert run(tmp_path, "b a b c b a", "--top", "2") == ["b 3", "a 2"]


def test_top_zero_prints_nothing(tmp_path):
    assert run(tmp_path, "b a b", "--top", "0") == []


def test_top_larger_than_word_count(tmp_path):
    assert run(tmp_path, "b a b c", "--top", "10") == ["b 2", "a 1", "c 1"]


def test_top_ties_are_alphabetical(tmp_path):
    assert run(tmp_path, "d c b a c d a b e", "--top", "3") == ["a 2", "b 2", "c 2"]


def test_top_korean_words(tmp_path):
    text = "사과 바나나 사과 배 배 바나나 포도"
    assert run(tmp_path, text, "--top", "2") == ["바나나 2", "배 2"]
