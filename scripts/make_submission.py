"""제출용 ZIP 을 만든다. git 이 추적하는 파일만 담으므로 .env·.venv·캐시는 애초에 들어가지 않는다.

outputs/(에이전트가 만든 결과물)는 코드가 아니라서 뺀다.
평가 과제의 가짜 .env(evals/tasks/*/repo/.env)는 canary 값만 든 fixture 라 그대로 둔다.
사용: uv run python scripts/make_submission.py [--out dist/code-agent.zip] [--ref HEAD]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = (":(exclude)outputs", ":(exclude)outputs/**")
# 들어가면 안 되는 파일 이름. ZIP 을 열어 다시 확인한다(git 추적 여부와 무관한 마지막 안전망)
FORBIDDEN = {".env", ".env.local"}
ALLOWED_ENV = "evals/tasks/"  # 평가 과제 안의 가짜 .env 만 허용


def build(out: Path, ref: str = "HEAD") -> Path:
    """`git archive` 로 ref 의 추적 파일을 out 에 쓴다.

    Args:
        out: 만들 ZIP 경로. 상위 폴더가 없으면 만든다.
        ref: 담을 git 리비전. 작업 트리의 미커밋 변경은 들어가지 않는다.

    Returns:
        만든 ZIP 경로.

    Raises:
        RuntimeError: ZIP 안에 실제 .env 가 있거나 git archive 가 실패했을 때.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "archive", "--format=zip", f"--output={out}", ref, ".", *EXCLUDE]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git archive 실패: {result.stderr.strip()}")
    bad = [name for name in zipfile.ZipFile(out).namelist()
           if Path(name).name in FORBIDDEN and not name.startswith(ALLOWED_ENV)]
    if bad:
        out.unlink()
        raise RuntimeError(f"ZIP 에 비밀값 파일이 들어 있어 지웠다: {', '.join(bad)}")
    return out


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점. 만든 ZIP 경로와 파일 수를 출력한다."""
    parser = argparse.ArgumentParser(description="제출용 ZIP(추적 파일만, outputs 제외)")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist" / "code-agent.zip"))
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args(argv)
    try:
        out = build(Path(args.out), args.ref)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    count = len(zipfile.ZipFile(out).namelist())
    print(f"{out} ({count}개 파일, {out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
