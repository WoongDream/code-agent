"""메모를 notes.txt 에 쌓고 보여 주는 작은 CLI."""

from __future__ import annotations

import sys
from pathlib import Path

NOTES = Path(__file__).with_name("notes.txt")


def main(argv: list[str]) -> int:
    if argv[:1] == ["add"] and len(argv) == 2:
        with NOTES.open("a", encoding="utf-8") as file:
            file.write(argv[1] + "\n")
        return 0
    if argv == ["list"]:
        if NOTES.exists():
            print(NOTES.read_text(encoding="utf-8"), end="")
        return 0
    print("사용: python notes.py add <메모> | list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
