"""단어 빈도를 세는 작은 CLI.

사용: python wordcount.py FILE
파일의 단어(글자·숫자 묶음, 소문자로 바꾼다)마다 '단어 빈도' 를 한 줄씩, 많이 나온 순서로 출력한다.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"\w+")


def count_words(text: str) -> Counter[str]:
    """소문자로 바꾼 단어의 빈도."""
    return Counter(word.lower() for word in WORD.findall(text))


def format_lines(pairs: list[tuple[str, int]]) -> list[str]:
    """(단어, 빈도) 목록을 '단어 빈도' 줄로 바꾼다."""
    return [f"{word} {count}" for word, count in pairs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="단어 빈도를 센다")
    parser.add_argument("file", help="읽을 UTF-8 텍스트 파일")
    args = parser.parse_args(argv)
    counts = count_words(Path(args.file).read_text(encoding="utf-8"))
    for line in format_lines(counts.most_common()):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
