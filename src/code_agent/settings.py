"""설정. 셸 환경변수 > 저장소 .env > ~/.code-agent/.env 순으로 읽는다. 키 값은 어디에도 출력하지 않는다.

작업 폴더의 .env 는 읽지 않는다. 그 파일은 에이전트가 다루는 대상 프로젝트의 비밀값이다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path.home() / ".code-agent"

DEFAULT_MODEL = "anthropic:claude-opus-5-5"
DEFAULT_EFFORT = "high"  # Opus 5.5 의 API 기본값은 medium 이라 항상 명시한다
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SHELLS = ("bash", "powershell")


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    shell: str = ""  # 비우면 Git Bash, 없으면 PowerShell
    anthropic_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    data_dir: Path = DATA_DIR


def load_settings(data_dir: Path = DATA_DIR) -> Settings:
    values = {**dotenv_values(data_dir / ".env"), **dotenv_values(REPO_ROOT / ".env"), **os.environ}

    def get(name: str) -> str:
        return (values.get(name) or "").strip()

    effort = get("CODE_AGENT_EFFORT").lower() or DEFAULT_EFFORT
    if effort not in EFFORTS:
        raise ValueError(f"CODE_AGENT_EFFORT 는 {', '.join(EFFORTS)} 중 하나여야 합니다: {effort!r}")
    shell = get("CODE_AGENT_SHELL").lower()
    if shell and shell not in SHELLS:
        raise ValueError(f"CODE_AGENT_SHELL 은 {', '.join(SHELLS)} 중 하나이거나 비어 있어야 합니다: {shell!r}")
    return Settings(
        model=get("CODE_AGENT_MODEL") or DEFAULT_MODEL,
        effort=effort,
        shell=shell,
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        openrouter_api_key=get("OPENROUTER_API_KEY"),
        data_dir=data_dir,
    )
