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
DEFAULT_LANGFUSE_URL = "https://cloud.langfuse.com"
# 입력 컨텍스트 상한(토큰). 요약 시점을 정한다. 5분 넘게 쉬어 캐시가 식으면 한 호출의 최악 입력 비용이
# Opus 5.5 기준 200K 는 $0.85, 1M 은 $4.25 라 기본은 200K 로 둔다.
DEFAULT_CONTEXT_TOKENS = 200_000
CONTEXT_TOKENS_RANGE = (20_000, 1_000_000)


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    shell: str = ""  # 비우면 Git Bash, 없으면 PowerShell
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    anthropic_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    langfuse_public_key: str = field(default="", repr=False)
    langfuse_secret_key: str = field(default="", repr=False)
    langfuse_base_url: str = DEFAULT_LANGFUSE_URL
    data_dir: Path = DATA_DIR

    @property
    def langfuse_enabled(self) -> bool:
        """두 키가 모두 있어야 트레이스를 보낸다."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


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
    raw = get("CODE_AGENT_CONTEXT_TOKENS").replace(",", "").replace("_", "")
    low, high = CONTEXT_TOKENS_RANGE
    if raw and not (raw.isdigit() and low <= int(raw) <= high):
        raise ValueError(f"CODE_AGENT_CONTEXT_TOKENS 는 {low:,}~{high:,} 사이의 정수여야 합니다: {raw!r}")
    return Settings(
        model=get("CODE_AGENT_MODEL") or DEFAULT_MODEL,
        effort=effort,
        shell=shell,
        context_tokens=int(raw) if raw else DEFAULT_CONTEXT_TOKENS,
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        openrouter_api_key=get("OPENROUTER_API_KEY"),
        langfuse_public_key=get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=get("LANGFUSE_SECRET_KEY"),
        # Langfuse v5 는 LANGFUSE_BASE_URL 을 쓴다. pjt(v4)의 LANGFUSE_HOST 도 받는다.
        langfuse_base_url=get("LANGFUSE_BASE_URL") or get("LANGFUSE_HOST") or DEFAULT_LANGFUSE_URL,
        data_dir=data_dir,
    )
