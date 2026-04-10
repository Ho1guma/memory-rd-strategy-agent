"""Scope Agent — Task T1.

scope.yaml은 실행 전 수동으로 설정한다.
이 에이전트는 scope.yaml을 읽어 state에 로드하는 단순 로더이다.
"""
from __future__ import annotations

import yaml
from pathlib import Path

from rd_strategy_agent.state import AgentState

SCOPE_YAML_PATH = Path("scope.yaml")


def scope_agent(_state: AgentState) -> dict:
    """T1: Load scope from scope.yaml (manually configured before run)."""
    if not SCOPE_YAML_PATH.exists():
        return {
            "scope": {},
            "last_error": f"scope.yaml not found at {SCOPE_YAML_PATH.resolve()}. Please configure it before running.",
        }

    scope = yaml.safe_load(SCOPE_YAML_PATH.read_text())
    if not scope:
        return {
            "scope": {},
            "last_error": "scope.yaml is empty. Please fill in technologies, competitors, keywords, n_evidence_min.",
        }

    return {"scope": scope}
