"""Scope Agent — Task T1.

Outcome: scope.yaml is populated with technologies, competitors, keywords,
n_evidence_min, and max_competitors.
"""
from __future__ import annotations

import yaml
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from rd_strategy_agent.state import AgentState

SCOPE_SYSTEM = """You are the Scope Agent for an R&D technology strategy analysis system.
Your job is to confirm or refine the analysis scope based on the user query.

Output ONLY valid YAML with exactly these fields:
- technologies: list of technology names (max 3)
- competitors: list of company names (max 5)
- keywords: list of search keywords (5–15 items)
- n_evidence_min: integer (recommended: 5)
- max_competitors: integer (max 5)

Do not add any explanation outside the YAML block.
"""

SCOPE_YAML_PATH = Path("scope.yaml")


def scope_agent(state: AgentState) -> dict:
    """T1: Determine and lock analysis scope."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    user_query = state.get("next_task", "Analyze HBM4, PIM, CXL R&D landscape and competitor positioning.")

    messages = [
        SystemMessage(content=SCOPE_SYSTEM),
        HumanMessage(content=f"User query: {user_query}\n\nCurrent scope.yaml:\n{SCOPE_YAML_PATH.read_text() if SCOPE_YAML_PATH.exists() else 'empty'}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    scope = yaml.safe_load(raw)

    # Write confirmed scope back to disk
    SCOPE_YAML_PATH.write_text(yaml.dump(scope, allow_unicode=True, default_flow_style=False))

    return {"scope": scope, "next_task": "T2"}
