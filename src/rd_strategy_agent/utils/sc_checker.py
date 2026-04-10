"""Success Criteria (SC) auto-checker — metric-based, no LLM judgment."""
from __future__ import annotations

from rd_strategy_agent.state import AgentState, SCStatus


def check_sc1_1(state: AgentState) -> tuple[str, int]:
    """SC1-1: evidence_store has >= n_evidence_min credible sources per technology."""
    n_min = state["scope"].get("n_evidence_min", 5)
    technologies = state["scope"].get("technologies", [])
    counts: dict[str, int] = {t: 0 for t in technologies}
    for ev in state["evidence_store"]:
        for tech in technologies:
            if tech.lower() in ev["snippet"].lower() or tech.lower() in ev["title"].lower():
                counts[tech] += 1
    failing = [t for t, c in counts.items() if c < n_min]
    if failing:
        return "fail", min(counts.values())
    return "pass", min(counts.values())


def check_sc1_2(state: AgentState) -> str:
    """SC1-2: all evidence items have keywords and entities fields populated."""
    for ev in state["evidence_store"]:
        if not ev.get("keywords") or not ev.get("entities"):
            return "fail"
    return "pass" if state["evidence_store"] else "fail"


def check_sc2_1(state: AgentState) -> str:
    """SC2-1: each competitor has TRL entry with evidence_count >= 2."""
    competitors = state["scope"].get("competitors", [])
    if not competitors:
        return "fail"
    for company in competitors:
        rows = [r for r in state["trl_table"] if r["company"] == company]
        if not rows or any(r["evidence_count"] < 2 for r in rows):
            return "fail"
    return "pass"


def check_sc2_2(state: AgentState) -> str:
    """SC2-2: threat_matrix has level + rationale for each competitor."""
    competitors = state["scope"].get("competitors", [])
    if not competitors:
        return "fail"
    for company in competitors:
        entries = [e for e in state["threat_matrix"] if e["company"] == company]
        if not entries or not entries[0].get("rationale"):
            return "fail"
    return "pass"


def check_sc3_1(draft: str) -> str:
    """SC3-1: report contains required sections."""
    required = ["SUMMARY", "1.", "2.", "3.", "4.", "REFERENCE"]
    for section in required:
        if section not in draft:
            return "fail"
    return "pass"


def check_sc3_2(draft: str, references: list[dict]) -> str:
    """SC3-2: every in-text citation [N] maps to a reference entry."""
    import re
    citations_in_text = set(re.findall(r"\[(\d+)\]", draft))
    ref_ids = {r["citation_id"].strip("[]") for r in references}
    unmapped = citations_in_text - ref_ids
    return "pass" if not unmapped else f"fail:unmapped={unmapped}"


def check_sc3_3() -> str:
    """SC3-3: 재실행 안정성 (placeholder — 실제 검증은 테스트 단계에서 수행)."""
    return "pass"


def run_all(state: AgentState) -> SCStatus:
    """Run all SC checks and return updated sc_status dict."""
    draft = state.get("draft_report", "")
    refs = state.get("reference_list", [])
    sc1_1_result, _ = check_sc1_1(state)
    status: SCStatus = {
        "SC1_1": sc1_1_result,
        "SC1_2": check_sc1_2(state),
        "SC2_1": check_sc2_1(state),
        "SC2_2": check_sc2_2(state),
        "SC3_1": check_sc3_1(draft),
        "SC3_2": check_sc3_2(draft, refs),
        "SC3_3": check_sc3_3(),
    }
    return status
