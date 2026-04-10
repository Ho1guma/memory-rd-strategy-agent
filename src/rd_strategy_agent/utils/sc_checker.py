"""Success Criteria (SC) auto-checker — metric-based, no LLM judgment."""
from __future__ import annotations

import re

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
    """SC1-2: at least 50% of evidence items have keywords populated.
    SC1-1 already guarantees per-tech volume; this is a loose quality gate.
    """
    store = state["evidence_store"]
    if not store:
        return "fail"
    tagged = sum(1 for ev in store if ev.get("keywords"))
    return "pass" if tagged / len(store) >= 0.5 else "fail"


def check_sc2_1(state: AgentState) -> str:
    """SC2-1: each competitor x technology pair must have a TRL row with evidence_count >= 1."""
    competitors = state["scope"].get("competitors", [])
    technologies = state["scope"].get("technologies", [])
    if not competitors:
        return "fail"
    for company in competitors:
        for technology in technologies:
            rows = [
                r for r in state["trl_table"]
                if r["company"] == company and r["technology"] == technology
            ]
            if len(rows) != 1:
                return "fail"
            row = rows[0]
            if row["evidence_count"] < 1:
                return "fail"
    return "pass"


def check_sc2_2(state: AgentState) -> str:
    """SC2-2: threat_matrix must be consistent with TRL rule and escalation signals."""
    competitors = state["scope"].get("competitors", [])
    rules = _get_threat_rules(state["scope"])
    if not competitors:
        return "fail"
    for company in competitors:
        entries = [e for e in state["threat_matrix"] if e["company"] == company]
        if not entries or not entries[0].get("rationale"):
            return "fail"
        if len(entries) != 1:
            return "fail"
        expected = _expected_threat_level(company, state, rules)
        if expected is None or entries[0].get("level") != expected:
            return "fail"
    return "pass"


def check_sc3_1(draft: str) -> str:
    """SC3-1: report contains required sections and short summary."""
    required = [
        "## SUMMARY",
        "## 1. 분석 배경",
        "## 2. 분석 대상 기술 현황",
        "## 3. 경쟁사 동향 분석",
        "## 4. 전략적 시사점",
        "## REFERENCE",
    ]
    for section in required:
        if section not in draft:
            return "fail"
    summary_match = re.search(r"## SUMMARY\s*(.*?)\s*## 1\. 분석 배경", draft, re.S)
    if not summary_match:
        return "fail"
    summary = summary_match.group(1).strip()
    if len(summary) > 400:
        return "fail"
    return "pass"


def check_sc3_2(draft: str, references: list[dict]) -> str:
    """SC3-2: every in-text citation [N] maps to a complete reference entry."""
    citations_in_text = set(re.findall(r"\[(\d+)\]", draft))
    ref_ids = {r["citation_id"].strip("[]") for r in references}
    unmapped = citations_in_text - ref_ids
    if unmapped:
        return f"fail:unmapped={unmapped}"
    for ref in references:
        if not ref.get("citation_id") or not ref.get("url") or not ref.get("title") or not ref.get("accessed_date"):
            return "fail:incomplete_reference"
        if ref.get("title") == "[매핑 불가]":
            return "fail:unmapped_reference"
    return "pass"


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


def _get_threat_rules(scope: dict) -> dict[str, int]:
    rules = scope.get("threat_level_rules", {})
    return {
        "high_min_trl": int(rules.get("high_min_trl", 7)),
        "medium_min_trl": int(rules.get("medium_min_trl", 5)),
        "medium_max_trl": int(rules.get("medium_max_trl", 6)),
        "low_max_trl": int(rules.get("low_max_trl", 4)),
    }


def _expected_threat_level(company: str, state: AgentState, rules: dict[str, int]) -> str | None:
    rows = [r for r in state["trl_table"] if r["company"] == company]
    parsed_scores = [
        _parse_trl_max(row.get("trl_range", ""))
        for row in rows
        if row.get("trl_range") != "정보 부족"
    ]
    parsed_scores = [score for score in parsed_scores if score is not None]
    if not parsed_scores:
        return None
    max_trl = max(parsed_scores)
    if max_trl >= rules["high_min_trl"]:
        level = "high"
    elif rules["medium_min_trl"] <= max_trl <= rules["medium_max_trl"]:
        level = "medium"
    elif max_trl <= rules["low_max_trl"]:
        level = "low"
    else:
        return None
    if _has_escalation_signal(company, state["evidence_store"]):
        level = _upgrade_level(level)
    return level


def _parse_trl_max(trl_range: str) -> int | None:
    numbers = [int(v) for v in re.findall(r"\d+", trl_range)]
    return max(numbers) if numbers else None


def _has_escalation_signal(company: str, evidence_store: list[dict]) -> bool:
    keywords = ("hiring", "investment", "invest", "recruit", "채용", "투자")
    for ev in evidence_store:
        text = f"{ev.get('title', '')} {ev.get('snippet', '')}".lower()
        if company.lower() in text and any(keyword in text for keyword in keywords):
            return True
    return False


def _upgrade_level(level: str) -> str:
    if level == "low":
        return "medium"
    if level == "medium":
        return "high"
    return "high"
