"""
Supervisor Agent
- Phase 1: SC(Success Criteria)를 계측 가능한 수치로 자동 판정
- Phase 2: 규칙 기반 품질 체크 (TRL-위협 일관성, SUMMARY 길이, 금지 표현, 인용 범위)
- Phase 3: LLM 기반 품질 검수 (환각 탐지, 인용 정확성, Compliance, 톤) — 보고서 단계에만 적용
- 판정 결과에 따라 다음 노드를 결정 (라우팅)
- 재시도 카운터 관리 및 에스컬레이션
"""

import re

from agents.state import AgentState, SCStatus
from agents.quality_review import review_report_quality


# ── SC1: 증거 수집 단계 ───────────────────────────────────────────

def _check_sc1(state: AgentState) -> SCStatus:
    scope = state.get("scope", {})
    n_min = scope.get("n_evidence_min", 5)
    technologies = scope.get("technologies", [])
    evidence_store = state.get("evidence_store", [])
    sc_status = dict(state.get("sc_status", {}))

    # SC1-1: 기술별 공신력 출처 건수 ≥ n_evidence_min
    counts = {tech: 0 for tech in technologies}
    for item in evidence_store:
        for tech in technologies:
            if tech.lower() in item.get("snippet", "").lower() or tech.lower() in item.get("title", "").lower():
                counts[tech] += 1

    sc1_1_pass = all(v >= n_min for v in counts.values())
    sc_status["sc1_1"] = "pass" if sc1_1_pass else "fail"
    sc_status["sc1_1_count"] = min(counts.values()) if counts else 0

    # SC1-2: 모든 수집 스니펫에 keywords, entities 필드 존재
    sc1_2_pass = all(
        "keywords" in item and "entities" in item
        for item in evidence_store
    ) if evidence_store else False
    sc_status["sc1_2"] = "pass" if sc1_2_pass else "fail"

    return sc_status


# ── SC2: 분석 단계 ────────────────────────────────────────────────

def _check_trl_threat_consistency(trl_table: list, threat_matrix: list) -> list[str]:
    """
    TRL 수치와 위협 등급 교차 검증 (PROJECT_PLAN 2.3 SC2-2 규칙).
    규칙: TRL ≥ 7 → 높음, 5-6 → 중간, ≤ 4 → 낮음
    """
    issues = []
    for threat_row in threat_matrix:
        company = threat_row.get("company", "")
        level = threat_row.get("level", "")
        company_trls = [r for r in trl_table if r.get("company") == company]

        for trl_row in company_trls:
            trl_range = str(trl_row.get("trl_range", ""))
            if trl_range == "정보 부족":
                if level == "높음":
                    issues.append(f"{company}: TRL '정보 부족'인데 위협 '높음' — 근거 없는 위협 상향")
                continue

            nums = re.findall(r"\d+", trl_range)
            if not nums:
                continue
            max_trl = max(int(n) for n in nums)

            if max_trl >= 7 and level == "낮음":
                issues.append(f"{company}: TRL {trl_range}(≥7)인데 위협 '낮음' → 규칙상 '높음'이어야 함")
            elif max_trl <= 4 and level == "높음":
                issues.append(f"{company}: TRL {trl_range}(≤4)인데 위협 '높음' → 규칙상 '낮음'이어야 함")
    return issues


def _check_evidence_refs(trl_table: list, evidence_store: list) -> list[str]:
    """TRL 행의 evidence_refs URL이 실제 evidence_store에 존재하는지 검증"""
    evidence_urls = {item.get("url", "") for item in evidence_store}
    invalid = []
    for row in trl_table:
        for ref_url in row.get("evidence_refs", []):
            if ref_url and ref_url not in evidence_urls:
                invalid.append(f"{row.get('company')}/{row.get('technology')}: {ref_url[:60]}")
    return invalid


def _check_sc2(state: AgentState) -> SCStatus:
    scope = state.get("scope", {})
    competitors = scope.get("competitors", [])
    trl_table = state.get("trl_table", [])
    threat_matrix = state.get("threat_matrix", [])
    evidence_store = state.get("evidence_store", [])
    sc_status = dict(state.get("sc_status", {}))

    # SC2-1: 각 경쟁사별 evidence_count ≥ 2
    covered = {row["company"]: row["evidence_count"] for row in trl_table}
    missing = [c for c in competitors if covered.get(c, 0) < 2]
    sc_status["sc2_1"] = "pass" if not missing else "fail"
    sc_status["sc2_1_missing"] = missing

    # SC2-2: 위협 매트릭스 내 모든 경쟁사에 level + rationale 존재
    threat_companies = {row["company"] for row in threat_matrix if row.get("level") and row.get("rationale")}
    sc2_2_pass = all(c in threat_companies for c in competitors)
    sc_status["sc2_2"] = "pass" if sc2_2_pass else "fail"

    # SC2-consistency: TRL ↔ 위협 등급 교차 검증
    consistency_issues = _check_trl_threat_consistency(trl_table, threat_matrix)
    sc_status["sc2_consistency"] = "pass" if not consistency_issues else "fail"
    sc_status["sc2_consistency_issues"] = consistency_issues

    # SC2-refs: evidence_refs URL 유효성
    invalid_refs = _check_evidence_refs(trl_table, evidence_store)
    sc_status["sc2_refs_valid"] = "pass" if not invalid_refs else "fail"
    sc_status["sc2_refs_invalid"] = invalid_refs

    return sc_status


# ── SC3: 보고서 단계 ──────────────────────────────────────────────

FORBIDDEN_PATTERNS = [
    (r"수율[이가은는을]\s*(?:약\s*)?\d+[%％]", "수율 수치 단정"),
    (r"원가[가이은는를]\s*(?:약\s*)?\d+", "원가 수치 단정"),
    (r"수율[이가은는을]\s*\d+", "수율 수치 단정"),
]


def _check_sc3(state: AgentState) -> SCStatus:
    draft = state.get("draft_report", "")
    reference_list = state.get("reference_list", [])
    evidence_store = state.get("evidence_store", [])
    sc_status = dict(state.get("sc_status", {}))

    # SC3-1: SUMMARY + 목차 + REFERENCE 섹션 존재
    required_sections = ["SUMMARY", "1.", "2.", "3.", "4.", "6.", "REFERENCE"]
    sc3_1_pass = all(sec in draft for sec in required_sections)
    sc_status["sc3_1"] = "pass" if sc3_1_pass else "fail"

    # SC3-2: REFERENCE 내 citation_id, url, title, accessed_date 모두 존재
    sc3_2_pass = bool(reference_list) and all(
        ref.get("citation_id") and ref.get("url") and ref.get("title") and ref.get("accessed_date")
        for ref in reference_list
    )
    sc_status["sc3_2"] = "pass" if sc3_2_pass else "fail"

    # SC3-summary_len: SUMMARY 400자 이내 (PROJECT_PLAN SC3-1 요구사항)
    summary_match = re.search(
        r"(?:^|\n)(?:#+\s*)?SUMMARY\s*\n(.*?)(?=\n(?:#+\s*)?\d+\.|\Z)",
        draft, re.DOTALL,
    )
    if summary_match:
        summary_text = re.sub(r"[#*_\[\]()]", "", summary_match.group(1)).strip()
        sc_status["sc3_summary_len"] = "pass" if len(summary_text) <= 400 else "fail"
        sc_status["sc3_summary_actual_len"] = len(summary_text)
    else:
        sc_status["sc3_summary_len"] = "fail"
        sc_status["sc3_summary_actual_len"] = -1

    # SC3-forbidden: 수율·원가 단정 금지 패턴 검출
    found_forbidden = []
    for pattern, label in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, draft)
        if matches:
            found_forbidden.append(f"{label}: {matches[0]}")
    sc_status["sc3_forbidden"] = "pass" if not found_forbidden else "fail"
    sc_status["sc3_forbidden_found"] = found_forbidden

    # SC3-citation_bounds: 인용 번호가 evidence_store 범위 내인지 확인
    citation_ids = [int(c) for c in re.findall(r"\[(\d+)\]", draft)]
    if citation_ids:
        max_citation = max(citation_ids)
        evidence_count = len(evidence_store)
        sc_status["sc3_citation_bounds"] = "pass" if max_citation <= evidence_count else "fail"
    else:
        sc_status["sc3_citation_bounds"] = "fail"

    return sc_status


def _build_structural_feedback(sc_status: dict) -> str:
    """SC3 구조·규칙 검증 실패 시 구체적 피드백 생성"""
    lines = []
    if sc_status.get("sc3_1") != "pass":
        lines.append("- 필수 섹션 누락: SUMMARY, 1.~6., REFERENCE 섹션이 모두 존재해야 합니다")
    if sc_status.get("sc3_2") != "pass":
        lines.append("- REFERENCE 항목의 citation_id/url/title/accessed_date 필드가 불완전합니다")
    if sc_status.get("sc3_summary_len") != "pass":
        actual = sc_status.get("sc3_summary_actual_len", "?")
        lines.append(f"- SUMMARY가 400자를 초과합니다 (현재 {actual}자) — 핵심만 압축하세요")
    if sc_status.get("sc3_forbidden") != "pass":
        found = sc_status.get("sc3_forbidden_found", [])
        lines.append(f"- Compliance 위반 표현 감지: {'; '.join(found)} — 수율·원가 단정 금지")
    if sc_status.get("sc3_citation_bounds") != "pass":
        lines.append("- 본문 인용 번호가 증거 자료 범위를 초과합니다 — 존재하지 않는 출처를 인용하고 있음")
    return "\n".join(lines) if lines else "구조·규칙 검증 미달"


# ── Supervisor 노드 ───────────────────────────────────────────────

def supervisor_after_retrieve(state: AgentState) -> dict:
    """Retrieve/WebSearch 완료 후: SC1 판정 → analysis 진입 or 재시도 or escalate"""
    sc_status = _check_sc1(state)
    iteration_count = state.get("iteration_count", 0)
    max_retry = state.get("max_retry", 3)

    print(f"[Supervisor] SC1 판정: {sc_status['sc1_1']} (최소 건수={sc_status['sc1_1_count']}), SC1-2: {sc_status['sc1_2']}")

    if sc_status["sc1_1"] == "pass" and sc_status["sc1_2"] == "pass":
        return {"sc_status": sc_status, "next": "analysis"}

    if iteration_count >= max_retry:
        msg = f"SC1 미달 ({iteration_count}회 재시도 후 포기): sc1_1={sc_status['sc1_1']}, sc1_2={sc_status['sc1_2']}"
        print(f"[Supervisor] {msg}")
        return {"sc_status": sc_status, "last_error": msg, "next": "escalate"}

    print(f"[Supervisor] SC1 미달 → 재시도 ({iteration_count + 1}/{max_retry})")
    return {
        "sc_status": sc_status,
        "iteration_count": iteration_count + 1,
        "next": "retrieve",
    }


def supervisor_after_analysis(state: AgentState) -> dict:
    """
    Analysis 완료 후: SC2 판정 → report 진입 or 재시도
    - 구조 미달(evidence_count 부족) → retrieve로 증거 보강
    - 품질 미달(TRL-위협 불일치, refs 무효) → analysis 직접 재시도
    """
    sc_status = _check_sc2(state)
    iteration_count = state.get("iteration_count", 0)
    max_retry = state.get("max_retry", 3)

    structural_pass = sc_status["sc2_1"] == "pass" and sc_status["sc2_2"] == "pass"
    quality_pass = sc_status.get("sc2_consistency") == "pass" and sc_status.get("sc2_refs_valid") == "pass"

    print(
        f"[Supervisor] SC2 판정: sc2_1={sc_status['sc2_1']}, sc2_2={sc_status['sc2_2']}, "
        f"consistency={sc_status.get('sc2_consistency')}, refs={sc_status.get('sc2_refs_valid')}"
    )

    if structural_pass and quality_pass:
        return {"sc_status": sc_status, "next": "report"}

    if iteration_count >= max_retry:
        msg = (
            f"SC2 미달 ({iteration_count}회 재시도 후 포기): "
            f"missing={sc_status.get('sc2_1_missing')}, "
            f"consistency_issues={sc_status.get('sc2_consistency_issues')}"
        )
        print(f"[Supervisor] {msg}")
        return {"sc_status": sc_status, "last_error": msg, "next": "escalate"}

    if not structural_pass:
        print(f"[Supervisor] SC2 구조 미달 → 증거 보강 재시도 ({iteration_count + 1}/{max_retry})")
        return {
            "sc_status": sc_status,
            "iteration_count": iteration_count + 1,
            "next": "retrieve",
        }

    # 구조는 통과했지만 품질(일관성/refs) 미달 → analysis 직접 재시도
    feedback_parts = []
    if sc_status.get("sc2_consistency") != "pass":
        feedback_parts.append(
            f"TRL-위협 일관성 오류: {'; '.join(sc_status.get('sc2_consistency_issues', []))}"
        )
    if sc_status.get("sc2_refs_valid") != "pass":
        feedback_parts.append(
            f"evidence_refs URL 불일치 {len(sc_status.get('sc2_refs_invalid', []))}건 "
            f"— evidence_store에 없는 URL을 참조하고 있음"
        )
    feedback = "\n".join(feedback_parts)
    print(f"[Supervisor] SC2 품질 미달 → analysis 재시도 ({iteration_count + 1}/{max_retry})\n  {feedback}")

    return {
        "sc_status": sc_status,
        "iteration_count": iteration_count + 1,
        "last_error": feedback,
        "next": "analysis",
    }


def supervisor_after_report(state: AgentState) -> dict:
    """
    Report 완료 후: 2단계 검증
    - Phase 1: 구조·규칙 기반 SC3 판정
    - Phase 2: LLM 품질 검수 (환각·인용·Compliance·톤)
    """
    sc_status = _check_sc3(state)
    iteration_count = state.get("iteration_count", 0)
    max_retry = state.get("max_retry", 3)

    structural_pass = (
        sc_status["sc3_1"] == "pass"
        and sc_status["sc3_2"] == "pass"
        and sc_status.get("sc3_summary_len") == "pass"
        and sc_status.get("sc3_forbidden") == "pass"
        and sc_status.get("sc3_citation_bounds") == "pass"
    )

    print(
        f"[Supervisor] SC3 Phase 1 (구조·규칙): "
        f"sc3_1={sc_status['sc3_1']}, sc3_2={sc_status['sc3_2']}, "
        f"summary_len={sc_status.get('sc3_summary_len')}, "
        f"forbidden={sc_status.get('sc3_forbidden')}, "
        f"citation_bounds={sc_status.get('sc3_citation_bounds')}"
    )

    # Phase 1 실패 → 구조 피드백과 함께 재작성 or escalate
    if not structural_pass:
        if iteration_count >= max_retry:
            msg = f"SC3 구조·규칙 미달 ({iteration_count}회 재시도 후 포기)"
            print(f"[Supervisor] {msg}")
            return {"sc_status": sc_status, "last_error": msg, "next": "escalate"}

        feedback = _build_structural_feedback(sc_status)
        print(f"[Supervisor] SC3 Phase 1 미달 → 보고서 재작성 ({iteration_count + 1}/{max_retry})")
        return {
            "sc_status": sc_status,
            "iteration_count": iteration_count + 1,
            "last_error": f"[구조·규칙 검증 실패]\n{feedback}",
            "next": "report",
        }

    # Phase 2: LLM 품질 검수
    print("[Supervisor] SC3 Phase 1 통과 → Phase 2 LLM 품질 검수 시작...")
    quality_result = review_report_quality(
        state.get("draft_report", ""),
        state.get("evidence_store", []),
        state.get("trl_table", []),
        state.get("threat_matrix", []),
    )

    sc_status["sc3_quality_review"] = "pass" if quality_result.get("pass", True) else "fail"
    sc_status["sc3_quality_issues"] = quality_result.get("issues", [])

    issues_summary = []
    for issue in quality_result.get("issues", []):
        severity = issue.get("severity", "?")
        criterion = issue.get("criterion", "?")
        detail = issue.get("detail", "")
        issues_summary.append(f"  [{severity}] {criterion}: {detail}")

    if issues_summary:
        print(f"[Supervisor] 품질 검수 이슈 {len(issues_summary)}건:\n" + "\n".join(issues_summary))

    if not quality_result.get("pass", True):
        if iteration_count >= max_retry:
            msg = f"품질 검수 미달 ({iteration_count}회 재시도 후 포기)"
            print(f"[Supervisor] {msg}")
            return {"sc_status": sc_status, "last_error": msg, "next": "escalate"}

        feedback = quality_result.get("feedback", "LLM 품질 검수 미달")
        print(f"[Supervisor] SC3 Phase 2 미달 → 보고서 재작성 ({iteration_count + 1}/{max_retry})")
        return {
            "sc_status": sc_status,
            "iteration_count": iteration_count + 1,
            "last_error": f"[품질 검수 실패]\n{feedback}",
            "next": "report",
        }

    print("[Supervisor] SC 전체 + 품질 검수 통과 → 보고서 완성")
    return {"sc_status": sc_status, "next": "end"}


def escalate(state: AgentState) -> dict:
    """최대 재시도 초과 시 진행 불가 사유 출력"""
    error = state.get("last_error", "알 수 없는 오류")
    evidence_count = len(state.get("evidence_store", []))
    print("\n" + "=" * 60)
    print("[Supervisor] ⛔ ESCALATION — 인간 개입 필요")
    print(f"  사유: {error}")
    print(f"  현재 수집된 증거: {evidence_count}건")
    print("  조치: scope.yaml의 n_evidence_min 조정 또는 키워드 수정 후 재실행")
    print("=" * 60 + "\n")
    return {}
