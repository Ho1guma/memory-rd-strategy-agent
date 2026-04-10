"""
Analysis Agent (T4, T5)
- evidence_store를 바탕으로 기술별·회사별 TRL 추정표 생성 (T4)
- 위협 수준 매트릭스 생성 (T5)
- TRL 4-6 구간은 "추정" 라벨 + 간접 지표 근거 명시
- LangChain ChatOpenAI 호출 → Supervisor에 보고
"""

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents.state import AgentState, TRLRow, ThreatRow
import agents.evidence_index as ev_idx


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
JSON_RETRY = 2  # JSON 파싱 실패 시 재시도 횟수


# ── 프롬프트 ─────────────────────────────────────────────────────

TRL_ANALYSIS_PROMPT = """당신은 반도체 기업의 시니어 기술 전략 분석가입니다.

아래 증거 자료를 바탕으로 각 회사의 기술별 TRL(Technology Readiness Level, 1-9)을 추정하세요.

**분석 대상 — 아래 모든 조합({total_combinations}개)을 빠짐없이 커버할 것 (완결성)**
- 기술: {technologies}
- 자사: {self_company} (company 필드에 "{self_company}"로 표기)
- 경쟁사: {competitors}

**경쟁사별·기술별 증거 건수 (시스템 자동 집계 — evidence_count에 이 수치를 반영할 것)**
{evidence_counts}

**경쟁사별 증거 자료**
{evidence_by_company}

**TRL 추정 규칙**
- TRL 1-3: 논문/학회/특허 근거 → "확정" 라벨
- TRL 4-6: 핵심 영업 비밀 구간 → 반드시 "추정" 라벨, 간접 지표(특허 출원 패턴·학회 빈도·채용 키워드) 명시
- TRL 7-9: 양산/샘플 공급/실적 공시 근거 → "확정" 라벨 (수율·원가는 비공개 전제 명시)

**근거성 규칙 (엄수)**
- 증거 자료에 없는 내용을 추정하거나 지어내지 말 것
- 증거가 1건 이하인 조합도 해당 증거를 기반으로 최선의 TRL 추정을 할 것 (무조건 "정보 부족"으로 처리하지 말 것)
- evidence_refs에는 증거 자료의 인덱스 번호를 문자열로 넣을 것 (예: ["1", "5", "23"])

**출력 형식 — JSON 배열만 출력, 마크다운 코드블록·설명 텍스트 없이**
[
  {{
    "company": "회사명",
    "technology": "기술명",
    "trl_range": "추정 구간 (예: 7-8)",
    "trl_label": "확정 또는 추정",
    "evidence_count": 정수,
    "evidence_refs": ["증거 인덱스 번호"],
    "rationale": "판단 근거 2-3줄"
  }}
]
"""

THREAT_ANALYSIS_PROMPT = """당신은 반도체 R&D 전략 분석가입니다.

아래 TRL 추정표를 바탕으로 경쟁사별·기술별 위협 수준을 평가하세요.

**TRL 추정표**
{trl_summary}

**위협 등급 판정 기준 (일관성 — TRL값과 반드시 일치할 것)**
- 높음: TRL ≥ 7
- 중간: TRL 5-6 (투자·채용 증가 추세가 확인되면 한 단계 상향 가능)
- 낮음: TRL ≤ 4 또는 정보 부족
- **TRL ≤ 4인 경우 투자·채용 추세와 관계없이 반드시 "낮음"으로 판정할 것 (예외 불가)**

**완결성 규칙**: 아래 경쟁사×기술 모든 조합({total_combinations}개)에 대해 행을 생성할 것
대상 경쟁사: {competitors}
대상 기술: {technologies}

**일관성 검증**: TRL 추정표의 trl_range 값과 level이 모순되지 않아야 함

**출력 형식 — JSON 배열만 출력, 마크다운 코드블록·설명 텍스트 없이**
[
  {{
    "company": "회사명",
    "technology": "기술명",
    "level": "높음 또는 중간 또는 낮음",
    "rationale": "위협 판정 근거 1-2줄 (TRL 수치 언급 포함)"
  }}
]
"""


# ── 헬퍼 함수 ────────────────────────────────────────────────────

def _group_evidence_by_company(
    evidence_store: list,
    competitors: list,
    technologies: list,
    max_per_company: int = 10,
) -> tuple[str, dict[str, list[tuple[int, dict]]], str]:
    """
    경쟁사×기술별로 증거를 그룹핑.
    Returns: (formatted_summary, grouped_data, counts_line)
    """
    grouped: dict[str, list[tuple[int, dict]]] = {c: [] for c in competitors}

    for i, item in enumerate(evidence_store):
        text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
        matched = False
        for company in competitors:
            if company.lower() in text:
                grouped[company].append((i, item))
                matched = True
        # 본문에 회사명 없지만 해당 회사 쿼리로 수집된 증거 → query_company로 폴백 매핑
        if not matched and item.get("query_company") in grouped:
            grouped[item["query_company"]].append((i, item))

    lines = []
    count_parts = []

    for company in competitors:
        items = grouped[company]
        tech_counts = {}
        for tech in technologies:
            tech_count = sum(
                1 for _, it in items
                if tech.lower() in (it.get("title", "") + " " + it.get("snippet", "")).lower()
            )
            tech_counts[tech] = tech_count

        tech_str = ", ".join(f"{t}: {tech_counts[t]}건" for t in technologies)
        count_parts.append(f"- {company}: 총 {len(items)}건 ({tech_str})")

        lines.append(f"\n### {company} 관련 증거 ({len(items)}건)")
        for idx, item in items[:max_per_company]:
            lines.append(
                f"  [{idx + 1}] {item.get('title', '')} ({item.get('date', 'N/A')})\n"
                f"      {item.get('snippet', '')[:180]}\n"
                f"      출처: {item.get('url', '')}"
            )
        if len(items) > max_per_company:
            lines.append(f"  ... 외 {len(items) - max_per_company}건")

    return "\n".join(lines), grouped, "\n".join(count_parts)


def _postprocess_trl_table(
    trl_table: list[TRLRow],
    evidence_store: list,
) -> list[TRLRow]:
    """
    LLM 출력 후처리: evidence_count와 evidence_refs를 실제 데이터로 교정.
    LLM이 건수를 과소 보고하거나 가짜 URL을 넣는 문제를 방지.
    query_company 메타데이터도 활용해 본문에 회사명 없는 논문도 카운트.
    """
    for row in trl_table:
        company = row.get("company", "")
        tech = row.get("technology", "")

        matching_indices = []
        matching_urls = []
        for i, item in enumerate(evidence_store):
            text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
            company_match = company.lower() in text or item.get("query_company") == company
            if company_match and tech.lower() in text:
                matching_indices.append(str(i + 1))
                matching_urls.append(item.get("url", ""))

        row["evidence_count"] = len(matching_indices)
        row["evidence_refs"] = matching_urls[:10]

        if row.get("trl_range") == "정보 부족" and len(matching_indices) >= 2:
            print(
                f"[Analysis] ⚠️ {company}/{tech}: LLM이 '정보 부족'으로 판정했으나 "
                f"실제 매칭 증거 {len(matching_indices)}건 — rationale에 '증거 재검토 필요' 추가"
            )
            row["rationale"] = (row.get("rationale", "") +
                                f" [시스템 참고: 실제 매칭 증거 {len(matching_indices)}건 존재]")

    return trl_table


def _parse_json_response(text: str) -> list:
    """LLM 응답에서 JSON 배열 파싱"""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return []


def _validate_trl_row(row: dict) -> bool:
    """TRL 행 필수 필드 검증 (Consistency)"""
    required = {"company", "technology", "trl_range", "trl_label", "evidence_count", "rationale"}
    return required.issubset(row.keys())


def _validate_threat_row(row: dict) -> bool:
    """위협 행 필수 필드 검증"""
    return {"company", "technology", "level", "rationale"}.issubset(row.keys())


def _llm_call_with_retry(llm: ChatOpenAI, prompt: str, validator, label: str) -> list:
    """
    JSON 파싱 실패 또는 필드 검증 실패 시 최대 JSON_RETRY회 재시도
    Error Handling: API 오류도 포착하여 last_error로 전달
    """
    messages = [HumanMessage(content=prompt)]
    for attempt in range(1, JSON_RETRY + 2):
        try:
            response = llm.invoke(messages)
            result = _parse_json_response(response.content)
            if result and all(validator(r) for r in result):
                return result
            print(f"[Analysis] {label} JSON 검증 실패 (시도 {attempt}) — 재요청")
            messages.append(response)
            messages.append(HumanMessage(
                content="JSON 배열 구조가 올바르지 않거나 필수 필드가 누락되었습니다. "
                        "위 형식에 맞게 JSON 배열만 다시 출력하세요. 설명 텍스트 없이."
            ))
        except Exception as e:
            print(f"[Analysis] {label} API 오류 (시도 {attempt}): {e}")
            if attempt > JSON_RETRY:
                raise
    return []


# ── 에이전트 ─────────────────────────────────────────────────────

def _group_evidence_by_company_indexed(
    competitors: list,
    technologies: list,
    evidence_store: list,
    k_per_combo: int = 8,
) -> tuple[str, dict, str]:
    """
    evidence_index 쿼리로 company×technology별 관련 증거 top-K 추출.
    인덱스 미구축 시 기존 전체 전달 방식으로 폴백.
    """
    if not ev_idx.is_index_ready():
        print("[Analysis] evidence_index 미구축 — 기존 전체 전달 방식 사용")
        return _group_evidence_by_company(evidence_store, competitors, technologies)

    # 경쟁사×기술 조합별 쿼리
    grouped: dict[str, list[tuple[int, dict]]] = {c: [] for c in competitors}
    seen_urls: dict[str, set] = {c: set() for c in competitors}

    for company in competitors:
        for tech in technologies:
            query = f"{company} {tech} semiconductor technology TRL development"
            results = ev_idx.query_evidence(query, k=k_per_combo)
            for item in results:
                url = item.get("url", "")
                if url not in seen_urls[company]:
                    seen_urls[company].add(url)
                    # evidence_store에서 원래 인덱스 찾기
                    idx = next(
                        (i for i, e in enumerate(evidence_store)
                         if e.get("url") == url),
                        len(evidence_store) - 1
                    )
                    grouped[company].append((idx, item))

    lines = []
    count_parts = []

    for company in competitors:
        items = grouped[company]
        tech_counts = {}
        for tech in technologies:
            tech_count = sum(
                1 for _, it in items
                if tech.lower() in (it.get("title", "") + " " + it.get("snippet", "")).lower()
            )
            tech_counts[tech] = tech_count

        tech_str = ", ".join(f"{t}: {tech_counts[t]}건" for t in technologies)
        count_parts.append(f"- {company}: 요청 top-{k_per_combo}건 각 기술 쿼리 ({tech_str})")

        lines.append(f"\n### {company} 관련 증거 ({len(items)}건)")
        for idx, item in items:
            lines.append(
                f"  [{idx + 1}] {item.get('title', '')} ({item.get('date', 'N/A')})\n"
                f"      {item.get('snippet', '')[:180]}\n"
                f"      출처: {item.get('url', '')}"
            )

    return "\n".join(lines), grouped, "\n".join(count_parts)


def analysis_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    scope = state.get("scope", {})
    evidence_store = state.get("evidence_store", [])
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])

    if not evidence_store:
        print("[Analysis] ⚠️ evidence_store 비어있음 — 빈 결과 반환")
        return {"trl_table": [], "threat_matrix": []}

    # TRL 분석: 자사 + 경쟁사 모두 포함
    self_company = scope.get("self_company", "SK Hynix")
    analysis_companies = [self_company] + competitors
    evidence_by_company, grouped, evidence_counts = _group_evidence_by_company_indexed(
        analysis_companies, technologies, evidence_store,
    )
    total_combinations = len(technologies) * len(analysis_companies)

    # T4: TRL 추정표
    print(f"[Analysis] TRL 추정표 생성 중 (증거 {len(evidence_store)}건, 조합 {total_combinations}개)...")
    trl_prompt = TRL_ANALYSIS_PROMPT.format(
        technologies=", ".join(technologies),
        competitors=", ".join(competitors),
        self_company=self_company,
        total_combinations=total_combinations,
        evidence_counts=evidence_counts,
        evidence_by_company=evidence_by_company,
    )
    trl_table: list[TRLRow] = _llm_call_with_retry(llm, trl_prompt, _validate_trl_row, "TRL")

    trl_table = _postprocess_trl_table(trl_table, evidence_store)
    print(f"[Analysis] TRL 행 {len(trl_table)}개 생성 (기대: {total_combinations}개)")

    # T5: 위협 수준 매트릭스 (기술별)
    total_threat_combinations = len(technologies) * len(competitors)
    print(f"[Analysis] 위협 수준 매트릭스 생성 중 (조합 {total_threat_combinations}개)...")
    trl_summary = json.dumps(trl_table, ensure_ascii=False, indent=2)
    threat_prompt = THREAT_ANALYSIS_PROMPT.format(
        trl_summary=trl_summary,
        competitors=", ".join(competitors),
        technologies=", ".join(technologies),
        total_combinations=total_threat_combinations,
    )

    quality_feedback = state.get("last_error", "")
    if quality_feedback and ("일관성" in quality_feedback or "TRL" in quality_feedback):
        threat_prompt += (
            "\n\n**[이전 분석 품질 피드백 — 반드시 반영하여 수정]**\n"
            f"{quality_feedback}"
        )
    threat_matrix: list[ThreatRow] = _llm_call_with_retry(llm, threat_prompt, _validate_threat_row, "위협")
    print(f"[Analysis] 위협 매트릭스 행 {len(threat_matrix)}개 생성")

    return {
        "trl_table": trl_table,
        "threat_matrix": threat_matrix,
    }
