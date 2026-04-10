"""
Report Agent (T5, T6)
- T5: TRL 표 + 위협 매트릭스 + evidence_store → 보고서 초안 생성
- T6: REFERENCE 정합성 검증 + SUMMARY 압축
- 출력: draft_report (Markdown), reference_list
"""

import json
import os
import re
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents.state import AgentState, ReferenceItem
import agents.evidence_index as ev_idx


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TODAY = date.today().isoformat()


# ── 프롬프트 ──────────────────────────────────────────────────────

REPORT_PROMPT = """당신은 {self_company} R&D 부서의 기술 전략 보고서 작성 전문가입니다.

**독자 (Target Actor)**
- {self_company} R&D 담당자 및 기술 전략 임원 (반도체 회로·공정·시스템 고급 전문가)
- 톤: 객관적·간결·논거 중심 (감탄사·수사적 표현 지양)
- 문장 길이: 1문장 최대 2줄 이내
- 오늘 날짜: {today}

**분석 범위**
- 기술: {technologies}
- 경쟁사: {competitors}

**=== 아래 데이터를 보고서의 근거로 사용하세요 ===**

**[데이터 A] 비교표 (자사 ★ 포함, 시스템 자동 생성 — 그대로 삽입)**
{prebuilt_tables}

**[데이터 B] 위협 수준 매트릭스**
{threat_matrix}

**[데이터 C] 수집 증거 ({evidence_count}건) — 본문에서 반드시 [N] 형식으로 인용**
{evidence_summary}

**=== 보고서 작성 지시 ===**

아래 섹션 순서와 제목을 **정확히** 지켜서 Markdown으로 작성하세요.
모든 사실적 주장에는 반드시 [N] 인용을 달아야 합니다.

---

**# SUMMARY**
- 400자 이내로 보고서 핵심을 압축
- 주요 경쟁사의 TRL 수준과 위협 등급 핵심 수치를 포함

**# 0. 보고서 개요**
- 보고서명: {self_company} R&D 기술 전략 분석 보고서
- 작성일: {today}
- 대상 독자: {self_company} R&D 담당자 및 기술 전략 임원
- 보안 등급: 내부용

**# 1. 분석 배경**
- 왜 지금 {technologies}을 분석해야 하는지 구체적으로 서술
- 시장 동향 데이터는 [데이터 C]의 증거를 인용하여 뒷받침 (수치 임의 생성 금지)
- 최소 3개 이상의 [N] 인용 포함
- 분량: 최소 10줄 이상

**# 2. 분석 대상 기술 현황**
{self_company} 중심으로 각 기술별 현황을 서술합니다.
- 각 기술별: {self_company}의 현재 기술 수준, 개발 방향, 시장 내 위치. [N] 인용 필수.
- 각 기술 섹션은 최소 5줄 이상
- 영업이익·수율·원가 등 미공개 수치는 "비공개" 명시 (단정 금지)

**# 3. 경쟁사 동향 분석**
기술별로 경쟁사를 비교합니다. [데이터 A]의 비교표를 **그대로 삽입**하세요.
- **1) HBM**
  - 1-1) [데이터 A]의 HBM 비교표 삽입 (★ 자사 행 포함)
  - 1-2) 주요 경쟁사별 세부 분석 (증거가 있는 경우만, 최소 3줄씩, [N] 인용 필수)
- **2) PIM** (동일 구조)
- **3) CXL** (동일 구조)
- 모든 경쟁사({competitors})가 본문에 최소 1회 언급되어야 함

**# 5. 전략적 시사점**

경쟁사 TRL 분석과 위협 매트릭스를 근거로 {self_company}가 취해야 할 전략을 아래 구조로 상세히 서술하세요.
각 제언은 반드시 3절 분석 결과(TRL 수치, 위협 등급)와 직접 연결된 논거를 제시하고, 가능한 경우 [N] 인용을 달 것.

**5-1. 단기 R&D 과제 (1-2년)**
- 최소 4개 과제, 각 과제마다:
  - 과제명 (굵게)
  - 배경: 왜 지금 해야 하는지 (경쟁사 TRL·위협 등급 언급)
  - 목표: 구체적 달성 지표 또는 산출물
  - 리스크: 실행 시 주요 장애 요인 1가지

**5-2. 중기 기술 포지셔닝 (3-5년)**
- 최소 3개 방향, 각 방향마다:
  - 방향명 (굵게)
  - 근거: 기술 트렌드 또는 경쟁사 동향 기반
  - 기대 효과: {self_company}가 얻을 경쟁 우위
  - 투자 우선순위: 높음 / 중간 / 낮음 + 이유

**5-3. 협력·파트너십 전략**
- 기술별로 협력이 유효한 외부 파트너 유형 (고객사, 장비사, 표준화 단체 등) 제시
- 각 기술당 최소 1개 협력 방향

전체 분량: 최소 30줄 이상

**# REFERENCE**
- [1] 제목, URL 또는 DOI, 접근일: {today}
- 본문에서 사용한 [N] 인용 번호와 1:1 대응

---

**Compliance 규칙**
- TRL 추정표에서 trl_label이 "추정"인 항목은 본문에 반드시 "추정" 또는 "(추정)" 표기
- TRL 추정표에서 trl_label이 "확정"인 항목은 그대로 서술 가능
- 수율·원가·미공개 로드맵은 단정 금지
- 증거 자료에 없는 내용을 창작하거나 수치를 임의로 생성하지 말 것 — 특히 시장 점유율(%)은 증거에 없는 한 절대 서술 금지
- 제품 출하/납품 실적에 대한 구체적 수치(업체명 + %)가 증거에 없는 한 서술 금지
- 섹션 4(장기 신기술)는 증거가 충분할 때만 작성, 없으면 생략

**Formatting 규칙**
- Markdown 형식 출력
- 기술 용어는 한국어(영문 병기): 예) 대역폭(Bandwidth)
- 경쟁사 비교표는 [데이터 A]를 그대로 사용 (★ 자사 행 포함)
- 논문 출처는 DOI 포함, 특허 출처는 특허번호 포함
"""


# ── 데이터 준비 헬퍼 ──────────────────────────────────────────────

def _build_prebuilt_tables(
    trl_table: list,
    threat_matrix: list,
    technologies: list,
    evidence_store: list | None = None,
    self_company: str = "SK Hynix",
) -> str:
    """TRL 추정표와 위협 매트릭스에서 기술별 Markdown 비교표를 사전 생성.
    - 자사(self_company)는 '★ 자사' 마커 + 위협 수준 '-' 로 표시
    - evidence_refs URL을 evidence_store 인덱스로 매핑해 근거 요약에 [N] 인용 추가
    """
    # URL → 1-indexed position 맵
    url_to_idx: dict[str, int] = {}
    if evidence_store:
        for i, item in enumerate(evidence_store):
            url = item.get("url", "")
            if url:
                url_to_idx[url] = i + 1

    # 기술별 threat 매핑: (company, technology) -> level
    threat_by_combo: dict[tuple[str, str], str] = {}
    for r in threat_matrix:
        company = r.get("company", "")
        tech = r.get("technology", "")
        threat_by_combo[(company, tech.upper())] = r.get("level", "-")
        threat_by_combo[(company, tech.lower())] = r.get("level", "-")

    tables = []

    for tech in technologies:
        all_rows = [r for r in trl_table if r.get("technology", "").upper() == tech.upper()
                    or tech.lower() in r.get("technology", "").lower()]

        # 자사 행과 경쟁사 행 분리 → 자사 먼저 표시
        sk_rows = [r for r in all_rows if r.get("company") == self_company]
        comp_rows = [r for r in all_rows if r.get("company") != self_company]

        lines = [f"### {tech} 비교표"]
        lines.append("| 회사 | TRL 구간 | 라벨 | 위협 수준 | 근거 요약 |")
        lines.append("|------|----------|------|----------|-----------|")

        for row in sk_rows + comp_rows:
            company = row.get("company", "")
            is_self = company == self_company
            company_label = f"★ {company} (자사)" if is_self else company

            trl_range = row.get("trl_range", "정보 부족")
            label = row.get("trl_label", "추정")
            threat = "-" if is_self else (
                threat_by_combo.get((company, tech.upper()))
                or threat_by_combo.get((company, tech.lower()))
                or "-"
            )

            # 근거 요약: 문장 단위로 자르되 최대 120자
            raw_rationale = row.get("rationale", "")
            if len(raw_rationale) > 120:
                cut = raw_rationale[:120]
                # 마지막 완성 문장 경계(마침표·이다·있다 등)에서 자름
                last_period = max(cut.rfind(". "), cut.rfind("다. "), cut.rfind("다."))
                rationale = (cut[:last_period + 1] if last_period > 40 else cut).rstrip()
            else:
                rationale = raw_rationale

            # evidence_refs → [N][N][N] 형식 인용 추가 (_renumber_citations가 인식하는 포맷)
            if url_to_idx:
                ref_nums = [
                    str(url_to_idx[u])
                    for u in row.get("evidence_refs", [])
                    if u in url_to_idx
                ][:3]
                if ref_nums:
                    rationale += " " + "".join(f"[{n}]" for n in ref_nums)

            lines.append(f"| {company_label} | {trl_range} | {label} | {threat} | {rationale} |")

        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def _build_evidence_summary(evidence_store: list, competitors: list, max_total: int = 50) -> str:
    """증거를 회사·기술별로 그룹핑하여 LLM에 전달"""
    by_topic: dict[str, list[tuple[int, dict]]] = {"일반": []}

    for i, item in enumerate(evidence_store):
        text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
        categorized = False
        for comp in competitors:
            if comp.lower() in text:
                key = comp
                by_topic.setdefault(key, [])
                by_topic[key].append((i, item))
                categorized = True
                break
        if not categorized:
            by_topic["일반"].append((i, item))

    lines = []
    shown = 0
    max_per_group = max(3, max_total // (len(competitors) + 1))

    for group in competitors + ["일반"]:
        items = by_topic.get(group, [])
        if not items:
            continue
        lines.append(f"\n--- {group} 관련 ({len(items)}건) ---")
        for idx, item in items[:max_per_group]:
            source_tag = ""
            st = item.get("source_type", "")
            if st == "paper":
                source_tag = " [논문]"
            elif st == "patent":
                source_tag = " [특허]"
            lines.append(
                f"[{idx + 1}]{source_tag} {item.get('title', '')} ({item.get('date', '')})\n"
                f"    {item.get('snippet', '')[:250]}\n"
                f"    출처: {item.get('url', '')}"
            )
            shown += 1
        if len(items) > max_per_group:
            lines.append(f"  ... 외 {len(items) - max_per_group}건")

    lines.insert(0, f"(전체 {len(evidence_store)}건 중 {shown}건 표시)")
    return "\n".join(lines)


def _build_evidence_summary_indexed(
    evidence_store: list,
    competitors: list,
    technologies: list,
    k_per_section: int = 10,
) -> str:
    """
    evidence_index 쿼리 기반 증거 요약.
    인덱스 미구축 시 기존 방식 폴백.
    """
    if not ev_idx.is_index_ready():
        return _build_evidence_summary(evidence_store, competitors)

    lines = []
    shown = 0

    # 배경 섹션 용 시장 동향 쿼리
    tech_str = " ".join(technologies)
    market_items = ev_idx.query_evidence(
        f"{tech_str} market demand trend 2025 2026",
        k=k_per_section,
    )
    if market_items:
        lines.append(f"\n--- 시장 동향 ({len(market_items)}건) ---")
        for item in market_items:
            idx = next((i for i, e in enumerate(evidence_store) if e.get("url") == item.get("url")), 0)
            st = item.get("source_type", "")
            source_tag = " [논문]" if st == "paper" else " [특허]" if st == "patent" else ""
            lines.append(
                f"[{idx + 1}]{source_tag} {item.get('title', '')} ({item.get('date', '')})\n"
                f"    {item.get('snippet', '')[:250]}\n"
                f"    출처: {item.get('url', '')}"
            )
            shown += 1

    # 경쟁사별 쿼리
    for comp in competitors:
        comp_items = ev_idx.query_evidence(
            f"{comp} {tech_str} technology strategy development",
            k=k_per_section,
        )
        if comp_items:
            lines.append(f"\n--- {comp} 관련 ({len(comp_items)}건) ---")
            for item in comp_items:
                idx = next((i for i, e in enumerate(evidence_store) if e.get("url") == item.get("url")), 0)
                st = item.get("source_type", "")
                source_tag = " [논문]" if st == "paper" else " [특허]" if st == "patent" else ""
                lines.append(
                    f"[{idx + 1}]{source_tag} {item.get('title', '')} ({item.get('date', '')})\n"
                    f"    {item.get('snippet', '')[:250]}\n"
                    f"    출처: {item.get('url', '')}"
                )
                shown += 1

    lines.insert(0, f"(전체 {len(evidence_store)}건 중 {shown}건 쿼리 기반 표시)")
    return "\n".join(lines)


def _strip_markdown_codeblock(text: str) -> str:
    """LLM이 응답을 ```markdown 코드블록으로 감쌌을 경우 제거"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```[^\n]*\n', '', stripped)
        stripped = re.sub(r'\n```\s*$', '', stripped)
    return stripped.strip()


def _renumber_citations(draft: str, evidence_store: list) -> tuple[str, list[ReferenceItem]]:
    """
    본문 [N] 인용 번호를 등장 순서대로 1부터 재할당.
    [1], [21], [76], [107] → [1], [2], [3], [4] 로 정리해 REFERENCE 갭 제거.
    """
    # 등장 순서대로 고유 ID 수집 (dict.fromkeys로 순서 유지)
    old_ids = list(dict.fromkeys(int(c) for c in re.findall(r'\[(\d+)\]', draft)))
    if not old_ids:
        return draft, []

    id_map = {old: new + 1 for new, old in enumerate(old_ids)}

    def replace_citation(m: re.Match) -> str:
        return f'[{id_map.get(int(m.group(1)), int(m.group(1)))}]'

    new_draft = re.sub(r'\[(\d+)\]', replace_citation, draft)

    refs: list[ReferenceItem] = []
    for old_id in old_ids:
        new_id = id_map[old_id]
        idx = old_id - 1
        if 0 <= idx < len(evidence_store):
            item = evidence_store[idx]
            refs.append(ReferenceItem(
                citation_id=f"[{new_id}]",
                url=item.get("url", ""),
                title=item.get("title", ""),
                accessed_date=TODAY,
            ))
        else:
            refs.append(ReferenceItem(
                citation_id=f"[{new_id}]",
                url="",
                title="[매핑 불가 — 원본 출처 확인 필요]",
                accessed_date=TODAY,
            ))
    return new_draft, refs


def _replace_reference_section(draft: str, reference_list: list[ReferenceItem]) -> str:
    """LLM이 생성한 REFERENCE 섹션을 시스템 reference_list로 교체"""
    if not reference_list:
        return draft

    # 시스템 기반 REFERENCE 섹션 생성
    ref_lines = ["# REFERENCE"]
    for ref in reference_list:
        cid = ref.get("citation_id", "")
        title = ref.get("title", "")
        url = ref.get("url", "")
        accessed = ref.get("accessed_date", TODAY)
        if "매핑 불가" in title:
            ref_lines.append(f"- {cid} [매핑 불가], 접근일: {accessed}")
        else:
            ref_lines.append(f"- {cid} {title}, {url}, 접근일: {accessed}")
    system_reference = "\n".join(ref_lines)

    # 기존 REFERENCE 섹션을 찾아 교체
    pattern = re.compile(
        r"(^|\n)(#+\s*REFERENCE\b.*)",
        re.DOTALL,
    )
    match = pattern.search(draft)
    if match:
        # REFERENCE 이후 전체를 교체
        draft = draft[:match.start(2)] + system_reference + "\n"
    else:
        # REFERENCE 섹션이 없으면 끝에 추가
        draft = draft.rstrip() + "\n\n" + system_reference + "\n"

    return draft


# ── 에이전트 ──────────────────────────────────────────────────────

def report_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=16384)

    scope = state.get("scope", {})
    trl_table = state.get("trl_table", [])
    threat_matrix = state.get("threat_matrix", [])
    evidence_store = state.get("evidence_store", [])
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    self_company = scope.get("self_company", "SK Hynix")

    existing_draft = state.get("draft_report", "")
    quality_feedback = state.get("last_error", "")
    is_retry = bool(existing_draft and quality_feedback)

    print(f"[Report] 보고서 {'재작성' if is_retry else '초안 생성'} 중...")

    prebuilt_tables = _build_prebuilt_tables(trl_table, threat_matrix, technologies, evidence_store, self_company)
    evidence_summary = _build_evidence_summary_indexed(evidence_store, competitors, technologies)

    prompt = REPORT_PROMPT.format(
        today=TODAY,
        self_company=self_company,
        technologies=", ".join(technologies),
        competitors=", ".join(competitors),
        prebuilt_tables=prebuilt_tables,
        threat_matrix=json.dumps(threat_matrix, ensure_ascii=False, indent=2),
        evidence_count=len(evidence_store),
        evidence_summary=evidence_summary,
    )

    if is_retry:
        # 이전 이슈에서 구체적 금지 수치(%, 숫자) 추출하여 명시
        forbidden_nums = re.findall(r'\d+[\.,]?\d*\s*%', quality_feedback)
        forbidden_note = ""
        if forbidden_nums:
            forbidden_note = (
                f"\n**[절대 금지 수치 — 아래 수치를 이 보고서에서 절대 사용하지 마세요]**\n"
                + "\n".join(f"- {n}: 증거 자료에 없는 환각 수치" for n in set(forbidden_nums))
                + "\n시장 점유율·판매량 등 구체적 수치는 증거에 있는 경우에만 인용 형태로 서술할 것\n"
            )
        prompt += (
            "\n\n**[품질 검수 피드백 — 아래 지적사항을 반드시 반영하여 보고서를 수정하세요]**\n"
            f"{quality_feedback}"
            f"{forbidden_note}"
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        draft = response.content
    except Exception as e:
        print(f"[Report] ⚠️ LLM 호출 실패: {e}")
        return {"draft_report": "", "reference_list": [], "last_error": str(e)}

    # 코드블록 감싸짐 제거
    draft = _strip_markdown_codeblock(draft)

    # 인용 번호 순차 재할당 ([1],[21],[107] → [1],[2],[3])
    draft, reference_list = _renumber_citations(draft, evidence_store)

    unmapped = [r for r in reference_list if "매핑 불가" in r["title"]]
    if unmapped:
        print(f"[Report] ⚠️ 매핑 불가 인용 {len(unmapped)}건: {[r['citation_id'] for r in unmapped]}")

    # LLM이 생성한 REFERENCE 섹션을 시스템 reference_list로 교체
    draft = _replace_reference_section(draft, reference_list)

    print(f"[Report] 보고서 생성 완료 (REFERENCE {len(reference_list)}건, 본문 약 {len(draft)}자)")

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/report_{TODAY}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(draft)
    print(f"[Report] 저장 완료: {output_path}")

    return {
        "draft_report": draft,
        "reference_list": reference_list,
    }
