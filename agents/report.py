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


# WRITER_MODEL: 보고서 작성 품질이 중요 → 고품질 모델 권장
# 우선순위: WRITER_MODEL > LLM_MODEL > 기본값(gpt-4o)
LLM_MODEL = os.environ.get("WRITER_MODEL") or os.environ.get("LLM_MODEL", "gpt-4o")
TODAY = date.today().isoformat()


# ── 프롬프트 ──────────────────────────────────────────────────────

REPORT_PROMPT = """당신은 {self_company} 내부 R&D 전략 담당자입니다.
이 보고서는 {self_company} 임원진과 R&D 팀장에게 배포되는 기밀 내부 문서입니다.
{self_company}의 관점에서, {self_company}가 무엇을 해야 하는지를 중심으로 작성합니다.

**독자·톤**
- 독자: 반도체 회로·공정·시스템 전문가 (배경 설명 불필요, 전문 용어 직접 사용)
- 톤: 사실 중심, 논거 명시, 감탄사·마케팅 표현 금지
- 1문장 최대 2줄 이내, 오늘 날짜: {today}

**분석 범위**
- 기술: {technologies}
- 경쟁사: {competitors}
- 자사: {self_company}

**=== 입력 데이터 ===**

**[데이터 A] TRL 비교표 (자사 ★ 포함 — 섹션 3에 그대로 삽입)**
{prebuilt_tables}

**[데이터 B] 위협 수준 매트릭스**
{threat_matrix}

**[데이터 C] 수집 증거 ({evidence_count}건) — 반드시 [N] 형식으로 인용**
{evidence_summary}

**=== 보고서 작성 지시 ===**

아래 섹션 번호·제목을 정확히 따르고, Markdown 헤딩(`#`, `##`)을 bold(`**`)로 감싸지 마세요.
모든 사실적 주장에는 [N] 인용을 필수로 달아야 합니다.

---

# {self_company} R&D 기술 전략 분석 보고서
## {technologies} 기술 경쟁력 및 대응 전략
**작성일**: {today} | **보안 등급**: 내부용 | **대상 독자**: {self_company} R&D 담당자 및 기술 전략 임원

---

# SUMMARY
핵심 내용을 아래 형식으로 작성하세요 (400자 이내):

- {self_company} 현재 포지션: TRL·양산 상태 핵심 수치 1-2줄
- 주요 경쟁사 위협: 각사 TRL·위협 등급 핵심 수치 (예: Samsung HBM4 TRL 7-8 높음)
- 시장·기술 변화 핵심 동인 1줄

**최우선 행동 권고안:**
> {self_company}가 향후 6-12개월 내 반드시 실행해야 할 최우선 과제를 굵게 1-2줄로 서술. TRL 수치와 위협 등급을 직접 연결할 것.

# 1. 분석 배경
- 왜 지금 {technologies}를 전략적으로 분석해야 하는가를 {self_company} 관점에서 서술
- 아래 항목을 순서대로 포함할 것:
  1. 분석 배경 및 목적 (2줄 이상)
  2. 시장 규모·수요 변화 — 증거에서 구체 수치(달러·성장률 등)를 인용 [N]
  3. 경쟁 압박 현황 — 주요 경쟁사({competitors}) 동향 요약 [N]
  4. 기술 발전 속도와 {self_company}의 전략적 위기감 (2줄 이상)
  5. 본 보고서의 분석 범위와 독자 활용 방법 (1줄)
- 수치는 증거에 있는 것만 인용, 임의 생성 금지
- 최소 [N] 인용 5개, 분량 최소 15줄

# 2. 기술별 개발 현황
각 기술에 대해 ## 2-N. 기술명 소제목으로 분리하세요.
각 소제목 아래 반드시 아래 네 파트를 포함합니다.

### 2-N-1. 기술 개요 및 성숙도
- 해당 기술의 핵심 원리, 현재 시장에서의 역할, 업계 전반 성숙도
- 현재 글로벌 TRL 범위(업계 평균)와 주요 표준화 현황 포함
- [N] 인용 최소 2개

### 2-N-2. 기업별 개발 현황
아래 기업 각각에 대해 번호 리스트로 서술합니다 (증거가 있는 기업만):
1. **{self_company} (자사)**: 현재 TRL, 주요 개발 마일스톤, 양산 또는 검증 상태 [N] 인용 필수
2. **Samsung**: TRL, 주요 동향 [N] 인용
3. **Micron**: TRL, 주요 동향 [N] 인용
4. **Intel**: TRL, 주요 동향 [N] 인용 (비메모리 제조사임을 명시)
- 각 기업 항목: 최소 2줄, [N] 인용 1개 이상

### 2-N-3. {self_company} 포지션 분석
- 경쟁사 대비 {self_company}의 강점(최소 2개)·약점(최소 2개) 항목별 서술
- 현재 TRL 갭: 선두 경쟁사와의 TRL 차이를 수치로 명시 (예: Samsung TRL 8-9 vs SK Hynix TRL X-Y → 갭 Z)
- {self_company}가 선도하는 영역과 추격이 필요한 영역 구분 서술
- 분량 최소 6줄, [N] 인용 최소 2개

### 2-N-4. 기술적 병목 및 도전 과제
- 해당 기술의 업계 공통 병목을 구체적으로 서술 (소재·공정·표준·생태계 각각 해당 사항 있으면 서술)
- {self_company}가 해결해야 할 핵심 기술 장벽 최소 2개 (각각 기술적 원인과 해결 방향 1줄씩)
- 분량 최소 5줄, [N] 인용 최소 1개

전체 2절 내 [N] 인용 최소 10개

**엄수**:
- 영업이익·수율·원가 등 미공개 수치는 "비공개" 명시
- 증거 없는 주장 금지

# 3. TRL 비교 분석
[데이터 A]의 "기업×기술 TRL 종합 비교표"를 먼저 삽입하세요.
그 아래에 기술별 상세 비교표를 순서대로 삽입하세요 (★ 자사 행 포함).
각 기술 비교표 아래에 주요 발견사항 2-3줄 요약 ([N] 인용 포함).
모든 경쟁사({competitors})가 최소 1회 이상 언급되어야 합니다.

## 3-1. 위협 매트릭스 해석
[데이터 A]의 "위협 매트릭스" 표를 삽입하고, 투자·채용 신호(investment_signal)가 있는 경쟁사를 강조하여 서술하세요.
위협 수준 "높음" 경쟁사의 단기 위협 시나리오를 1-2줄로 서술.

## 3-2. 주목할 스타트업 및 신흥 플레이어
아래 형식으로 표를 작성하세요 (증거에서 언급된 스타트업·신흥 기업이 있을 때만, 없으면 "확인된 스타트업 없음" 명시):

| 기업명 | 기술 분야 | 주요 활동 | 위협 수준 | 출처 |
|--------|----------|----------|----------|------|

# 4. 전략적 시사점
3절 분석 결과를 근거로 {self_company}가 취해야 할 전략을 시계열로 서술합니다.
각 제언은 TRL 수치·위협 등급과 직접 연결된 논거를 제시해야 합니다.

**엄수**: 4-1·4-2·4-3·4-4 전체에서 [N] 인용 총 10개 이상, 인용 없는 주장 금지

## 4-1. 단기 과제 (0~1년)
최소 3개 과제, 각 과제:
- **과제명**
- 배경: 경쟁사 TRL·위협 등급 언급 + [N] 인용
- 목표: 구체적 달성 지표 또는 산출물
- 리스크: 주요 장애 요인 1가지

## 4-2. 중기 기술 포지셔닝 (2~3년)
최소 3개 방향, 각 방향:
- **방향명**
- 근거: 기술 트렌드·경쟁사 동향 + [N] 인용
- 기대 효과: {self_company}가 얻을 경쟁 우위
- 투자 우선순위: 높음/중간/낮음 + 이유

## 4-3. 장기·신규 개념 (3~5년+)
{self_company} R&D가 3-5년 시계로 선점해야 할 차세대 기술 방향을 서술합니다.
아래 기술 컨셉 중 증거가 있는 것을 **최소 3개 이상** 포함하세요:
- CXL 3.0/4.0 Fabric 및 메모리 풀링 아키텍처
- Neuromorphic PIM / Analog In-Memory Computing
- 3D-stacked Compute Memory / Processing-in-Memory 집적
- 기타 증거 기반 차세대 기술 개념

**각 항목은 아래 구조로 최소 5줄 이상 서술할 것:**
- **기술 컨셉명**
- 현황·타당성: 현재 글로벌 TRL 수준, 주요 연구기관·기업 동향 + [N] 인용 1개 이상
- 핵심 기술 과제: 상용화까지 해결해야 할 기술 장벽 2개 이상
- {self_company} 선점 논리: 기존 보유 역량(HBM·PIM·CXL 실적)과의 구체적 연결점 + [N] 인용

전체 4-3 분량 최소 20줄, [N] 인용 최소 5개

## 4-4. 협력·파트너십 전략
- 기술별 유효한 외부 파트너 유형 (고객사, 장비사, 표준화 단체, 연구기관)
- 각 기술당 최소 1개 협력 방향 + [N] 인용 1개

전체 4절 분량: 최소 40줄

# REFERENCE
- [1] 제목, URL 또는 DOI, 접근일: {today}
- 본문에서 사용한 [N] 번호와 1:1 대응

---

**Compliance 규칙**
- trl_label이 "추정"인 항목은 본문에 반드시 "추정" 또는 "(추정)" 표기
- trl_label이 "확정"인 항목은 그대로 서술 가능
- 수율·원가·미공개 로드맵 단정 금지
- **수치 생성 절대 금지 (엄수)**: 시장 점유율(%)·매출액·판매량·목표 수치 등 구체적 숫자는 증거 원문에 명시된 경우에만 "[N]" 인용 형태로 서술. 증거 없는 수치는 서술하지 말 것.
  - 금지 예시: "시장 점유율 10% 확보", "매출 $X억 달성", "생산량 Y% 증가"
  - 허용 형식: "증거 [N]에 따르면 HBM 시장 규모는 약 $XX억 규모로 추정됨"
- 제품 출하/납품 실적 구체 수치는 증거 없으면 서술 금지
- 목표 지표 서술 시 "증거에 기반한 방향성"으로 서술: "양산 가속화", "시장 리더십 강화" 등 방향 표현 사용, 구체 수치는 증거가 있을 때만 인용

**Formatting 규칙**
- Markdown 헤딩: `#`, `##`, `###` 사용 — bold(`**`) 감싸기 금지
- 기술 용어: 한국어(영문 병기) 예) 대역폭(Bandwidth)
- 논문 출처: DOI 포함, 특허 출처: 특허번호 포함
"""


# ── 데이터 준비 헬퍼 ──────────────────────────────────────────────

def _build_prebuilt_tables(
    trl_table: list,
    threat_matrix: list,
    technologies: list,
    evidence_store: list | None = None,
    self_company: str = "SK Hynix",
) -> str:
    """TRL 추정표와 위협 매트릭스에서 Markdown 비교표를 사전 생성.
    - 자사(self_company)는 '★ 자사' 마커 + 위협 수준 '-' 로 표시
    - evidence_refs URL을 evidence_store 인덱스로 매핑해 근거 요약에 [N] 인용 추가
    - 기업×기술 종합 크로스 테이블(TRL + 종합 위협도) 포함
    - 위협 매트릭스에 투자·채용 신호(investment_signal) 열 포함
    """
    # URL → 1-indexed position 맵
    url_to_idx: dict[str, int] = {}
    if evidence_store:
        for i, item in enumerate(evidence_store):
            url = item.get("url", "")
            if url:
                url_to_idx[url] = i + 1

    # 기술별 threat 매핑: (company, technology) -> (level, investment_signal)
    threat_by_combo: dict[tuple[str, str], tuple[str, str]] = {}
    for r in threat_matrix:
        company = r.get("company", "")
        tech = r.get("technology", "")
        level = r.get("level", "-")
        signal = r.get("investment_signal", "-")
        threat_by_combo[(company, tech.upper())] = (level, signal)
        threat_by_combo[(company, tech.lower())] = (level, signal)

    # 모든 회사 목록 수집 (자사 + 경쟁사)
    all_companies_ordered: list[str] = []
    if self_company:
        all_companies_ordered.append(self_company)
    for r in trl_table:
        c = r.get("company", "")
        if c and c != self_company and c not in all_companies_ordered:
            all_companies_ordered.append(c)

    tables: list[str] = []

    # ── 1. 기업×기술 종합 크로스 테이블 ────────────────────────────
    cross_header = ["기업"] + technologies + ["종합 위협도"]
    cross_lines = ["### 기업×기술 TRL 종합 비교표"]
    cross_lines.append("| " + " | ".join(cross_header) + " |")
    cross_lines.append("|" + "|".join(["------"] * len(cross_header)) + "|")

    for company in all_companies_ordered:
        is_self = company == self_company
        company_label = f"★ {company} (자사)" if is_self else company

        row_cells = [company_label]
        threat_levels = []
        for tech in technologies:
            trl_row = next(
                (r for r in trl_table
                 if r.get("company") == company and (
                     r.get("technology", "").upper() == tech.upper()
                     or tech.lower() in r.get("technology", "").lower()
                 )),
                None,
            )
            if trl_row:
                trl_range = trl_row.get("trl_range", "-")
                label = trl_row.get("trl_label", "추정")
                label_mark = "" if label == "확정" else "(추정)"
                row_cells.append(f"TRL {trl_range} {label_mark}".strip())
            else:
                row_cells.append("-")

            # 위협 수준 (경쟁사만)
            if not is_self:
                combo = threat_by_combo.get((company, tech.upper())) or threat_by_combo.get((company, tech.lower()))
                if combo:
                    threat_levels.append(combo[0])

        # 종합 위협도: 경쟁사는 가장 높은 등급, 자사는 '-'
        if is_self:
            row_cells.append("-")
        elif threat_levels:
            priority = {"높음": 3, "중간": 2, "낮음": 1}
            overall = max(threat_levels, key=lambda x: priority.get(x, 0))
            row_cells.append(overall)
        else:
            row_cells.append("-")

        cross_lines.append("| " + " | ".join(row_cells) + " |")

    tables.append("\n".join(cross_lines))

    # ── 2. 위협 매트릭스 (투자·채용 신호 열 포함) ───────────────────
    if threat_matrix:
        threat_lines = ["### 위협 매트릭스 (경쟁사별 상세)"]
        threat_lines.append("| 회사 | 기술 | 위협 수준 | 판정 근거 | 투자·채용 신호 |")
        threat_lines.append("|------|------|----------|-----------|--------------|")
        for r in threat_matrix:
            company = r.get("company", "")
            tech = r.get("technology", "")
            level = r.get("level", "-")
            rationale = r.get("rationale", "")[:120]
            signal = r.get("investment_signal", "-")
            threat_lines.append(f"| {company} | {tech} | {level} | {rationale} | {signal} |")
        tables.append("\n".join(threat_lines))

    # ── 3. 기술별 상세 비교표 ────────────────────────────────────────
    for tech in technologies:
        all_rows = [r for r in trl_table if r.get("technology", "").upper() == tech.upper()
                    or tech.lower() in r.get("technology", "").lower()]

        # 자사 행과 경쟁사 행 분리 → 자사 먼저 표시
        sk_rows = [r for r in all_rows if r.get("company") == self_company]
        comp_rows = [r for r in all_rows if r.get("company") != self_company]

        lines = [f"### {tech} 상세 비교표"]
        lines.append("| 회사 | TRL 구간 | 라벨 | 위협 수준 | 근거 요약 |")
        lines.append("|------|----------|------|----------|-----------|")

        for row in sk_rows + comp_rows:
            company = row.get("company", "")
            is_self = company == self_company
            company_label = f"★ {company} (자사)" if is_self else company

            trl_range = row.get("trl_range", "정보 부족")
            label = row.get("trl_label", "추정")
            combo = (
                threat_by_combo.get((company, tech.upper()))
                or threat_by_combo.get((company, tech.lower()))
            )
            threat = "-" if is_self else (combo[0] if combo else "-")

            # 근거 요약: 문장 단위로 자르되 최대 200자
            raw_rationale = row.get("rationale", "")
            if len(raw_rationale) > 200:
                cut = raw_rationale[:200]
                # 마지막 완성 문장 경계(마침표·이다·있다 등)에서 자름
                last_period = max(cut.rfind(". "), cut.rfind("다. "), cut.rfind("다."))
                rationale = (cut[:last_period + 1] if last_period > 60 else cut).rstrip()
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


def _normalize_headings(draft: str) -> str:
    """LLM이 **# Heading** 또는 **# Heading** 형식으로 헤딩을 bold 처리할 때 정규화.
    예) **# SUMMARY** → # SUMMARY
        **# 1. 분석 배경** → # 1. 분석 배경
    이 변환이 없으면 supervisor regex가 헤딩을 찾지 못해 summary_len=-1,
    _replace_reference_section이 REFERENCE를 찾지 못해 섹션이 중복된다.
    """
    # 각 줄 시작의 **# ... ** 패턴 제거
    return re.sub(r'^\*{1,2}(#+\s)', r'\1', draft, flags=re.MULTILINE)


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

    # 재시도: 외과적 수정 프롬프트 (evidence_summary 생략 → 프롬프트 크기 대폭 축소)
    # 초안 생성: 전체 데이터 포함
    if not is_retry:
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
        # 재시도는 전면 재생성 대신 기존 초안 + 지적사항 수정 방식 사용
        # → 프롬프트 크기 대폭 축소, LLM 거부 위험 감소
        forbidden_nums = re.findall(r'\d+[\.,]?\d*\s*%', quality_feedback)
        forbidden_note = ""
        if forbidden_nums:
            forbidden_note = (
                "\n[주의] 아래 수치는 증거에 없는 임의 생성 수치입니다 — 사용 금지:\n"
                + "\n".join(f"- {n}" for n in set(forbidden_nums))
                + "\n"
            )
        prompt = (
            f"아래는 {self_company} R&D 기술 전략 보고서 초안입니다.\n"
            f"지적사항만 수정하여 보고서 전문(全文)을 다시 출력하세요.\n"
            f"섹션 구조·인용·Markdown 형식은 그대로 유지하세요.\n\n"
            f"[지적사항]\n{quality_feedback}{forbidden_note}\n\n"
            f"[수정 규칙]\n"
            f"- Markdown 헤딩 `#`, `##`, `###` — bold(**) 감싸기 금지\n"
            f"- 추정 TRL 항목: 반드시 '추정' 또는 '(추정)' 표기\n"
            f"- 증거에 없는 수치(시장 점유율 %·매출액 등) 서술 금지\n"
            f"- 모든 사실적 주장에 [N] 인용 유지\n\n"
            f"[초안]\n{existing_draft}"
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        draft = response.content
    except Exception as e:
        print(f"[Report] ⚠️ LLM 호출 실패: {e}")
        return {
            "draft_report": existing_draft,
            "reference_list": state.get("reference_list", []),
            "last_error": str(e),
        }

    # 짧은 응답 방어: LLM 거부/오류 메시지 감지 → 기존 초안 유지
    # 정상 보고서는 최소 500자 이상이어야 함
    if len(draft.strip()) < 500:
        print(f"[Report] ⚠️ LLM 응답이 너무 짧음 ({len(draft.strip())}자) — 기존 초안 유지")
        return {
            "draft_report": existing_draft,
            "reference_list": state.get("reference_list", []),
            "last_error": f"LLM 짧은 응답 ({len(draft.strip())}자): {draft.strip()[:80]}",
        }

    # 코드블록 감싸짐 제거
    draft = _strip_markdown_codeblock(draft)

    # **# Heading** → # Heading 정규화 (supervisor regex·REFERENCE 교체 오작동 방지)
    draft = _normalize_headings(draft)

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
