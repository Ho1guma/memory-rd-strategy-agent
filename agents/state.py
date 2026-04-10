from __future__ import annotations
from typing import Annotated, NotRequired, Optional, TypedDict
import operator


class EvidenceItem(TypedDict):
    url: str
    title: str
    date: str
    snippet: str
    domain: str
    keywords: list[str]
    entities: list[str]
    source_type: NotRequired[str]       # "web" | "paper" | "patent" | "local"
    doi: NotRequired[str]
    patent_number: NotRequired[str]
    assignee: NotRequired[str]
    publication_date: NotRequired[str]
    query_company: NotRequired[str]     # 이 증거를 수집할 때 사용한 경쟁사 쿼리 (회사명이 본문에 없어도 매핑용)


class TRLRow(TypedDict):
    company: str
    technology: str
    trl_range: str        # e.g. "4-6"
    trl_label: str        # "확정" | "추정"
    evidence_count: int
    evidence_refs: list[str]  # citation_id list
    rationale: str


class ThreatRow(TypedDict):
    company: str
    technology: str       # "HBM4" | "PIM" | "CXL" 등
    level: str            # "낮음" | "중간" | "높음"
    rationale: str


class ReferenceItem(TypedDict):
    citation_id: str      # e.g. "[1]"
    url: str
    title: str
    accessed_date: str


class SCStatus(TypedDict):
    sc1_1: str            # "pass" | "fail"
    sc1_1_count: int
    sc1_2: str
    sc2_1: str
    sc2_1_missing: list[str]  # companies with evidence_count < 2
    sc2_2: str
    sc3_1: str
    sc3_2: str

    # Phase 2 — 품질 검수 (규칙 기반 + LLM)
    sc2_consistency: NotRequired[str]
    sc2_consistency_issues: NotRequired[list[str]]
    sc2_refs_valid: NotRequired[str]
    sc2_refs_invalid: NotRequired[list[str]]
    sc3_summary_len: NotRequired[str]
    sc3_summary_actual_len: NotRequired[int]
    sc3_forbidden: NotRequired[str]
    sc3_forbidden_found: NotRequired[list[str]]
    sc3_citation_bounds: NotRequired[str]
    sc3_quality_review: NotRequired[str]
    sc3_quality_issues: NotRequired[list[dict]]


class AgentState(TypedDict):
    # Scope
    scope: dict                                    # technologies, competitors, keywords, n_evidence_min, max_competitors

    # Evidence collection
    evidence_store: Annotated[list[EvidenceItem], operator.add]

    # Loop control
    iteration_count: int
    max_retry: int

    # SC 판정 결과
    sc_status: SCStatus

    # Analysis outputs
    trl_table: list[TRLRow]
    threat_matrix: list[ThreatRow]

    # Report outputs
    draft_report: str
    reference_list: list[ReferenceItem]

    # Error tracking
    last_error: Optional[str]

    # Routing signal from supervisor
    next: str             # 다음 노드 이름 ("scope" | "retrieve" | "web_search" | "analysis" | "report" | "escalate" | "end")
