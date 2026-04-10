"""Shared state schema for the R&D Strategy Agent workflow."""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
import operator


class EvidenceItem(TypedDict):
    url: str
    title: str
    date: str
    snippet: str
    domain: str
    keywords: list[str]
    entities: list[str]


class TRLEntry(TypedDict):
    company: str
    technology: str
    trl_range: str        # e.g., "5-6"
    evidence_count: int
    label: str            # "confirmed" | "estimated"
    sources: list[str]    # citation_ids


class ThreatEntry(TypedDict):
    company: str
    level: str            # "low" | "medium" | "high"
    rationale: str


class ReferenceItem(TypedDict):
    citation_id: str      # e.g., "[1]"
    url: str
    title: str
    accessed_date: str


class SCStatus(TypedDict, total=False):
    SC1_1: str            # "pass" | "fail" | count str
    SC1_2: str
    SC2_1: str
    SC2_2: str
    SC3_1: str
    SC3_2: str
    SC3_3: str


class AgentState(TypedDict):
    # G1 — scope
    scope: dict                                                    # scope.yaml contents
    # G1 — evidence
    evidence_store: Annotated[list[EvidenceItem], operator.add]
    iteration_count: int                                           # T2 retry counter (max 3)
    sc_status: SCStatus
    # G2 — analysis
    trl_table: list[TRLEntry]
    threat_matrix: list[ThreatEntry]
    # G3 — report
    draft_report: str
    reference_list: list[ReferenceItem]
    # Control
    last_error: str | None
    next_task: str | None                                          # Supervisor routing signal
