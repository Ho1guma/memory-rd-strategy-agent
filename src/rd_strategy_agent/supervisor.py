"""Supervisor — orchestrates the full workflow via LangGraph.

Control strategy (per PROJECT_PLAN.md §2.5):
- T1 → Conditional branch: scope explosion guard
- T2–T3 → Loop + Retry (max 3): SC1 failure triggers query rewrite
- Escalation: abort + human intervention after 3 retries
- T4–T5 → SC2 miss → re-route to T2 (counts toward retry)
- T6–T7 → Linear
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown as md_lib
from weasyprint import HTML, CSS
from langgraph.graph import StateGraph, END

from rd_strategy_agent.state import AgentState

_PDF_CSS = CSS(string="""
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
body {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-size: 11pt; line-height: 1.7; margin: 2.5cm 2cm; color: #1a1a1a;
}
h2 { font-size: 14pt; border-bottom: 1.5px solid #333; padding-bottom: 4px; font-weight: 700; }
h3 { font-size: 12pt; font-weight: 700; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #aaa; padding: 6px 10px; text-align: left; }
th { background-color: #f0f0f0; font-weight: 700; }
tr:nth-child(even) td { background-color: #fafafa; }
a { color: #1a56db; }
""")
from rd_strategy_agent.agents.scope import scope_agent
from rd_strategy_agent.agents.websearch import websearch_agent
from rd_strategy_agent.agents.retrieve import retrieve_index
from rd_strategy_agent.agents.analysis import analysis_agent
from rd_strategy_agent.agents.report import report_agent
from rd_strategy_agent import utils
from rd_strategy_agent.utils import sc_checker

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def node_scope(state: AgentState) -> dict:
    return scope_agent(state)


def node_evidence_gather(state: AgentState) -> dict:
    """Run WebSearch + Retrieve in sequence (parallel via LangGraph send possible)."""
    ws_update = websearch_agent(state)
    # Merge new evidence into state before indexing
    merged = {**state, "evidence_store": state.get("evidence_store", []) + ws_update.get("evidence_store", [])}
    retrieve_index(merged)
    return ws_update


def node_sc1_check(state: AgentState) -> dict:
    sc = sc_checker.run_all(state)
    return {"sc_status": sc}


def node_analysis(state: AgentState) -> dict:
    return analysis_agent(state)


def node_sc2_check(state: AgentState) -> dict:
    sc = sc_checker.run_all(state)
    return {"sc_status": sc}


def node_report(state: AgentState) -> dict:
    return report_agent(state)


def node_sc3_check(state: AgentState) -> dict:
    sc = sc_checker.run_all(state)
    return {"sc_status": sc}


def node_pdf_export(state: AgentState) -> dict:
    draft = state.get("draft_report", "")
    draft = re.sub(r"^```(?:markdown)?\n?", "", draft.strip())
    draft = re.sub(r"\n?```$", "", draft).strip()
    html_body = md_lib.markdown(draft, extensions=["tables", "fenced_code", "toc"])
    html = f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"></head><body>{html_body}</body></html>'
    out = Path("report.pdf")
    HTML(string=html, base_url=".").write_pdf(out, stylesheets=[_PDF_CSS])
    print(f"[PDFExport] PDF saved → {out}")
    return {}


def node_escalate(state: AgentState) -> dict:
    msg = (
        f"[ESCALATION] SC not met after {MAX_RETRIES} retries.\n"
        f"SC Status: {state.get('sc_status')}\n"
        f"Last error: {state.get('last_error')}\n"
        f"Evidence collected: {len(state.get('evidence_store', []))} items.\n"
        "Human review required."
    )
    print(msg)
    return {"last_error": msg}


# ---------------------------------------------------------------------------
# Routing functions (Supervisor decision logic)
# ---------------------------------------------------------------------------

def route_after_scope(state: AgentState) -> str:
    # scope.yaml 로드 실패 시 즉시 종료
    if state.get("last_error"):
        return "escalate"

    scope = state.get("scope", {})
    competitors = scope.get("competitors", [])
    technologies = scope.get("technologies", [])
    max_comp = scope.get("max_competitors", 5)

    if len(competitors) > max_comp or len(technologies) > 3:
        # scope는 수동 설정이므로 루프백 없이 경고만 출력하고 진행
        print(
            f"[Supervisor] WARNING: scope exceeds limits "
            f"({len(competitors)} competitors, {len(technologies)} technologies). "
            "Proceeding — reduce scope.yaml manually if needed."
        )
    return "evidence_gather"


def route_after_sc1(state: AgentState) -> str:
    sc = state.get("sc_status", {})
    iteration = state.get("iteration_count", 0)

    if sc.get("SC1_1") == "pass" and sc.get("SC1_2") == "pass":
        return "analysis"
    if iteration >= MAX_RETRIES:
        return "escalate"
    return "evidence_gather_retry"


def route_after_sc2(state: AgentState) -> str:
    sc = state.get("sc_status", {})
    iteration = state.get("iteration_count", 0)

    if sc.get("SC2_1") == "pass" and sc.get("SC2_2") == "pass":
        return "report"
    if iteration >= MAX_RETRIES:
        return "escalate"
    return "evidence_gather_retry"


def route_after_sc3(state: AgentState) -> str:
    sc = state.get("sc_status", {})
    if sc.get("SC3_1") == "pass" and sc.get("SC3_2") == "pass":
        return "pdf_export"
    return "report"  # Re-draft (counted separately, no retry limit here)


def increment_retry(state: AgentState) -> dict:
    return {"iteration_count": state.get("iteration_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("scope", node_scope)
    g.add_node("evidence_gather", node_evidence_gather)
    g.add_node("evidence_gather_retry", lambda s: {**increment_retry(s), **node_evidence_gather(s)})
    g.add_node("sc1_check", node_sc1_check)
    g.add_node("analysis", node_analysis)
    g.add_node("sc2_check", node_sc2_check)
    g.add_node("report", node_report)
    g.add_node("sc3_check", node_sc3_check)
    g.add_node("pdf_export", node_pdf_export)
    g.add_node("escalate", node_escalate)

    g.set_entry_point("scope")

    g.add_conditional_edges("scope", route_after_scope, {
        "evidence_gather": "evidence_gather",
        "escalate": "escalate",
    })
    g.add_edge("evidence_gather", "sc1_check")
    g.add_edge("evidence_gather_retry", "sc1_check")
    g.add_conditional_edges("sc1_check", route_after_sc1, {
        "analysis": "analysis",
        "evidence_gather_retry": "evidence_gather_retry",
        "escalate": "escalate",
    })
    g.add_edge("analysis", "sc2_check")
    g.add_conditional_edges("sc2_check", route_after_sc2, {
        "report": "report",
        "evidence_gather_retry": "evidence_gather_retry",
        "escalate": "escalate",
    })
    g.add_edge("report", "sc3_check")
    g.add_conditional_edges("sc3_check", route_after_sc3, {
        "pdf_export": "pdf_export",
        "report": "report",
    })
    g.add_edge("pdf_export", END)
    g.add_edge("escalate", END)

    return g
