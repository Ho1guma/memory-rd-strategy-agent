"""
app.py — Supervisor 패턴 LangGraph 그래프 진입점

흐름:
  scope.yaml 로드 → [retrieve + web_search] → supervisor_after_retrieve
    → analysis → supervisor_after_analysis
    → report → supervisor_after_report
    → end (또는 escalate)

모든 에이전트는 Supervisor에게 결과를 반환하고,
Supervisor가 SC 판정 후 next 필드로 다음 노드를 결정합니다.
"""

import os
import yaml
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.retrieve import retrieve_agent
from agents.web_search import web_search_agent
from agents.analysis import analysis_agent
from agents.report import report_agent
from agents.supervisor import (
    supervisor_after_retrieve,
    supervisor_after_analysis,
    supervisor_after_report,
    escalate,
)

load_dotenv()


# ── scope.yaml 로드 ──────────────────────────────────────────────
REQUIRED_FIELDS = ["technologies", "competitors", "keywords", "n_evidence_min", "max_competitors"]

def load_scope(scope_path: str = None) -> dict:
    path = scope_path or os.environ.get("SCOPE_FILE", "scope.yaml")
    with open(path, "r", encoding="utf-8") as f:
        scope = yaml.safe_load(f)
    missing = [f for f in REQUIRED_FIELDS if f not in scope or not scope[f]]
    if missing:
        raise ValueError(f"scope.yaml 필수 필드 누락: {missing}")
    # 범위 과대 시 자동 축소
    if len(scope["competitors"]) > scope["max_competitors"]:
        scope["competitors"] = scope["competitors"][:scope["max_competitors"]]
    if len(scope["technologies"]) > 3:
        scope["technologies"] = scope["technologies"][:3]
    print(f"[App] 분석 범위 — 기술: {scope['technologies']}, 경쟁사: {scope['competitors']}")
    return scope


# ── 라우팅 함수 ──────────────────────────────────────────────────
def route_after_retrieve(state: AgentState) -> str:
    return state.get("next", "retrieve")


def route_after_analysis(state: AgentState) -> str:
    nxt = state.get("next", "report")
    return nxt if nxt in ("retrieve", "report", "analysis", "escalate") else "report"


def route_after_report(state: AgentState) -> str:
    nxt = state.get("next", "end")
    return nxt if nxt != "end" else END


# ── 병렬 retrieve + web_search 래퍼 ─────────────────────────────
def retrieve_and_web_search(state: AgentState) -> dict:
    """Retrieve + WebSearch를 순차 실행 (결과는 evidence_store에 누적)"""
    r1 = retrieve_agent(state)
    merged = {**state, "evidence_store": state.get("evidence_store", []) + r1.get("evidence_store", [])}
    r2 = web_search_agent(merged)
    return {
        "evidence_store": r1.get("evidence_store", []) + r2.get("evidence_store", [])
    }


# ── 그래프 구성 ──────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("retrieve", retrieve_and_web_search)
    graph.add_node("supervisor_retrieve", supervisor_after_retrieve)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("supervisor_analysis", supervisor_after_analysis)
    graph.add_node("report", report_agent)
    graph.add_node("supervisor_report", supervisor_after_report)
    graph.add_node("escalate", escalate)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "supervisor_retrieve")

    graph.add_conditional_edges(
        "supervisor_retrieve",
        route_after_retrieve,
        {
            "retrieve": "retrieve",
            "analysis": "analysis",
            "escalate": "escalate",
        },
    )

    graph.add_edge("analysis", "supervisor_analysis")

    graph.add_conditional_edges(
        "supervisor_analysis",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "analysis": "analysis",
            "report": "report",
            "escalate": "escalate",
        },
    )

    graph.add_edge("report", "supervisor_report")

    graph.add_conditional_edges(
        "supervisor_report",
        route_after_report,
        {
            "report": "report",
            "escalate": "escalate",
            END: END,
        },
    )

    graph.add_edge("escalate", END)

    return graph.compile()


# ── 실행 ─────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  Semiconductor R&D Intelligence Agent 시작")
    print("=" * 60 + "\n")

    scope = load_scope()
    app = build_graph()

    initial_state: AgentState = {
        "scope": scope,
        "evidence_store": [],
        "iteration_count": 0,
        "max_retry": int(os.environ.get("MAX_RETRY", 3)),
        "sc_status": {},
        "trl_table": [],
        "threat_matrix": [],
        "draft_report": "",
        "reference_list": [],
        "last_error": None,
        "next": "",
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state.get("draft_report"):
        print("✅ 보고서 생성 완료")
        sc = final_state.get("sc_status", {})
        print(f"  SC 판정: {sc}")
    else:
        print("⛔ 보고서 생성 실패 — outputs/ 및 last_error 확인")
        print(f"  last_error: {final_state.get('last_error')}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    run()
