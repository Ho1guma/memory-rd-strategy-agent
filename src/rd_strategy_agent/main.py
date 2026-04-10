"""Entry point — run the R&D Strategy Agent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="R&D Strategy Agent")
    parser.add_argument(
        "--query",
        default="Analyze HBM4, PIM, CXL R&D landscape and competitor positioning for semiconductor strategy.",
        help="User query / analysis request",
    )
    parser.add_argument(
        "--output", default="report.md", help="Output file for the final report"
    )
    args = parser.parse_args()

    # Validate required env vars
    missing = [k for k in ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"] if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}\nCopy .env.example → .env and fill in keys.")

    from rd_strategy_agent.supervisor import build_graph
    from rd_strategy_agent.state import AgentState

    graph = build_graph().compile()

    initial_state: AgentState = {
        "scope": {},
        "evidence_store": [],
        "iteration_count": 0,
        "sc_status": {},
        "trl_table": [],
        "threat_matrix": [],
        "draft_report": "",
        "reference_list": [],
        "last_error": None,
        "next_task": args.query,
    }

    print(f"[Main] Starting R&D Strategy Agent workflow...")
    final_state = graph.invoke(initial_state)

    report = final_state.get("draft_report", "")
    if report:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[Main] Report saved → {args.output}")
    else:
        err = final_state.get("last_error", "Unknown error")
        print(f"[Main] No report generated. Last error: {err}")


if __name__ == "__main__":
    main()
