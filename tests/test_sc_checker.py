"""Unit tests for Success Criteria checker (no LLM, no API calls)."""
from rd_strategy_agent.utils.sc_checker import (
    check_sc1_1,
    check_sc1_2,
    check_sc2_1,
    check_sc2_2,
    check_sc3_1,
    check_sc3_2,
)
from rd_strategy_agent.state import AgentState


def make_state(**kwargs) -> AgentState:
    base: AgentState = {
        "scope": {
            "technologies": ["HBM4", "PIM", "CXL"],
            "competitors": ["Samsung", "SK Hynix"],
            "keywords": [],
            "n_evidence_min": 2,
            "max_competitors": 5,
        },
        "evidence_store": [],
        "iteration_count": 0,
        "sc_status": {},
        "trl_table": [],
        "threat_matrix": [],
        "draft_report": "",
        "reference_list": [],
        "report_retry_count": 0,
        "last_error": None,
        "next_task": None,
    }
    base.update(kwargs)
    return base


def make_evidence(tech: str, count: int) -> list[dict]:
    return [
        {
            "url": f"https://example.com/{tech}/{i}",
            "title": f"{tech} paper {i}",
            "date": "2024-01-01",
            "snippet": f"This is about {tech} technology.",
            "domain": "example.com",
            "keywords": [tech],
            "entities": ["Samsung"],
            "tagging_status": "ok",
        }
        for i in range(count)
    ]


class TestSC1_1:
    def test_pass(self):
        evidence = make_evidence("HBM4", 3) + make_evidence("PIM", 3) + make_evidence("CXL", 3)
        state = make_state(evidence_store=evidence)
        result, count = check_sc1_1(state)
        assert result == "pass"

    def test_fail_insufficient(self):
        evidence = make_evidence("HBM4", 1)
        state = make_state(evidence_store=evidence)
        result, _ = check_sc1_1(state)
        assert result == "fail"


class TestSC1_2:
    def test_pass(self):
        evidence = make_evidence("HBM4", 2)
        state = make_state(evidence_store=evidence)
        assert check_sc1_2(state) == "pass"

    def test_pass_tagging_unavailable(self):
        ev = make_evidence("HBM4", 1)
        ev[0]["keywords"] = []
        ev[0]["entities"] = []
        ev[0]["tagging_status"] = "tagging_unavailable"
        state = make_state(evidence_store=ev)
        assert check_sc1_2(state) == "pass"

    def test_fail_missing_metadata_field(self):
        ev = make_evidence("HBM4", 1)
        ev[0].pop("entities")
        state = make_state(evidence_store=ev)
        assert check_sc1_2(state) == "fail"


class TestSC2_1:
    def test_pass(self):
        trl_table = [
            {"company": "Samsung", "technology": "HBM4", "trl_range": "7", "evidence_count": 3, "label": "confirmed", "sources": ["[1]", "[2]", "[3]"]},
            {"company": "Samsung", "technology": "PIM", "trl_range": "5-6", "evidence_count": 2, "label": "estimated", "sources": ["[4]", "[5]"]},
            {"company": "Samsung", "technology": "CXL", "trl_range": "5-6", "evidence_count": 2, "label": "estimated", "sources": ["[6]", "[7]"]},
            {"company": "SK Hynix", "technology": "HBM4", "trl_range": "6", "evidence_count": 2, "label": "estimated", "sources": ["[8]", "[9]"]},
            {"company": "SK Hynix", "technology": "PIM", "trl_range": "6", "evidence_count": 2, "label": "estimated", "sources": ["[10]", "[11]"]},
            {"company": "SK Hynix", "technology": "CXL", "trl_range": "7", "evidence_count": 2, "label": "confirmed", "sources": ["[12]", "[13]"]},
        ]
        state = make_state(trl_table=trl_table)
        assert check_sc2_1(state) == "pass"

    def test_fail_missing_pair(self):
        trl_table = [
            {"company": "Samsung", "technology": "HBM4", "trl_range": "7", "evidence_count": 1, "label": "confirmed", "sources": ["[1]"]},
            {"company": "SK Hynix", "technology": "HBM4", "trl_range": "6", "evidence_count": 2, "label": "estimated", "sources": ["[4]", "[5]"]},
        ]
        state = make_state(trl_table=trl_table)
        assert check_sc2_1(state) == "fail"


class TestSC2_2:
    def test_pass(self):
        evidence = [
            {
                "url": "https://example.com/1",
                "title": "Samsung hiring for HBM4",
                "date": "2024-01-01",
                "snippet": "Samsung hiring around HBM4 and packaging.",
                "domain": "example.com",
                "keywords": ["HBM4"],
                "entities": ["Samsung"],
                "tagging_status": "ok",
            }
        ]
        trl_table = [
            {"company": "Samsung", "technology": "HBM4", "trl_range": "5-6", "evidence_count": 2, "label": "estimated", "sources": ["[1]", "[2]"]},
            {"company": "Samsung", "technology": "PIM", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "Samsung", "technology": "CXL", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "SK Hynix", "technology": "HBM4", "trl_range": "7", "evidence_count": 2, "label": "confirmed", "sources": ["[3]", "[4]"]},
            {"company": "SK Hynix", "technology": "PIM", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "SK Hynix", "technology": "CXL", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
        ]
        threat_matrix = [
            {"company": "Samsung", "level": "high", "rationale": "TRL 5-6 with hiring signal."},
            {"company": "SK Hynix", "level": "high", "rationale": "TRL 7 reached."},
        ]
        state = make_state(evidence_store=evidence, trl_table=trl_table, threat_matrix=threat_matrix)
        assert check_sc2_2(state) == "pass"

    def test_fail_on_trl_mismatch(self):
        trl_table = [
            {"company": "Samsung", "technology": "HBM4", "trl_range": "5-6", "evidence_count": 2, "label": "estimated", "sources": ["[1]", "[2]"]},
            {"company": "Samsung", "technology": "PIM", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "Samsung", "technology": "CXL", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "SK Hynix", "technology": "HBM4", "trl_range": "7", "evidence_count": 2, "label": "confirmed", "sources": ["[3]", "[4]"]},
            {"company": "SK Hynix", "technology": "PIM", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
            {"company": "SK Hynix", "technology": "CXL", "trl_range": "정보 부족", "evidence_count": 0, "label": "insufficient", "sources": []},
        ]
        threat_matrix = [
            {"company": "Samsung", "level": "low", "rationale": "Wrong level."},
            {"company": "SK Hynix", "level": "high", "rationale": "TRL 7 reached."},
        ]
        state = make_state(trl_table=trl_table, threat_matrix=threat_matrix)
        assert check_sc2_2(state) == "fail"


class TestSC3:
    VALID_REPORT = """## SUMMARY
짧은 요약 [1].

## 1. 분석 배경
Background [1].

## 2. 분석 대상 기술 현황
Technology status [1].

## 3. 경쟁사 동향 분석
Competitor analysis [2].

## 4. 전략적 시사점
Strategic implications.

## REFERENCE
[1] Title A. https://example.com/a. Accessed 2024-01-01.
[2] Title B. https://example.com/b. Accessed 2024-01-01.
"""

    def test_sc3_1_pass(self):
        assert check_sc3_1(self.VALID_REPORT) == "pass"

    def test_sc3_1_fail_missing_section(self):
        report = self.VALID_REPORT.replace("## SUMMARY\n", "")
        assert check_sc3_1(report) == "fail"

    def test_sc3_2_pass(self):
        refs = [
            {"citation_id": "[1]", "url": "https://example.com/a", "title": "A", "accessed_date": "2024-01-01"},
            {"citation_id": "[2]", "url": "https://example.com/b", "title": "B", "accessed_date": "2024-01-01"},
        ]
        assert check_sc3_2(self.VALID_REPORT, refs) == "pass"

    def test_sc3_2_fail_unmapped(self):
        refs = [{"citation_id": "[1]", "url": "", "title": "A", "accessed_date": ""}]
        result = check_sc3_2(self.VALID_REPORT, refs)
        assert result.startswith("fail")
