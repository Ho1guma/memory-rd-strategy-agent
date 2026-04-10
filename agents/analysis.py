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


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
JSON_RETRY = 2  # JSON 파싱 실패 시 재시도 횟수


# ── 프롬프트 ─────────────────────────────────────────────────────

TRL_ANALYSIS_PROMPT = """당신은 반도체 기업의 시니어 기술 전략 분석가입니다.

아래 증거 자료를 바탕으로 각 경쟁사의 기술별 TRL(Technology Readiness Level, 1-9)을 추정하세요.

**분석 대상 — 아래 모든 조합({total_combinations}개)을 빠짐없이 커버할 것 (완결성)**
- 기술: {technologies}
- 경쟁사: {competitors}

**수집된 증거 ({evidence_count}건)**
{evidence_summary}

**TRL 추정 규칙**
- TRL 1-3: 논문/학회/특허 근거 → "확정" 라벨
- TRL 4-6: 핵심 영업 비밀 구간 → 반드시 "추정" 라벨, 간접 지표(특허 출원 패턴·학회 빈도·채용 키워드) 명시
- TRL 7-9: 양산/샘플 공급/실적 공시 근거 → "확정" 라벨 (수율·원가는 비공개 전제 명시)

**근거성 규칙 (엄수)**
- 증거 자료에 없는 내용을 추정하거나 지어내지 말 것
- 해당 조합의 증거가 없으면 trl_range를 "정보 부족"으로 표기하고 rationale에 이유 기술
- evidence_refs는 위 증거 자료의 실제 URL만 사용할 것

**출력 형식 — JSON 배열만 출력, 마크다운 코드블록·설명 텍스트 없이**
[
  {{
    "company": "회사명",
    "technology": "기술명",
    "trl_range": "추정 구간 (예: 7-8) 또는 정보 부족",
    "trl_label": "확정 또는 추정",
    "evidence_count": 정수,
    "evidence_refs": ["실제 출처 URL"],
    "rationale": "판단 근거 2-3줄"
  }}
]
"""

THREAT_ANALYSIS_PROMPT = """당신은 반도체 R&D 전략 분석가입니다.

아래 TRL 추정표를 바탕으로 경쟁사별 위협 수준을 평가하세요.

**TRL 추정표**
{trl_summary}

**위협 등급 판정 기준 (일관성 — TRL값과 반드시 일치할 것)**
- 높음: TRL ≥ 7 (또는 TRL 5-6이라도 투자·채용 증가 추세 확인 시)
- 중간: TRL 5-6
- 낮음: TRL ≤ 4 또는 정보 부족

**완결성 규칙**: 아래 경쟁사 목록 전체에 대해 행을 생성할 것
대상 경쟁사: {competitors}

**일관성 검증**: TRL 추정표의 trl_range 값과 level이 모순되지 않아야 함

**출력 형식 — JSON 배열만 출력, 마크다운 코드블록·설명 텍스트 없이**
[
  {{
    "company": "회사명",
    "level": "높음 또는 중간 또는 낮음",
    "rationale": "위협 판정 근거 1-2줄 (TRL 수치 언급 포함)"
  }}
]
"""


# ── 헬퍼 함수 ────────────────────────────────────────────────────

def _build_evidence_summary(evidence_store: list, max_items: int = 30) -> str:
    lines = []
    for i, item in enumerate(evidence_store[:max_items]):
        lines.append(
            f"[{i+1}] {item['title']} ({item['date']})\n"
            f"    {item['snippet'][:200]}\n"
            f"    출처: {item['url']}"
        )
    return "\n\n".join(lines)


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
    return {"company", "level", "rationale"}.issubset(row.keys())


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
            # 파싱/검증 실패 시 LLM에 피드백 포함 재시도
            print(f"[Analysis] {label} JSON 검증 실패 (시도 {attempt}) — 재요청")
            messages.append(response)  # assistant 응답 포함
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

def analysis_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    scope = state.get("scope", {})
    evidence_store = state.get("evidence_store", [])
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])

    # Operational: evidence_store 비어있을 때 명시적 가드
    if not evidence_store:
        print("[Analysis] ⚠️ evidence_store 비어있음 — 빈 결과 반환")
        return {"trl_table": [], "threat_matrix": []}

    evidence_summary = _build_evidence_summary(evidence_store)
    total_combinations = len(technologies) * len(competitors)

    # T4: TRL 추정표
    print(f"[Analysis] TRL 추정표 생성 중 (증거 {len(evidence_store)}건, 조합 {total_combinations}개)...")
    trl_prompt = TRL_ANALYSIS_PROMPT.format(
        technologies=", ".join(technologies),
        competitors=", ".join(competitors),
        total_combinations=total_combinations,
        evidence_count=len(evidence_store),
        evidence_summary=evidence_summary,
    )
    trl_table: list[TRLRow] = _llm_call_with_retry(llm, trl_prompt, _validate_trl_row, "TRL")
    print(f"[Analysis] TRL 행 {len(trl_table)}개 생성 (기대: {total_combinations}개)")

    # T5: 위협 수준 매트릭스
    print("[Analysis] 위협 수준 매트릭스 생성 중...")
    trl_summary = json.dumps(trl_table, ensure_ascii=False, indent=2)
    threat_prompt = THREAT_ANALYSIS_PROMPT.format(
        trl_summary=trl_summary,
        competitors=", ".join(competitors),
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
