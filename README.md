# R&D Strategy Agent

HBM4 / PIM / CXL 최신 반도체 R&D 정보를 수집·분석하고, 경쟁사별 TRL·위협 수준을 비교하여 **기술 전략 분석 보고서**를 자동 생성하는 multi-agent 시스템.

## Architecture

```
User Query → Supervisor (LangGraph)
                 ├─► Scope Agent       (T1: scope.yaml 로드 — 수동 설정)
                 ├─► WebSearch Agent   (T2: Tavily 병렬 검색 + OpenAlex 논문 검색)
                 ├─► Retrieve Agent    (T2/T3: dense+BM25 hybrid 인덱싱)
                 ├─► Analysis Agent    (T4/T5: TRL 추정표 + 위협 매트릭스)
                 ├─► Report Agent      (T6/T7: 보고서 초안 + REFERENCE)
                 └─► QualityReview    (Phase 2: LLM 환각·인용 검수)
```

검색 소스:
- **Tavily**: 뉴스, 블로그, IR, 채용 신호 등 웹 전반 (다각도 쿼리, 병렬 실행)
- **OpenAlex**: 학술 논문 (기술당 최신 10건, `https://api.openalex.org`)
- 특허 DB 체계적 검색은 미포함 — Exa 웹 검색으로 간접 수집

Supervisor는 각 단계에서 **Success Criteria(SC)를 수치 기반으로 자동 판정**하고, 미달 시 재시도(최대 3회) 또는 Escalation(인간 개입 요청)을 수행합니다.

## Threat Rules

위협 등급 규칙은 `scope.yaml`의 `threat_level_rules`를 단일 원천으로 사용합니다.

- `TRL >= 7` → `high`
- `TRL 5-6` → `medium`
- `TRL <= 4` → `low`
- 채용 또는 투자 신호가 있으면 한 단계 상향

## Quickstart

```bash
# 1. 의존성 설치
uv sync

# 2. 환경 변수 설정
cp .env.example .env
# .env에 OPENAI_API_KEY, TAVILY_API_KEY 입력
# OPENALEX_API_KEY는 선택 — 없어도 무료 사용 가능

# 3. scope.yaml 설정 (실행 전 수동 설정 필수)
#    technologies, competitors, keywords, n_evidence_min 항목을 채운다

# 4. 분석 실행 (보고서 → report.md)
uv run rd-agent
```

> **참고**: 특허 DB 체계적 검색은 미포함. 특허 관련 정보는 Exa 웹 검색으로 간접 수집된다.

## Retrieval 평가

embedding 모델과 retrieval 기법 선택 근거를 아래에 기록합니다.

| 항목 | 선택 값 | 비고 |
|------|---------|------|
| Embedding model | `BAAI/bge-m3` | 한·영 혼합 문서, 다국어 dense |
| Retrieval | Hybrid (dense + BM25, RRF) | 제품명 정확도 + 의미 매칭 결합 |
| Hit Rate@5 | TBD (eval/evaluate.py 실행 후 기입) | 기준: ≥ 0.60 |
| MRR | TBD | 기준: ≥ 0.45 |
| 평가 쿼리 수 | 10 | eval/golden_queries.yaml |

```bash
# Retrieval 벤치마크 실행 (evidence 수집 후)
uv run python eval/evaluate.py --model BAAI/bge-m3 --hybrid
```

## Project Structure

```
.
├── src/rd_strategy_agent/
│   ├── main.py           # CLI entry point
│   ├── state.py          # Shared state schema (TypedDict)
│   ├── supervisor.py     # LangGraph graph + routing
│   ├── agents/
│   │   ├── scope.py      # T1: scope.yaml 확정
│   │   ├── websearch.py  # T2: Tavily 병렬 검색 + OpenAlex
│   │   ├── quality_review.py  # Phase 2: LLM 환각·인용 검수
│   │   ├── retrieve.py   # T2/T3: ChromaDB + BM25 hybrid
│   │   ├── analysis.py   # T4/T5: TRL 추정 + 위협 등급
│   │   └── report.py     # T6/T7: 보고서 작성
│   └── utils/
│       └── sc_checker.py # SC 자동 판정 (수치 기반)
├── eval/
│   ├── golden_queries.yaml  # 평가 질의 세트
│   └── evaluate.py          # Hit Rate@K / MRR 측정
├── tests/
│   └── test_sc_checker.py
├── scope.yaml            # T1 출력 템플릿 (실행 시 덮어씀)
├── .env.example
└── pyproject.toml
```

## Tests

```bash
uv run pytest tests/ -v
```

## TRL 한계 고지

TRL 4–6 구간은 **공개 정보 기반 추정** 영역입니다. 보고서 내 해당 항목은 "(추정)" 라벨과 간접 지표(특허 출원, 채용 키워드, 학회 발표 빈도)를 명시하며, 정확한 수율·원가 정보는 포함하지 않습니다. 위협 등급은 `scope.yaml`의 고정 규칙과 교차 검증됩니다.

## Contributors

| 기여자 | 브랜치 | 주요 기여 |
|--------|--------|-----------|
| **sumin** | `sumin` | WebSearch 병렬화 (AsyncTavilyClient + asyncio.gather), 보고서 품질 개선 (SK Hynix 관점·기업별 상세 섹션·반할루시네이션 규칙), PDF 출력 (weasyprint + Noto Sans KR), Reference 중복 제거 및 순차 번호 재부여, SC1-2·SC2-1 조건 완화 (파이프라인 unblock), 스타트업·차세대 개념 쿼리 추가, OpenAlex API key query param 방식 수정 |
| **geonwook** | `geonwook` | EvidenceItem에 `tagging_status` 필드 추가, TRLEntry에 `label`/`technology` 필드 추가, SC2-1 경쟁사×기술 쌍 커버리지 검증, 보고서 재시도 횟수 제한 (`MAX_REPORT_RETRIES=2`), `report_retry_count` state 필드 |
| **jongman** | `jongman_agent` | LLM 기반 품질 검수 모듈 (`quality_review.py`: 환각 탐지·인용 정확성·톤 검사), Retrieval 평가 프레임워크 (`eval/run_eval.py`, `eval/golden_queries.json`: Hit Rate@K / MRR) |
