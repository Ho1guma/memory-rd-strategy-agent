# AGNETS.md

## 목적

이 문서는 `PROJECT_PLAN_REV.md`를 기준으로 본 저장소의 에이전트 역할, 입출력 계약, 성공 기준, 제어 전략을 빠르게 파악하기 위한 운영 문서다. 구현 변경 시 `README.md`, `scope.yaml`, `src/rd_strategy_agent/` 하위 코드와 함께 동기화해야 한다.

## 시스템 목표

- HBM4, PIM, CXL 관련 최신 공개 정보를 수집한다.
- 경쟁사별 기술 성숙도(TRL)와 위협 수준을 동일 기준으로 비교한다.
- R&D 담당자가 바로 사용할 수 있는 한국어 전략 보고서를 생성한다.

## 에이전트 구성

### 1. Supervisor

- 위치: `src/rd_strategy_agent/supervisor.py`
- 책임:
  - 전체 워크플로우 진입점
  - Success Criteria(SC) 판정
  - 재시도, 루프백, 중단 여부 결정
  - 하위 에이전트 간 직접 전이 금지
- 원칙:
  - 각 단계 결과는 반드시 Supervisor로 반환한다.
  - 다음 단계 이동은 Supervisor만 결정한다.

### 2. Scope Agent

- 위치: `src/rd_strategy_agent/agents/scope.py`
- 입력: `scope.yaml`
- 출력: `state["scope"]`
- 책임:
  - 실행 전 수동 작성된 범위를 로드한다.
  - 필수 필드 누락 시 즉시 실패시킨다.
- 필수 필드:
  - `technologies`
  - `competitors`
  - `keywords`
  - `n_evidence_min`
  - `max_competitors`
  - `threat_level_rules`

### 3. WebSearch Agent

- 위치: `src/rd_strategy_agent/agents/websearch.py`
- 책임:
  - Exa 다각도 검색 수행
  - OpenAlex 논문 검색 수행
  - 증거를 `EvidenceItem` 형식으로 정규화
  - `keywords`, `entities` 메타데이터 태깅
- 기본 검색 관점:
  - 최신 동향
  - 한계 및 비판
  - 경쟁 기술 비교
  - 채용 및 투자 신호
  - 실패 및 셋백

### 4. Retrieve Agent

- 위치: `src/rd_strategy_agent/agents/retrieve.py`
- 책임:
  - `evidence_store`를 dense + BM25 hybrid 검색 구조로 인덱싱
  - 추후 분석 및 평가용 검색 인터페이스 제공
- 평가 지표:
  - Hit Rate@K
  - MRR

### 5. Analysis Agent

- 위치: `src/rd_strategy_agent/agents/analysis.py`
- 책임:
  - 경쟁사 × 기술 조합별 TRL 범위 추정
  - 위협 수준 매트릭스 생성
- 규칙:
  - TRL 4-6은 반드시 `(추정)` 또는 동등한 추정 라벨을 사용한다.
  - 각 조합은 근거 2건 이상을 요구한다.
  - 위협 수준은 README의 고정 규칙과 일치해야 한다.

### 6. Report Agent

- 위치: `src/rd_strategy_agent/agents/report.py`
- 책임:
  - 최종 한국어 보고서 초안 작성
  - 본문 인용 `[N]`과 참고문헌 매핑 생성
- 필수 섹션:
  - `SUMMARY`
  - `1. 분석 배경`
  - `2. 분석 대상 기술 현황`
  - `3. 경쟁사 동향 분석`
  - `4. 전략적 시사점`
  - `REFERENCE`

## 상태 계약

공유 상태는 `src/rd_strategy_agent/state.py`의 `AgentState`를 따른다.

### 핵심 필드

- `scope`: 수동 설정 범위
- `evidence_store`: 수집된 증거 목록
- `iteration_count`: 재시도 횟수
- `sc_status`: SC 판정 결과
- `trl_table`: TRL 추정 결과
- `threat_matrix`: 위협 수준 결과
- `draft_report`: 최종 보고서 초안
- `reference_list`: 인용 매핑 결과
- `last_error`: 중단 사유

### EvidenceItem 최소 스키마

- `url`
- `title`
- `date`
- `snippet`
- `domain`
- `keywords`
- `entities`

## Success Criteria

### G1. 정보 수집

- SC1-1: 기술별 최소 증거 수를 충족해야 한다.
- SC1-2: 각 증거에 `keywords`, `entities`가 존재해야 한다.

### G2. 비교 분석

- SC2-1: 경쟁사 × 기술 조합마다 TRL 구간과 근거 2건 이상이 있어야 한다.
- SC2-2: 위협 수준은 `low`, `medium`, `high` 중 하나이며 TRL 규칙과 모순되면 안 된다.

### G3. 보고서

- SC3-1: 보고서는 필수 섹션과 본문 인용을 포함해야 한다.
- SC3-2: 본문 인용 `[N]`은 참고문헌과 1:1 대응해야 한다.
- SC3-3: 동일 입력에서 섹션 구조, TRL 구간, 위협 수준이 바뀌면 안 된다.

## 제어 전략

### 실행 전

- `scope.yaml`은 사람이 직접 작성한다.
- 범위가 과도하면 경고하거나 축소 정책을 적용한다.

### T1-T2

- SC1 실패 시 최대 3회 재시도한다.
- 중복률이 높거나 증거 수가 부족하면 쿼리를 보강한다.

### T3-T4

- TRL 근거 수 부족 시 검색 단계로 되돌린다.
- 이 루프백은 재시도 카운트에 포함한다.

### T5-T6

- 보고서 구조와 인용 정합성을 선형 검증한다.
- 동일 입력 재실행 안정성을 유지하기 위해 결정적 설정을 우선한다.

### Escalation

- 재시도 한도 초과 시 중단한다.
- 중단 시 현재 SC 상태, 누적 증거 수, 마지막 오류를 함께 남긴다.

## 구현 체크리스트

- `README.md`의 위협 등급 규칙과 `scope.yaml`이 일치하는가
- `Supervisor`가 SC 결과를 근거로만 라우팅하는가
- `evidence_store`가 재시도 시 누락 없이 누적되는가
- `Retrieve` 인덱스와 실제 분석 입력이 같은 증거 집합을 보는가
- `TRL` 결과가 경쟁사 × 기술 전체 조합을 덮는가
- `REFERENCE`가 본문 인용과 완전히 매핑되는가
- 테스트가 SC 해석과 실제 기획서 정의를 함께 검증하는가

## 변경 시 우선 동기화할 파일

- `PROJECT_PLAN_REV.md`
- `README.md`
- `scope.yaml`
- `src/rd_strategy_agent/supervisor.py`
- `src/rd_strategy_agent/utils/sc_checker.py`
- `src/rd_strategy_agent/agents/analysis.py`
- `src/rd_strategy_agent/agents/report.py`
