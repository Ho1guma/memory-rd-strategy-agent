# To Solve

`PROJECT_PLAN_REV.md` 기준으로 현재 구현과 어긋나는 항목 중, 반드시 먼저 수정해야 하는 순서를 정리한다.

## 1. SC 규칙 원천 단일화

- 문제:
  - TRL 기반 위협 등급 규칙이 `PROJECT_PLAN_REV.md`, `scope.yaml`, `README.md`, `analysis.py` 사이에서 다르다.
  - 현재 상태로는 같은 입력에도 서로 다른 threat level이 나올 수 있다.
- 수정 대상:
  - `scope.yaml`
  - `README.md`
  - `src/rd_strategy_agent/agents/analysis.py`
  - `src/rd_strategy_agent/utils/sc_checker.py`
- 완료 기준:
  - 위협 등급 규칙이 한 군데에서 정의되고 나머지는 이를 참조한다.
  - `SC2-2` 판정이 해당 규칙과 정확히 일치한다.

## 2. SC2-1을 경쟁사 × 기술 전체 조합 기준으로 수정

- 문제:
  - 현재 `SC2-1`은 회사별 행 존재만 확인한다.
  - 기획서는 모든 경쟁사 × 기술 조합마다 TRL과 근거 2건 이상을 요구한다.
- 수정 대상:
  - `src/rd_strategy_agent/utils/sc_checker.py`
  - `src/rd_strategy_agent/agents/analysis.py`
  - `tests/test_sc_checker.py`
- 완료 기준:
  - `scope.technologies × scope.competitors` 전체 조합을 검사한다.
  - 누락 조합, 근거 2건 미만, 라벨 누락이 모두 실패 처리된다.

## 3. SC2-2 교차 검증 구현

- 문제:
  - 현재는 threat entry에 `rationale`만 있으면 통과한다.
  - 기획서의 핵심 요구인 “TRL과 threat level 모순 시 재실행”이 구현되지 않았다.
- 수정 대상:
  - `src/rd_strategy_agent/utils/sc_checker.py`
  - `src/rd_strategy_agent/supervisor.py`
  - 필요 시 `src/rd_strategy_agent/state.py`
- 완료 기준:
  - 각 경쟁사의 TRL 결과와 threat level을 교차 검증한다.
  - 규칙 위반 시 `SC2_2=fail`이 되고 Supervisor가 analysis 또는 evidence 단계로 루프백한다.

## 4. SC3 무한 재작성 루프 제거

- 문제:
  - 보고서/참고문헌 정합성이 맞지 않으면 현재 그래프는 종료 조건 없이 `report`를 반복 호출한다.
  - 이 상태는 실패를 감추고 실행을 멈추지 못하는 구조다.
- 수정 대상:
  - `src/rd_strategy_agent/supervisor.py`
  - 필요 시 `src/rd_strategy_agent/state.py`
- 완료 기준:
  - `SC3` 전용 재시도 상한이 생긴다.
  - 한도 초과 시 명시적으로 escalation 또는 실패 종료한다.

## 5. SC1-2 / SC3-2를 기획서 정의에 맞게 보정

- 문제:
  - `SC1-2`는 현재 빈 `keywords`, `entities`를 모두 실패 처리해 기획서의 “필드 존재 여부”와 “태깅 불가 예외”를 반영하지 못한다.
  - `SC3-2`는 인용 ID 존재만 보고 있어 URL, 제목, 접근일 1:1 대응을 검증하지 못한다.
- 수정 대상:
  - `src/rd_strategy_agent/state.py`
  - `src/rd_strategy_agent/agents/websearch.py`
  - `src/rd_strategy_agent/agents/report.py`
  - `src/rd_strategy_agent/utils/sc_checker.py`
  - `tests/test_sc_checker.py`
- 완료 기준:
  - 증거 메타데이터에 `태깅 불가` 또는 동등한 상태 표현이 포함된다.
  - 참고문헌 검증이 `citation_id`, `url`, `title`, `accessed_date`까지 확인한다.
  - 본문 인용과 참고문헌 간 매핑 실패를 실제로 잡아낸다.

## 메모

- 위 1~5번을 끝내기 전에는 retrieval 성능 개선이나 리포트 스타일 튜닝보다, 판정 로직과 제어 흐름부터 고치는 쪽이 맞다.
- 특히 1~4번은 시스템이 “기획서대로 동작한다”고 말하기 위한 최소 조건이다.
