# AI Agent Mini Project — 계획서

## 1. 프로젝트 개요


| 항목      | 내용                                                         |
| ------- | ---------------------------------------------------------- |
| **주제**  | HBM4, PIM(Processing-in-Memory), CXL 등 최신 반도체 R&D 정보 수집·분석 |
| **분석**  | 경쟁사별 기술 성숙도·위협 수준 비교                                       |
| **산출물** | R&D 담당자가 즉시 활용 가능한 **기술 전략 분석 보고서** (구조화된 문서)              |


---

## 2. A. 설계 — Agentic Workflow

### 2.1 Supervisor 역할

- **Supervisor 에이전트**: 목표·성공 기준·계획 승인, 하위 에이전트 조율, 품질 게이트(재시도·분기), 최종 보고서 구조·톤 검수.
- **실행 에이전트**: 검색·추출·비교·초안 작성 등 **Outcome 단위**로 위임; Supervisor는 “무엇을 완료해야 다음 단계로 갈 수 있는가”만 판단.

### 2.2 Goal


| Goal ID | Outcome                                                                |
| ------- | ---------------------------------------------------------------------- |
| G1      | 분석 대상 기술(HBM4/PIM/CXL)에 대해 **의사결정에 쓰일 수 있는** 최신 공개 정보가 체계적으로 수집·정리된 상태 |
| G2      | 경쟁사별로 **TRL 추정·위협 수준**이 동일한 기준으로 비교 가능한 형태로 정리된 상태                     |
| G3      | 위 정보를 바탕으로 **우선순위·대응 방향**을 논의할 수 있는 **기술 전략 분석 보고서**가 완성된 상태           |


### 2.3 Success Criteria

각 기준은 **대상 / 변화 / 결정·행동 / 범위**를 명시한다.  
각 SC는 아래 품질 프레임워크 3축을 기준으로 설계한다.

- **결과 품질 (Quality)**: 근거성(출처) · 완결성(누락 없음) · 일관성(내부 모순 없음) — TOOL로 자동 검증
- **업무적 제약 (Constraints)**: Target Actor · Compliance · Scope Boundaries · Formatting
- **서비스 기준 (Operational)**: Error Handling(Reflection) · Consistency · Stability — LOOP·Reflection 패턴으로 대응

#### G1 관련


| 기준 ID | 대상                | 변화                                                                            | 결정·행동                   | 범위(포함/제외)                                     |
| ----- | ----------------- | ----------------------------------------------------------------------------- | ----------------------- | --------------------------------------------- |
| SC1-1 | HBM4, PIM, CXL 각각 | 출처·날짜가 붙은 요약 카드 ≥ N건(기술당 5건 이상); 근거 없는 스니펫 포함 불가 **(근거성)**                    | 기술 현황 섹션 작성 시 근거로 인용 가능 | 포함: 논문·학회·특허·공식 보도·IR; 제외: 익명 커뮤니티 단독 주장      |
| SC1-2 | 수집 문서 메타데이터       | `keywords`, `entities` 필드 존재 여부로 자동 판정; 추출 불가 문서는 “태깅 불가” 표기 **(Reflection)** | 검색·재랭킹·중복 제거에 사용        | 포함: 모든 수집 스니펫; 제외: 텍스트 추출 불가 문서(이미지 전용 PDF 등) |


#### G2 관련


| 기준 ID | 대상                 | 변화                                                                   | 결정·행동           | 범위                                                                     |
| ----- | ------------------ | -------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------- |
| SC2-1 | 사전 정의 경쟁사×기술 전체 조합 | 모든 조합에 TRL 구간 + 근거 ≥ 2개; 근거 부족 시 “정보 부족” 명시 **(완결성·근거성·Reflection)** | 경쟁사 비교 표·서술에 사용 | TRL 4–6은 “추정” 라벨 및 간접 지표 명시; 근거 없으면 “정보 부족” 표기                         |
| SC2-2 | 동일 조합 전체           | 위협 수준(낮음/중간/높음) + TRL값과 모순 없는 판정 규칙 **(완결성·일관성)**                    | 전략 시사점 우선순위 논의  | 규칙은 README에 고정; TRL ≥ 7 → 높음, 5–6 → 중간, ≤ 4 → 낮음 (투자·채용 추세로 1단계 상향 가능) |


> **일관성 판정**: SC2-2의 위협 등급은 SC2-1의 TRL 수치와 교차 검증하여 모순 시 Supervisor가 Analysis 재실행 지시.

#### G3 관련


| 기준 ID | 대상          | 변화                                                                                                                                                           | 결정·행동            | 범위                                        |
| ----- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | ----------------------------------------- |
| SC3-1 | 최종 보고서      | SUMMARY 400자 이내 + 4절 목차 + REFERENCE; 모든 주장에 본문 [N] 인용 존재 **(근거성·완결성)**; TRL 4–6 구간 “추정” 한계 고지 필수 **(Compliance)**; 독자(R&D 담당자) 기준 객관·간결 톤 **(Target Actor)** | R&D 리뷰 미팅 자료로 제출 | 언어: 한국어(영문 병기); 회사명·단위 표기 통일; 수율·원가 단정 금지 |
| SC3-2 | REFERENCE   | 본문 [N] 인용과 1:1 대응 URL/DOI/제목/접근일; 매핑 불가 인용은 “[매핑 불가]” 명시 **(Reflection)**                                                                                    | 출처 검증 가능         | 포함: 본문 인용 자료 전체; 제외: 인용 없이 단순 참고한 자료      |
| SC3-3 | 보고서 재실행 안정성 | 동일 evidence_store 입력 시 섹션 구조·TRL 구간·위협 등급 변동 없음 (`temperature=0` 보장) **(Stability·Consistency)**                                                             | 품질 일관성 유지        | 허용: 표현 미세 변동; 불허: 섹션 구조·등급 변경             |


*(N=5, 경쟁사 목록, 위협 등급 규칙은 scope.yaml 및 README에 수치·정의로 고정)*

### 2.4 Plan — Task 분해


| Task ID | Outcome                                                                                            | Success Criteria 연결   | 의존성    |
| ------- | -------------------------------------------------------------------------------------------------- | --------------------- | ------ |
| T1      | 웹/문서에서 근거가 있는 스니펫이 저장·메타데이터화된 상태 (scope.yaml은 실행 전 수동 설정)                                          | SC1-1, SC1-2          | —      |
| T2      | 로컬/인덱스 기반 검색으로 질의별 관련 청크가 반환되는 상태; **임베딩·Retrieval 선정 평가(Hit Rate@K/MRR)는 T2 없이는 수행 불가하므로 사실상 필수** | SC1-2, Hit Rate@K/MRR | T1     |
| T3      | 기술별·회사별 TRL 추정표(추정 구간 명시) 초안                                                                       | SC2-1                 | T1     |
| T4      | 위협 수준 매트릭스 초안                                                                                      | SC2-2                 | T3     |
| T5      | 보고서 목차에 맞춘 통합 초안                                                                                   | SC3-1                 | T3, T4 |
| T6      | REFERENCE 정합성·SUMMARY 압축; 동일 입력 재실행 시 섹션 구조·등급 변동 없음 확인                                            | SC3-1, SC3-2, SC3-3   | T5     |


### 2.5 Control Strategy


| 구간         | 전략                         | 설명                                                                                                                  |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 실행 전       | **수동 설정**                  | `scope.yaml`에서 technologies, competitors, keywords, n_evidence_min 설정; 범위 과대(경쟁사 > 5개 또는 기술 > 3개) 시 `app.py`가 자동 축소 |
| T1–T2      | **Loop + Retry (최대 3회)**   | SC1-1 미달(출처 N건 미충족) 또는 중복률 > 50% 시 쿼리 재작성 후 재검색; 3회 초과 시 **Escalation**                                             |
| Escalation | **Abort + 인간 개입 요청**       | 최대 재시도 후 SC 미달 시 수집된 증거와 함께 진행 불가 사유를 Supervisor가 출력하고 중단                                                           |
| TRL·위협     | **Human-in-the-loop (선택)** | Supervisor가 SC2-1 근거 인용 수 < 2개인 경쟁사를 감지하면 T2로 회귀 (재시도 카운터에 합산)                                                      |
| 최종         | **Linear**                 | T6 → T7 순차 확정                                                                                                       |


**Supervisor SC 판정 메커니즘**: Supervisor는 LLM 판단이 아닌 **계측 가능한 수치**를 우선 사용한다.

- SC1-1: `evidence_store`의 출처별 레코드 수 카운트 ≥ N (자동 판정)
- SC1-2: 메타데이터 내 `keywords`, `entities` 필드 존재 여부 (자동 판정)
- SC2-1: TRL 표 내 각 경쟁사 행의 `evidence_count` ≥ 2 (자동 판정)
- SC2-2: 위협 매트릭스 내 판정 규칙 필드 존재 여부 (자동 판정)
- SC3-1/SC3-2: 목차 섹션 존재 + REFERENCE ↔ 본문 인용 매핑 검증 (자동 판정)
- **수치 판정 통과 후** 품질 검수(톤·논리) 단계에만 LLM 평가 사용

### 2.6 구조 (고수준)

```
[scope.yaml 수동 설정] → app.py 로드 → [Supervisor] ← 모든 에이전트가 결과를 반환하는 허브
                                             │
                    ├─► Retrieve Agent → chunks + 메타데이터 반환 → Supervisor
                    ├─► WebSearch Agent→ Tavily 결과 반환 → Supervisor
                    │       ↑ SC1 미달 시 Supervisor가 재지시 (최대 3회)
                    │       ↑ 3회 초과 시 Supervisor → Escalate
                    │
                    ├─► Analysis Agent → TRL 표 + 위협 매트릭스 반환 → Supervisor
                    │       ↑ SC2 미달 시 Supervisor가 Retrieve/WebSearch 재지시
                    │
                    ├─► Report Agent   → 초안 + REFERENCE 반환 → Supervisor
                    │       ↑ SC3 미달 시 Supervisor가 재작성 지시
                    │
                    └─► SC 전체 통과 → END
```

> **Supervisor 패턴 핵심**: 에이전트는 절대 다음 단계로 직접 이동하지 않는다. 반드시 Supervisor에게 결과를 반환하고, Supervisor가 SC 판정 후 다음 행동을 결정한다.

### 2.7 Retrieve 설계

#### Open-source Embedding 후보군 및 선정 기준


| 후보 예시                                             | 특징              | 선정 기준                      |
| ------------------------------------------------- | --------------- | -------------------------- |
| `sentence-transformers` 계열 (e.g. multilingual 모델) | 한·영 혼합 문서 적합 가능 | 한국어/영어 쿼리 재현율, 라이선스, 추론 비용 |
| BGE-m3 등 다국어 dense                                | 긴 문맥·다국어        | 동일 + 벡터 차원·메모리             |
| (선택) Korean-specific 임베딩                          | 국내 보도 비중이 클 때   | 도메인 벤치(소규모 golden set)     |


**선정 절차**: (1) golden 질의 세트 10–30개 작성 → (2) 후보별 임베딩·검색 파이프라인 동일 조건 비교 → (3) Hit Rate@K, MRR 기록 → **최종 모델·버전을 GitHub README에 반영**.

#### Retrieval technique 후보군 및 선정 기준


| 후보                            | 설명       | 선정 기준              |
| ----------------------------- | -------- | ------------------ |
| Dense (vector)                | 임베딩 유사도  | 의미적 매칭, 동의어        |
| BM25 / 키워드                    | 어휘 일치    | 제품명·표준명 정확도        |
| Hybrid (dense + BM25, RRF 등)  | 결합       | 종합 Hit Rate@K, MRR |
| (선택) Re-ranking cross-encoder | 상위 K 재정렬 | 정확도 vs 지연          |


**성능 기록**: 개발 완료 후 README에 **Hit Rate@K**, **MRR** 및 평가 세트 요약(질의 수, K값) 명시.

### 2.8 Web Search (Tavily) — 확증 편향 완화

- **다각도 쿼리**: 동일 주제에 대해 아래 쿼리 유형을 병렬 실행.

  | 유형       | 쿼리 템플릿 예시                          |
  | -------- | ---------------------------------- |
  | 현황       | `”{기술명} latest development {연도}”`  |
  | 한계·비판    | `”{기술명} limitations challenges”`   |
  | 경쟁·대안    | `”{기술명} vs {대안기술} comparison”`     |
  | 채용·투자 신호 | `”{기업명} {기술명} hiring investment”`  |
  | 반증       | `”{기술명} failed abandoned setback”` |

- **출처 다양성**: 단일 도메인 출처가 전체 인용의 40% 초과 시 Supervisor가 T2 재실행 트리거.
- **Tavily**: API 기반 웹 검색 도구로 구현; 검색 결과는 `{url, title, date, snippet, domain}` 메타데이터와 함께 `evidence_store`에 저장.

---

## 3. B. 기술 전략 분석 보고서

### 3.1 목차 (상세)

1. **SUMMARY**
  - 보고서 전체 핵심 요약 (1/2페이지 이내)
2. **1. 분석 배경**
  - 왜 지금 이 기술을 분석해야 하는가 (시장·공급망·표준·정책 등)
3. **2. 분석 대상 기술 현황**
  - HBM4 / PIM / CXL 각각: 현재 기술 수준, 개발 방향, 주요 병목
4. **3. 경쟁사 동향 분석**
  - 경쟁사별 기술 전략, 최신 움직임, TRL·위협 요약 표
5. **4. 전략적 시사점**
  - R&D 우선순위 관점 대응 방향 제언 (단기/중기 구분 가능)
6. **REFERENCE**
  - 본문 인용과 대응하는 자료 목록 (URL/DOI/제목/접근일)

### 3.2 전략 비교 — TRL 기반

- **TRL 1–9** 정의는 가이드와 동일하게 보고서 부속 또는 본문에 요약 인용.  
- **공개 정보 기반 추정**  
  - TRL 1–3: 논문·학회·특허 중심 근거.  
  - TRL 4–6: **정보 갭 큼** → 수치 단정 금지, “추정” 및 간접 지표(특허 출원 패턴, 학회 발표 빈도 변화, 채용 키워드 등) 명시.  
  - TRL 7–9: 양산·실적·샘플 공급 등 공개 자료 위주, 단 수율·원가는 비공개 전제 명시.
- **한계 고지**: 에이전트 성능과 무관하게 TRL 4–6은 추정 영역임을 보고서 본문에 **명시적으로** 기술.

### 3.3 SWOT vs TRL

- SWOT는 상대·주관적 비교에 유리.  
- TRL은 “현재 단계” 절대 위치 제공; 보고서에서는 **TRL을 주축**으로 표를 만들고, 필요 시 SWOT는 보조 섹션으로 제한.

---

## 4. 설계 산출물 (자유 양식 정리)

### 4.1 Workflow 요약


| 항목                   | 내용                                        |
| -------------------- | ----------------------------------------- |
| **Goal**             | G1–G3 (Outcome)                           |
| **Success Criteria** | SC1-* ~ SC3-* (판정 가능)                     |
| **Task**             | T1–T7 (Outcome 단위, SC 연결)                 |
| **Control Strategy** | 분기·루프·재시도·(선택) HITL                       |
| **Structure**        | Supervisor + Retrieve/Web/Analysis/Report |


### 4.2 Workflow → Agent 정의 (초안)


| Agent          | 책임       | 주요 State 입력     | 주요 State 출력        |
| -------------- | -------- | --------------- | ------------------ |
| **Supervisor** | 분기·승인·품질 | 현재 단계, SC 달성 여부 | 다음 Task, 재시도 플래그   |
| **Retrieve**   | T1, T2   | 쿼리, 인덱스         | `chunks[]` + 메타데이터 |
| **WebSearch**  | T1 (병렬)  | 쿼리 세트           | Tavily 결과 정규화      |
| **Analysis**   | T3, T4   | 근거 풀            | TRL 표, 위협 매트릭스     |
| **Report**     | T5, T6   | 분석 산출           | 최종 Markdown        |


### 4.3 State · Graph 흐름 (개념)

- **State 필드 정의**

  | 필드                | 타입         | 설명                                                                   |
  | ----------------- | ---------- | -------------------------------------------------------------------- |
  | `scope`           | object     | scope.yaml 로드값 (technologies, competitors, keywords, n_evidence_min) |
  | `evidence_store`  | list[dict] | 수집 스니펫 (url, title, date, snippet, domain, keywords, entities)       |
  | `iteration_count` | int        | 현재 T1 재시도 횟수 (최대 3)                                                  |
  | `sc_status`       | dict       | SC별 자동 판정 결과 (SC1-1 ~ SC3-2 → pass/fail/count)                       |
  | `trl_table`       | list[dict] | 경쟁사별 TRL 추정 (company, technology, trl_range, evidence_count, label)  |
  | `threat_matrix`   | list[dict] | 경쟁사별 위협 등급 (company, level, rationale)                               |
  | `draft_report`    | str        | 보고서 Markdown 초안                                                      |
  | `reference_list`  | list[dict] | 본문 인용 ↔ 출처 매핑 (citation_id, url, title, accessed_date)               |
  | `last_error`      | str        | null                                                                 |

- **Graph (LangGraph)**
  ```text
  scope.yaml 로드 (app.py) → Supervisor

  Supervisor → Retrieve Agent      : T1 문서 검색 지시
  Supervisor → WebSearch Agent     : T1 웹 검색 지시
  Retrieve Agent → Supervisor      : chunks + 메타데이터 반환
  WebSearch Agent → Supervisor     : Tavily 결과 반환

  # SC1 판정 (evidence_store count ≥ N?)
  Supervisor → Retrieve Agent      : SC1 미달 · 쿼리 재작성 후 재시도 (iteration_count < 3)
  Supervisor → WebSearch Agent     : SC1 미달 · 쿼리 재작성 후 재시도 (iteration_count < 3)
  Supervisor → Escalate            : retry 3회 초과 → 진행 불가 사유 출력 후 END

  Supervisor → Analysis Agent      : T3-4 TRL · 위협 분석 지시
  Analysis Agent → Supervisor      : TRL 추정표 · 위협 매트릭스 반환

  # SC2 판정 (evidence_count ≥ 2 per company?)
  Supervisor → Retrieve Agent      : SC2 미달 · 증거 보강 요청

  Supervisor → Report Agent        : T5-6 보고서 작성 지시
  Report Agent → Supervisor        : 초안 · REFERENCE 반환

  # SC3 판정 (목차 완성 + 인용 매핑 OK?)
  Supervisor → Report Agent        : SC3 미달 · 재작성 요청
  Supervisor → END                 : SC 전체 통과
  ```

*(실제 프레임워크명은 구현 단계에서 확정)*

---

## 5. 개발·운영 계획 (uv 기준)

1. `uv init` / `uv venv --python 3.11`로 프로젝트·가상환경 생성 (Python ≥ 3.11 필요).
2. 의존성: LLM SDK, (선택) LangGraph/LangChain, Tavily 클라이언트, 임베딩·벡터스토어 라이브러리 등 `pyproject.toml`에 명시.
3. `.env`에 API 키(Tavily 등); 샘플은 `.env.example`만 커밋.
4. **평가 자산**: golden 질의, relevance 판정 규칙, Hit Rate@K/MRR 스크립트 → `eval/` 또는 `tests/`에 배치.
5. **README**: 워크플로 요약, 최종 임베딩·Retrieval 선택 근거, **Hit Rate@K·MRR** 수치, 보고서 생성 방법.

---

## 6. 프롬프트 품질 기준

LLM 호출 시 아래 3가지 기준을 Analysis·Report 에이전트 프롬프트에 명시적으로 반영한다.

### 6.1 결과 품질 기준 (Quality)


| 기준      | 정의                | 구현                                                                       |
| ------- | ----------------- | ------------------------------------------------------------------------ |
| **근거성** | 주장마다 근거가 있는가 (출처) | 증거 자료에 없는 내용 창작 금지 명시; `evidence_refs`는 실제 URL만 허용; 근거 없으면 "정보 부족" 표기 강제 |
| **완결성** | 필수 항목을 빠짐없이 포함했는가 | 경쟁사×기술 전체 조합(N개) 커버 요구; 위협 매트릭스 모든 경쟁사 행 생성 강제                           |
| **일관성** | 내부 모순이 없는가        | 위협 등급이 TRL 수치와 모순되지 않도록 프롬프트에 교차 검증 규칙 명시                                |


### 6.2 업무적 제약 조건 (Constraints)


| 항목                   | 내용                                                                         |
| -------------------- | -------------------------------------------------------------------------- |
| **Target Actor**     | 독자: 반도체 R&D 담당자·기술 전략 임원 (고급 전문가); 톤: 객관·간결·논거 중심; SUMMARY 400자 이내         |
| **Compliance**       | TRL 4–6 "추정" 명시 필수; 수율·원가 단정 금지; 증거 없는 수치 생성 금지; 매핑 불가 인용 `[매핑 불가]` 명시적 표기 |
| **Scope Boundaries** | scope.yaml 기준 포함/제외 범위 고정                                                  |
| **Formatting**       | TRL·위협 출력은 JSON 배열; 보고서는 Markdown; 회사명·기술 용어 표기 통일                         |


### 6.3 서비스 기준 (Operational)


| 항목                 | 내용                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| **Error Handling** | JSON 파싱·필드 검증 실패 시 피드백 포함 최대 2회 재요청 (`_llm_call_with_retry`); LLM API 오류 try/except 처리 후 `last_error` 전달 |
| **Consistency**    | `temperature=0` 고정; 필수 필드 검증 (`_validate_trl_row`, `_validate_threat_row`)                               |
| **Stability**      | evidence_store 비어있을 때 빈 결과 반환 + 경고 출력으로 그래프 중단 방지                                                        |


---

## 7. 리스크·완화


| 리스크           | 완화                                                |
| ------------- | ------------------------------------------------- |
| TRL 4–6 과신 서술 | 추정 라벨·간접 지표·한계 절 필수; Compliance 규칙으로 프롬프트에 명시     |
| 확증 편향         | Tavily 다각도·반증 쿼리·출처 다양성 규칙                        |
| 환각 인용         | REFERENCE와 본문 교차 검증; 매핑 불가 인용 자동 감지·표기; 원문 스니펫 보관 |
| 범위 폭발         | T1에서 경쟁사·기술 수 상한 및 SC1-1 최소 건수만 조정                |
| JSON 파싱 실패    | `_llm_call_with_retry` 재시도 + 필드 검증으로 빈 결과 방지      |
| LLM API 오류    | try/except → `last_error` 기록 → Supervisor 에스컬레이션  |


---

*본 문서는 Mini Project 가이드에 따른 계획서이며, 구현 세부사항은 리포지토리 README 및 코드와 동기화하여 갱신한다.*