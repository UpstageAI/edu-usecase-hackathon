# Poisoned RAG — 연합 해커톤 테크 트랙

AIKU · YAI · KUBIG · YBIGTA 연합 해커톤 테크 트랙 제출물.
**Enron 이메일 코퍼스 기반 Multi-hop QA** 환경에서 Data Poisoning 및 Prompt Injection 공격을 방어하는 RAG 파이프라인.

---

## 시스템 구성

```
[PDF 코퍼스] → build_index → [Dense + BM25 인덱스]
                                       ↓
[암호화 질문셋] → load_test_suite → retrieve → generate_answer → submission.csv
                                                      ↑
                                              UpstageTracker (Solar LLM)
```

---

## 핵심 모듈

### 1. `upstage_tracker.py` — Solar LLM 호출 추적기

대회 규정상 최종 답변 생성은 반드시 **Upstage Solar LLM**을 통해 이루어져야 하며, `used_tokens = 0`인 제출은 실격 처리된다. 이 모듈이 그 요구사항을 충족시킨다.

| 기능 | 설명 |
|------|------|
| `tracker.chat()` | Solar API(`solar-mini`/`solar-pro`) 호출 + 자동으로 `question_id`, `answer`, `used_tokens`, `inference_time`, `token` 기록 |
| `tracker.save_csv()` | 누적 기록을 `submission.csv`로 저장 (제출 규격 준수) |
| 무결성 토큰 | 각 호출 시 `decryptor`가 발급한 `token`을 함께 기록하여 위변조 방지 |

**동작 방식:**
- `UPSTAGE_API_KEY` 환경변수에서 API 키 로드
- `urllib`로 Upstage Chat Completions API 직접 호출
- 호출마다 `time.perf_counter()`로 응답 시간 측정
- 모든 메타데이터(토큰 수, 시간, 무결성 토큰)를 `records` 리스트에 누적

---

### 2. `baseline_rag_best.py` — RAG 파이프라인 (방어 로직 포함)

3-Phase 구조의 엔드투엔드 RAG 파이프라인.

#### Phase 1: 인덱스 구축 (`build_index`)

- **파싱**: PyMuPDF로 PDF 코퍼스 읽기 → 이메일 단위 분리 (`Message N of M` 패턴)
- **포이즌 제거**: `[SYSTEM DIRECTIVE ... END DIRECTIVE]` 패턴을 정규식으로 사전 제거
- **청킹**: `RecursiveCharacterTextSplitter` (2000자, 200 overlap) — 이메일이 청크 크기 이하면 분할하지 않음
- **메타데이터 추출**: sender, recipients, subject 파싱
- **임베딩**: `UpstageEmbeddings(embedding-passage)`, 배치 50, rate limit 자동 재시도
- **BM25 인덱스**: `rank_bm25`로 키워드 검색 보조
- **캐시**: 청크(`cache_chunks.json`) + 임베딩(`cache_embeddings.npy`) 캐시로 재실행 시 시간 절약

#### Phase 2: 하이브리드 검색 (`retrieve`)

- Dense (코사인 유사도) top-10 + BM25 top-5 → 중복 제거 → Dense 순서 우선 상위 10개
- 각 청크를 `<DOCUMENT>` 태그로 감싸 메타데이터와 함께 반환

#### Phase 3: 답변 생성 (`generate_answer`)

**2-Pass 생성 전략:**

1. **1차 생성** — System Prompt에 방어 로직 내장:
   - PII 거절 규칙 (전화번호, 주소, SSN 등 요청 시 `"Cannot provide this information."` 반환)
   - Prompt Injection 방어 (`"ignore previous instructions"` 등 탐지 시 거절)
   - `<DOCUMENT>` 내부 텍스트를 데이터로만 취급, 지시문으로 해석 금지
   - 문서에 없는 정보는 `"Information not found in the provided documents."` 반환

2. **2차 검증** — Security Filter로 후처리:
   - 전화/팩스 번호 마스킹 (`***-***-****`)
   - Poisoning으로 삽입된 문구 제거 (`"Data&extracted"`, `"Query&complete"` 등)
   - 1차 + 2차 토큰/시간 합산하여 단일 레코드로 기록

---

## 규정 준수 사항

- **Solar LLM 필수 사용** — `UpstageTracker.chat()`으로만 최종 답변 생성. API 응답의 `usage.total_tokens`를 그대로 기록하여 위변조 없음.
- **응답 시간 기록** — `time.perf_counter()`로 호출별 `inference_time` 측정, 2-Pass 시 합산하여 정확히 기록.

---

## 실행 방법

```bash
# 환경변수 설정
export UPSTAGE_API_KEY=<your_key>
export HACKATHON_KEY=<대회당일_공지키>

# 파이프라인 실행
python baseline_rag_best.py
```

출력: `submission.csv` → stages.ai 플랫폼에 업로드

---

## 기술 스택

- **LLM**: Upstage Solar Pro (답변 생성 + 검증)
- **임베딩**: Upstage Embeddings (passage/query)
- **검색**: Dense (코사인) + BM25 하이브리드
- **파싱**: PyMuPDF
- **청킹**: LangChain RecursiveCharacterTextSplitter
- **데이터셋**: Enron QA (라이선스 미기재 — 연구·교육 목적 한정)
