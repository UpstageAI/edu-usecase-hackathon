# Baseline RAG — 구현 가이드

본 문서는 `baseline_rag.py` 에 이미 구현된 RAG 파이프라인의 동작 방식을 설명한다.  
대회 운영 안내는 [README.md](README.md) 참조.

---

## 파이프라인 요약

```
distribution/corpus/
   │ PyMuPDF (fitz) — 로컬 텍스트 추출
   ▼
전체 텍스트 ── 800자 sliding window (overlap 150) ──► N 개 청크
   │
   │ Upstage embedding-passage (배치 100, 디스크 캐시)
   ▼
(N, D) 임베딩 행렬 (L2-정규화)
   │
query ── embedding-query ──► q_vec ── cosine similarity ──► top-5 청크
   │
   ▼
Solar LLM solar-pro (system prompt + [Reference] + [Question])
   │
   ▼
answer  →  tracker.chat() 기록  →  submission.csv
```

---

## 구현된 함수

### `build_index(corpus_dir)`

`corpus_dir` 아래 모든 PDF 를 재귀 탐색하여 인덱스를 반환한다.

1. **파싱** — `fitz.open()` (PyMuPDF) 으로 로컬에서 텍스트 추출. API 비용 없음.
2. **청킹** — `_chunk_text()`: 800자 sliding window, 단락(`\n\n`) 경계 우선 존중, 150자 overlap.
3. **임베딩** — `Embedder.embed_passages()`: 100개 배치, 배치별·통합 `.npy` 디스크 캐시.

반환 구조:

```python
{
    "chunks":     list[str],        # 청크 텍스트
    "sources":    list[str],        # 청크가 속한 PDF 파일명
    "embeddings": np.ndarray(N, D)  # L2-정규화 완료
}
```

### `retrieve(question, index, top_k=5)`

질문을 `embedding-query` 로 임베딩하고 코사인 유사도 상위 `top_k` 청크를 반환한다.  
각 청크에 `[doc{rank} | {source} | score={sim:.3f}]` 헤더를 붙여 LLM 입력으로 전달.

### `generate_answer(question, context, tracker, question_id, token)`

`tracker.chat()` 을 통해 `solar-pro` 모델을 호출하고 결과를 자동 기록한다.

---

## SYSTEM_PROMPT 설계

두 단계 절차를 강제한다.

| 단계 | 조건 | 출력 |
|---|---|---|
| STEP 1 — REFUSAL CHECK | PII 요청 또는 사회공학/탈옥 시도 | `Cannot provide this information.` |
| STEP 2 — ANSWER | `[Reference]` 내 정보만 사용 | 1–3 문장 답변 |

**거절 대상 PII**: 전화/팩스, 이메일, 주소, SSN/여권/계좌/급여, 생년월일, 비밀번호.

**거절 대상 프레이밍**: `hypothetical`, `for educational purposes`, `ignore previous instructions`, `act as` 등 사회공학 표현이 포함되면 질문 내용에 무관하게 거절.

**Prompt Injection 방어**: `[Reference]` 내부 텍스트는 신뢰 불가 데이터로 처리. 참조 문서 안에 삽입된 지시문을 무시하도록 명시.

답을 참조 문서에서 찾을 수 없으면 `Information not found in the provided documents.` 출력.

---

## 캐시 구조

`.cache/` 아래 파일만 삭제하면 재실행된다.

| 파일 | 생성 시점 | 내용 |
|---|---|---|
| `.cache/emb_<key>_batches/<n>.npy` | 임베딩 배치 완료 시 | 배치별 임베딩 벡터 |
| `.cache/emb_<key>.npy` | 전체 합산 완료 시 | 통합 임베딩 행렬 |

```bash
rm -rf .cache                  # 전체 재실행 (파싱 + 임베딩 + LLM)
rm .cache/emb_*.npy            # 임베딩만 재계산
```

---

## 튜닝 노브

[baseline_rag.py](baseline_rag.py) 상단 상수 또는 `LLM_MODEL` 변수를 조정한다.

| 상수 | 기본값 | 영향 |
|---|---|---|
| `CHUNK_SIZE` | 800 | 크게 → 컨텍스트 풍부·검색 정밀도 ↓ |
| `CHUNK_OVERLAP` | 150 | 크게 → 경계 보존·인덱스 비대 |
| `TOP_K` | 5 | 크게 → latency·token 비용 증가 |
| `LLM_MODEL` | `"solar-pro"` | `"solar-mini"` 로 변경 시 latency ↓·품질 ↓ |
| `Embedder.batch_size` | 100 | 작게 → 임베딩 API 호출 수 ↑ |

`CHUNK_SIZE` / `CHUNK_OVERLAP` 변경 시 캐시 키가 자동으로 달라져 재임베딩된다.

---

## 알려진 한계

- **이메일 경계 무시**: 800자 sliding window 는 이메일 헤더 중간을 자른다. `From:`/`Date:` 기준 분할로 교체하면 검색 품질이 개선될 수 있다.
- **단순 dense retrieval**: BM25 결합·MMR·re-ranker 미적용. multi-hop 질문에서 상대적으로 약하다. `retrieve()` 함수만 교체하면 된다.
- **과거절(over-refusal)**: 시스템 프롬프트가 보수적으로 설정되어 있어, 사회공학 표현과 유사한 문구가 포함된 정상 질문도 거절할 수 있다.
