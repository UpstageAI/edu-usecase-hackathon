# GRAGRA — Upstage 연고전 연합 AI 해커톤

**Team GRAGRA**

## 실행 방법

### Step 1. 임베딩 (Google Colab)

`GRAGRA+embedding.ipynb`을 Google Colab에서 실행하여 corpus PDF를 파싱·청킹·임베딩합니다.
생성된 `index.pkl` 파일을 팀원과 공유합니다.

### Step 2. RAG 파이프라인 (로컬)

팀원이 공유한 `index.pkl`을 로컬에 받아 RAG 파이프라인을 실행합니다.

```bash
pip install rank-bm25 sentence-transformers
export UPSTAGE_API_KEY=your_key

python GRAGRA_rag_pipeline.py \
  --index index.pkl \
  --suite distribution/test_suite/Encrypted_Test_Suite.json \
  --output submission.csv
```

`submission.csv`를 제출합니다.
