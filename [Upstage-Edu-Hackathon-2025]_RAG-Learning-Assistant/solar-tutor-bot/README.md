# solar-tutor-bot

> 학습자가 PDF 교재를 업로드하면 Solar LLM이 그 내용을 학습해 **1:1 튜터처럼 질문에 답하고, 핵심을 요약하며, 연습문제까지 출제**해주는 챗봇.

---

## 1. 프로젝트 개요 & 문제 정의

### 해결하려는 문제
- **타겟 사용자**: 시험을 준비하는 중·고등학생 및 자기주도학습자
- **어떤 상황에서**: 두꺼운 교재·강의노트로 시험 공부를 할 때
- **어떤 어려움을 겪는가**: 핵심을 빠르게 잡기 어렵고, 모르는 부분을 즉시 질문할 곳이 없음

### 제안하는 솔루션
사용자가 학습 자료(PDF)를 업로드하면 Document Parse로 텍스트·표·이미지를 추출하고, 청크화 후 Solar Embeddings로 벡터 DB에 적재합니다. 학습자가 질문하면 RAG로 관련 청크를 가져와 Solar Chat이 **출처와 함께** 답변합니다. "이 단원 핵심만 알려줘", "연습문제 5개 만들어줘" 같은 학습 모드도 지원합니다.

### 기대 효과
- 사교육 없이도 모르는 내용을 즉시 해소
- 교재 전체를 검색·요약 가능해 학습 시간 단축
- 출제 기능으로 자기 점검 루프 강화

---

## 2. 사용한 Upstage API & 사용 방식

| API | 사용 위치 | 사용 목적 |
|---|---|---|
| Document Parse | `src/ingest.py` | PDF 교재 파싱 (표·이미지 포함) |
| Solar Embeddings | `src/embed.py` | 청크 임베딩 → FAISS 인덱싱 |
| Solar Chat (`solar-pro2`) | `src/chat.py` | RAG 답변 / 요약 / 문제 출제 |

### 핵심 호출 예시

**Document Parse**
```python
import requests

resp = requests.post(
    "https://api.upstage.ai/v1/document-ai/document-parse",
    headers={"Authorization": f"Bearer {UPSTAGE_API_KEY}"},
    files={"document": open("textbook.pdf", "rb")},
)
parsed = resp.json()
```

**Embeddings + Chat (OpenAI SDK 호환)**
```python
from openai import OpenAI

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1",
)

# 1) 청크 임베딩
vec = client.embeddings.create(
    model="solar-embedding-1-large-passage",
    input=chunk_text,
).data[0].embedding

# 2) RAG 답변 생성
answer = client.chat.completions.create(
    model="solar-pro2",
    messages=[
        {"role": "system", "content": "당신은 친절한 학습 튜터입니다. 제공된 문맥만 활용해 답변하고, 출처를 명시하세요."},
        {"role": "user", "content": f"문맥:\n{retrieved_chunks}\n\n질문: {user_question}"},
    ],
)
```

### 구현 디테일
- **청크 전략**: 단원 제목 기준 1차 분할 → 700 토큰 단위 슬라이딩 윈도우(오버랩 100토큰)
- **검색**: top-k=5, 코사인 유사도, 동일 단원 청크는 가중치 +0.1
- **프롬프트**: 모드별 시스템 프롬프트 분리 (질의응답 / 요약 / 출제 / 해설)
- **출처 표기**: 응답 말미에 `[p.12, 3단원]` 형식으로 페이지·단원 메타 포함

---

## 3. 데모 / 결과

### 스크린샷
![demo](./assets/demo.png)

### 라이브 데모
- 데모 링크: <https://solar-tutor-bot.example.com>
- 데모 계정: `guest / guest1234`

### 핵심 결과 지표
- 평균 응답 시간 **1.8초**
- 자체 평가셋(100문항) 정답률 **88%**
- 베타 테스터 만족도 **4.6 / 5**

---

## 4. 실행 방법

### 사전 요구사항
- Python 3.10+
- Upstage API Key — [발급받기](https://console.upstage.ai/)

### 설치
```bash
git clone <repo-url>
cd solar-tutor-bot
pip install -r requirements.txt
```

### 환경 변수
```bash
cp .env.example .env
# .env 파일을 열어 UPSTAGE_API_KEY 값을 채워주세요
```

### 실행
```bash
# 1) PDF 인덱싱
python -m src.ingest --pdf ./data/textbook.pdf

# 2) 챗봇 실행
streamlit run app.py
```

---

## 5. 팀원

| 이름 | 역할 | GitHub / 연락처 |
|---|---|---|
| 김수완 | 기획 / 백엔드 / RAG 파이프라인 | [@SuWanKim-code](https://github.com/SuWanKim-code) |
| 홍길동 | 프론트엔드 / 데모 UX | [@gildong](https://github.com/gildong) |
| 김영희 | 데이터셋 / 평가 | [@younghee](https://github.com/younghee) |

---

## 6. 라이선스

[MIT License](../../LICENSE)
