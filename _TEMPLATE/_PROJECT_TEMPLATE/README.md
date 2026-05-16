# <프로젝트명>

> 한줄 소개: 누구의(타겟 사용자) 어떤 문제를, 어떤 Upstage API를 활용해 해결하는지 한 문장으로 작성해주세요.

---

## 1. 프로젝트 개요 & 문제 정의

### 해결하려는 문제
- **타겟 사용자**: <e.g. 중·고등학생, 자기주도학습자, 교사>
- **어떤 상황에서**: <e.g. 두꺼운 교재로 시험 공부를 할 때>
- **어떤 어려움을 겪는가**: <e.g. 핵심을 빠르게 잡기 어렵고, 모르는 부분을 질문할 곳이 없음>

### 제안하는 솔루션
<한 단락으로 솔루션을 요약합니다. 어떤 입력을 받아 어떤 결과를 주는지, 차별점은 무엇인지 작성해주세요.>

### 기대 효과
- <학습 시간 단축 / 이해도 향상 / 접근성 개선 등>

---

## 2. 사용한 Upstage API & 사용 방식

활용한 API와 코드 내 사용 위치를 표로 정리해주세요.

| API | 사용 위치 | 사용 목적 |
|---|---|---|
| Solar Chat | `src/chat.py` | <e.g. 학습자 질문에 답변 생성> |
| Document Parse | `src/ingest.py` | <e.g. PDF 교재 파싱 → RAG 인덱싱> |
| Solar Embeddings | `src/embed.py` | <e.g. 청크 임베딩 → 벡터 DB 적재> |

### 핵심 호출 예시

```python
from openai import OpenAI

client = OpenAI(
    api_key="<UPSTAGE_API_KEY>",
    base_url="https://api.upstage.ai/v1",
)

response = client.chat.completions.create(
    model="solar-pro2",
    messages=[{"role": "user", "content": "..."}],
)
```

### 구현 디테일
<프롬프트 설계, 청크 크기, 검색 전략(top-k, reranking 여부), 후처리 등 구현에서 신경 쓴 부분을 자유롭게 서술해주세요.>

---

## 3. 데모 / 결과

### 스크린샷 / 영상
스크린샷, GIF, 또는 짧은 영상 링크를 첨부해주세요.

![demo](./assets/demo.png)

### 라이브 데모
- 데모 링크: <https://...>
- 데모 계정: <id / pw, 필요한 경우>

### 핵심 결과 지표(선택)
- <e.g. 평균 응답 시간 1.2초, RAG 정확도 87%, 사용자 만족도 4.5/5>

---

## 4. 실행 방법

### 사전 요구사항
- Python 3.10+
- Upstage API Key — [발급받기](https://console.upstage.ai/)

### 설치
```bash
git clone <repo-url>
cd <project-name>
pip install -r requirements.txt
```

### 환경 변수
```bash
export UPSTAGE_API_KEY=<your_key_here>
```

`.env.example` 파일을 참고하여 `.env` 파일을 생성해주세요.

### 실행
```bash
python main.py
```

---

## 5. 팀원

| 이름 | 역할 | GitHub / 연락처 |
|---|---|---|
| <홍길동> | <기획 / 백엔드> | [@username](https://github.com/username) |
| | | |

---

## 6. 라이선스

별도 명시가 없는 한 [MIT License](../../LICENSE)를 따릅니다.
