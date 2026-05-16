# edu-usecase-hackathon

업스테이지(Upstage) 교육 분야 해커톤에서 발굴된 유즈케이스를 수집하는 레포지토리입니다.
Solar LLM, Document Parse, Embeddings 등 Upstage API가 교육 현장에서 어떻게 활용되었는지 사례를 모읍니다.

## 폴더 구조

```
edu-usecase-hackathon/
├── README.md                                     # 본 문서
├── _TEMPLATE/                                    # 신규 해커톤 추가용 템플릿
│   ├── README.md                                 # 해커톤 README 템플릿
│   └── _PROJECT_TEMPLATE/
│       └── README.md                             # 프로젝트 README 템플릿
└── [HackathonName]_Topic/                        # 해커톤별 폴더
    ├── README.md                                 # 해커톤 개요
    └── project-name/                             # 프로젝트별 폴더
        ├── README.md                             # 프로젝트 README
        └── ...                                   # 프로젝트 코드/리소스
```

## 네이밍 규칙

### 해커톤 폴더 — `[HackathonName]_Topic` (영문)

폴더명만 보고도 **어떤 해커톤**에서 **어떤 Upstage API/주제**로 진행됐는지 드러나게 작성합니다.

예시:
- `[Upstage-Edu-Hackathon-2025]_RAG-Learning-Assistant`
- `[UpstageX-2024]_Document-Parse-for-EduContent`
- `[GlobalAI-Edu]_Solar-Multilingual-Tutoring`

### 프로젝트 폴더 — `project-name` (kebab-case, 영문 권장)

프로젝트명만으로도 **어떤 use case** 인지 파악 가능하게 작성합니다.

예시:
- `solar-tutor-bot` — 업로드한 PDF 기반 1:1 튜터 챗봇
- `pdf-quiz-generator` — 교재 PDF에서 자동 문제 생성
- `essay-feedback-coach` — 학생 에세이 피드백 자동화

## 새 해커톤/프로젝트 추가하기

1. `_TEMPLATE/` 폴더를 복사하여 `[해커톤명]_주제` 형식으로 이름을 변경합니다.
2. 해커톤 폴더 안의 `_PROJECT_TEMPLATE/` 을 복사하여 프로젝트명으로 변경합니다.
3. 각 README의 플레이스홀더(`<...>`)를 채워주세요.
4. Pull Request를 생성해주세요.

## 라이선스

별도 명시가 없는 한 MIT License를 따릅니다.
