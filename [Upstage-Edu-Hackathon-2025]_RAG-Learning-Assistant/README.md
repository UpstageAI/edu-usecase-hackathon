# [Upstage-Edu-Hackathon-2025] RAG-based Learning Assistant

> 학생이 업로드한 PDF 교재/강의자료를 기반으로 Solar LLM이 1:1 학습 도우미가 되어주는 해커톤.

## 개요

| 항목 | 내용 |
|---|---|
| 개최일 | 2025-09-20 ~ 2025-09-21 |
| 주최/주관 | Upstage AI |
| 장소 | Seoul, Online 병행 |
| 참가 규모 | 30팀 / 약 120명 |
| 공식 링크 | https://upstage.ai/events/example |

## 주제

학습자가 보유한 **교재·강의자료 PDF를 업로드**하면 이를 이해하고 **질의응답·요약·문제 출제·해설**까지 제공하는 RAG 기반 학습 도우미를 만드는 것이 미션이었습니다.

평가 기준
- Upstage API 활용도 (Document Parse 정확도, Solar LLM 응답 품질)
- 교육적 효용성 (실제 학습자에게 도움이 되는가)
- 데모 완성도 / UX

## 활용된 Upstage API

- **Solar LLM (Chat)** — 학습자 질문 응답, 문제 출제, 풀이 해설
- **Document Parse** — PDF 교재의 표·이미지 포함 텍스트 추출
- **Solar Embeddings** — 청크 임베딩 → 벡터 DB 검색 (RAG)

## 프로젝트 목록

| 프로젝트 | 한줄 설명 | 주요 활용 API |
|---|---|---|
| [solar-tutor-bot](./solar-tutor-bot) | 업로드한 PDF 기반 1:1 튜터 챗봇 | Solar Chat, Document Parse, Embeddings |
