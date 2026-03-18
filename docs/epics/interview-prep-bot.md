# Epic: Interview Prep Bot

## Overview

Add an AI-powered interview preparation feature to Vett. When a user is applying for a job, they can launch an interview prep session from the JobDetails page. A conversational agent simulates a real interviewer — asking role and company-specific questions, probing on answers, and providing end-of-session feedback. A second phase adds full voice chat support.

---

## Background & Motivation

Vett already scores how well a candidate fits a job. The natural next step is helping them *get* the job. Interview prep is high-effort and hard to personalise — this feature automates the research and simulation work using the context Vett already has (CV, JD, job metadata).

---

## Stories

### Story 1: Document Knowledge Base — [#13](../../issues/13)
**Goal:** Users can upload reference materials (PDF, DOCX, TXT, MD) that the interview bot uses to generate better questions.

- Global upload in Settings ("Interview Docs") — applies to all sessions
- Per-job upload on JobDetails — applies only to that job's sessions
- Reuses existing file parser; both scopes fed into the vector store tagged with `global` or `job_id`
- RAG queries at session time merge both scopes
- UI: file list with upload + delete, clearly labelled scope

---

### Story 2: Agentic Interview Question Research — [#14](../../issues/14)
**Goal:** Before a session starts, an agent automatically researches role/company-specific interview questions from the web.

**Infrastructure**
- Add `searxng` to `docker-compose.yml` (self-hosted meta-search, no API key, internal only)
- Settings toggle to enable/disable research phase

**Agent**
A LangChain ReAct agent with tools:

| Tool | Purpose |
|---|---|
| `search_interview_questions` | Search for interview Q&A for this role/company across Glassdoor, Reddit, Leetcode |
| `search_role_skills` | Identify commonly tested skills and technologies for the role |
| `search_company_engineering_culture` | Scrape eng blogs, known tech stack, system design topics |
| `query_vector_store` | RAG over user-uploaded docs for domain-specific questions |
| `fetch_page` | Follow promising URLs and extract Q&A content |

Agent produces a structured **question bank** stored in SQLite:
```json
{
  "behavioral": [...],
  "technical": [...],
  "system_design": [...],
  "company_specific": [...],
  "source_urls": [...]
}
```

**UX**
- Triggered when user clicks "Prep Interview" on a job
- Streaming SSE status updates: "Researching interview questions for [Role] at [Company]..."
- Runs in ~10–20s, then session begins
- Question provenance visible to user (source URLs)

---

### Story 3: Job-Contextual Interview Chat — [#15](../../issues/15)
**Goal:** A conversational agent simulates a real interviewer, personalised to the role, company, and candidate.

**Session Initialisation**
- Backend assembles context: CV, JD, uploaded docs (RAG), question bank
- LLM generates an **interview plan** — ordered questions across categories, scaled to seniority inferred from CV

**Interview Flow**

| Phase | Behaviour |
|---|---|
| Opening | Bot introduces itself, sets context ("We're interviewing for Sr. Engineer at Acme...") |
| Questioning | One question at a time. Bot stays in character — no hints, no encouragement |
| Probing | Follows up on vague answers rather than moving on |
| Transition | Natural flow: behavioral → technical → system design |
| Closing | "Do you have any questions for me?" — bot answers from JD/company context |

**Question Categories**
- Behavioral (STAR): 2–3 questions
- Technical/role-specific: 3–4 questions from CV skills + JD requirements
- System design: 1–2 questions
- Company-specific: 1–2 questions

**Agent Tools (mid-session)**

| Tool | Purpose |
|---|---|
| `query_vector_store` | Pull relevant context if a topic arises mid-session |
| `get_cv_detail` | Personalise follow-ups ("You mentioned at Company X...") |
| `get_job_detail` | Reference JD requirements during questioning |

**Session State**
- Full transcript saved to SQLite per turn, resumable if closed
- Each turn tagged: `question`, `answer`, `follow_up`, `transition`
- Multiple sessions per job (practice rounds), listed with dates

**UX**
- Chat panel on JobDetails (slide-in or dedicated route)
- Streaming responses via SSE
- "End Interview" triggers Story 4

---

### Story 4: Answer Feedback & Session Summary — [#16](../../issues/16)
**Goal:** After the session, give the candidate actionable feedback.

- Per-answer critique: STAR structure, relevance, depth, what was missing
- End-of-session summary: strengths, gaps, suggested improvement areas
- Summary stored in SQLite, viewable from session history

---

### Story 5: Voice Input (STT) — [#17](../../issues/17) (renamed from Story 6 in issue tracker)
**Goal:** User can speak their answers instead of typing.

- Mic button in chat UI
- Transcription via OpenAI Whisper API (uses existing OpenAI key config) or faster-whisper Docker container for fully local option
- User selects STT provider in Settings (same pattern as LLM config)
- Transcribed text feeds into existing chat pipeline unchanged

---

### Story 6: Voice Output (TTS) — [#17](../../issues/17)
**Goal:** Bot speaks its questions aloud.

- Start with browser `SpeechSynthesis` API (zero cost, zero infra)
- Upgrade path: OpenAI TTS API for better voice quality (configurable in Settings)
- Mute/unmute toggle in chat UI

---

### Story 7: Full Voice Interview Mode
**Goal:** Hands-free interview simulation.

- Bot speaks question → listens for answer → auto-submits on silence detection → next question
- Visual states: listening / thinking / speaking
- Feels like a real phone screen

---

## Technical Notes

- **Vector store:** Existing implementation, extend tagging to support `global` / `job_id` scopes
- **SearXNG:** Add to `docker-compose.yml`, internal only, use `langchain_community.utilities.SearxSearchWrapper`
- **SSE:** Reuse existing SSE infrastructure for research + chat streaming
- **SQLite schema additions:** `interview_sessions`, `session_turns`, `question_banks`, `interview_docs`
- **LangChain:** ReAct agent for research (Story 2), conversation agent for chat (Story 3)

## Out of Scope
- Multi-user / cloud sync
- Video interview simulation
- Scoring against industry benchmarks (future)
