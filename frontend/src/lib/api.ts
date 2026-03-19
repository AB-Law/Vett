import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

api.interceptors.response.use((response) => {
  if (import.meta.env.DEV && response.config.url?.startsWith('/jobs/')) {
    // eslint-disable-next-line no-console
    console.debug('[Vett API Response]', response.config.method?.toUpperCase(), response.config.url, response.data)
  }
  return response
})

// ── CV ────────────────────────────────────────────────────────────────────────

export interface CV {
  id: number
  filename: string
  file_size: number
  file_type: string
  parsed_text: string
  created_at: string
}

export const uploadCV = async (file: File): Promise<CV> => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<CV>('/cv/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getCV = async (): Promise<CV | null> => {
  const { data } = await api.get<CV | null>('/cv/')
  return data
}

export const deleteCV = async (): Promise<void> => {
  await api.delete('/cv/')
}

// ── Score ──────────────────────────────────────────────────────────────────────

export interface ScoreResult {
  id?: number
  fit_score: number
  matched_keywords: string[]
  missing_keywords: string[]
  gap_analysis: string
  rewrite_suggestions: string[]
  matched_keyword_evidence: ScoreEvidenceRecord[]
  missing_keyword_evidence: ScoreEvidenceRecord[]
  rewrite_suggestion_evidence: ScoreEvidenceRecord[]
  reason?: string
  agent_plan?: ScoreAgentPlan
  job_title?: string
  company?: string
  llm_provider?: string
  llm_model?: string
  run?: ScoreRun
  run_transitions?: ScoreRunTransition[]
  run_artifacts?: ScoreRunArtifact[]
  run_id?: string
  run_state?: string
  run_status?: string
  run_attempt_count?: number
  run_error?: string
}

export interface ScoreRun {
  id: string
  current_state: string
  status: string
  attempt_count: number
  failed_step: string | null
  failure_reason: string | null
  created_at: string
  updated_at: string
  completed_at: string | null
  score_history_id: number | null
}

export interface ScoreRunTransition {
  id: number
  previous_state: string | null
  next_state: string
  trigger: string
  attempt: number
  failure_reason: string | null
  latency_ms: number | null
  idempotency_key: string
  created_at: string
  actor: string
  source: string
}

export interface ScoreRunArtifact {
  id: number
  step: string
  payload: Record<string, unknown> | null
  evidence: Record<string, unknown> | null
  attempt: number
  latency_ms: number | null
  actor: string
  source: string
  created_at: string
  transition_id: number | null
  score_history_id: number | null
}

export interface EvidenceCitation {
  section_id?: string
  phrase_id?: string
  line_start?: number
  line_end?: number
  snippet?: string
}

export interface ScoreEvidenceRecord {
  value: string
  cv_citations: EvidenceCitation[]
  jd_phrase_citations: EvidenceCitation[]
  evidence_missing_reason?: string
}

export interface ScoreAgentPlan {
  role_signal_map: Record<string, string | string[]>
  skills_to_fix_first: string[]
  concrete_edit_actions: string[]
  interview_topics_to_prioritize: string[]
  study_order: string[]
}

export interface ScoreRequest {
  job_description: string
  job_title?: string
  company?: string
}

export interface CVRewriteRequest {
  job_description: string
  job_title?: string
  company?: string
}

export interface CVRewriteProposal {
  before: string
  after: string
  reason: string
  risk_or_uncertainty: string
}

export interface CVRewriteResponse {
  proposals: CVRewriteProposal[]
  job_title?: string
  company?: string
  llm_provider?: string
  llm_model?: string
}

export const scoreJD = async (req: ScoreRequest): Promise<ScoreResult> => {
  const { data } = await api.post<ScoreResult>('/score/', req)
  return data
}

export const getCVRewriteProposals = async (req: CVRewriteRequest): Promise<CVRewriteResponse> => {
  const { data } = await api.post<CVRewriteResponse>('/score/rewrite-proposals', req)
  return data
}

export interface HistoryItem extends ScoreResult {
  id: number
  created_at: string
}

export const getHistory = async (): Promise<HistoryItem[]> => {
  const { data } = await api.get<HistoryItem[]>('/score/history')
  return data
}

export const deleteHistoryItem = async (id: number): Promise<void> => {
  await api.delete(`/score/history/${id}`)
}

export const clearHistory = async (): Promise<void> => {
  await api.delete('/score/history')
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface AppSettings {
  active_provider: string
  claude_model: string
  openai_model: string
  azure_openai_endpoint: string
  azure_openai_deployment: string
  azure_openai_api_version: string
  ollama_base_url: string
  ollama_model: string
  lm_studio_base_url: string
  lm_studio_model: string
  save_history: boolean
  default_export_format: string
  tts_provider: 'native' | 'kokoro'
  voice_preferred_name: string
  voice_rate: number
  voice_pitch: number
  has_anthropic_key: boolean
  has_openai_key: boolean
  has_azure_key: boolean
}

export interface CandidateProfile {
  id: number | null
  full_name: string
  headline_or_target_role: string
  current_company: string
  years_experience: number | null
  top_skills: string[]
  location: string
  linkedin_url: string
  summary: string
  source: string
}

export interface CandidateProfileUpdate {
  full_name?: string
  headline_or_target_role?: string
  current_company?: string
  years_experience?: number | null
  top_skills?: string[]
  location?: string
  linkedin_url?: string
  summary?: string
  source?: string
}

export interface InterviewKnowledgeDocument {
  id: number
  owner_type: 'global' | 'job'
  job_id: number | null
  source_filename: string
  content_type: string
  status: string
  error_message: string | null
  parser_version: string | null
  source_ref: string | null
  created_at: string
  created_by_user_id: string | null
  total_chunks: number
  embedded_chunks: number
  parsed_word_count: number
}

export interface InterviewKnowledgeDocumentProgress {
  id: number
  owner_type: 'global' | 'job'
  job_id: number | null
  source_filename: string
  status: string
  total_chunks: number
  embedded_chunks: number
  progress_percent: number
  error_message: string | null
  parsed_word_count: number
  created_at: string
  created_by_user_id: string | null
}

export const getSettings = async (): Promise<AppSettings> => {
  const { data } = await api.get<AppSettings>('/settings/')
  return data
}

export const updateSettings = async (settings: Partial<AppSettings> & { [k: string]: unknown }): Promise<void> => {
  await api.post('/settings/', settings)
}

export const getUserProfile = async (): Promise<CandidateProfile> => {
  const { data } = await api.get<CandidateProfile>('/profile/')
  return data
}

export const updateUserProfile = async (payload: CandidateProfileUpdate): Promise<CandidateProfile> => {
  const { data } = await api.post<CandidateProfile>('/profile/', payload)
  return data
}

export const testConnection = async (): Promise<{ ok: boolean; reply?: string; error?: string }> => {
  const { data } = await api.post('/settings/test-connection')
  return data
}

export const getEmbeddingProgress = async (): Promise<{ total: number; embedded: number; percent: number }> => {
  const { data } = await api.get('/settings/embedding-progress')
  return data
}

export const getInterviewDocuments = async (): Promise<InterviewKnowledgeDocument[]> => {
  const { data } = await api.get<InterviewKnowledgeDocument[]>('/settings/interview-documents')
  return data
}

export const getInterviewDocumentProgress = async (): Promise<InterviewKnowledgeDocumentProgress[]> => {
  const { data } = await api.get<InterviewKnowledgeDocumentProgress[]>('/settings/interview-documents/progress')
  return data
}

export const uploadInterviewDocument = async (file: File): Promise<InterviewKnowledgeDocument> => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<InterviewKnowledgeDocument>('/settings/interview-documents', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getJobInterviewDocuments = async (jobId: number): Promise<InterviewKnowledgeDocument[]> => {
  const { data } = await api.get<InterviewKnowledgeDocument[]>(`/jobs/${jobId}/interview-documents`)
  return data
}

export const uploadJobInterviewDocument = async (jobId: number, file: File): Promise<InterviewKnowledgeDocument> => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<InterviewKnowledgeDocument>(`/jobs/${jobId}/interview-documents`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getInterviewResearchSession = async (
  jobId: number,
  sessionId: string,
): Promise<InterviewResearchSession> => {
  const { data } = await api.get<InterviewResearchSession>(`/jobs/${jobId}/interview-research/session/${sessionId}`)
  return data
}

export const cancelInterviewResearchSession = async (
  jobId: number,
  sessionId: string,
): Promise<{ status: string; job_id: number; session_id: string }> => {
  const { data } = await api.post<{ status: string; job_id: number; session_id: string }>(
    `/jobs/${jobId}/interview-research/session/${sessionId}/cancel`,
  )
  return data
}

export const buildInterviewResearchStreamUrl = (jobId: number): string =>
  `/api/jobs/${jobId}/interview-research/stream`

export interface InterviewChatToolCall {
  tool: string
  status: string
  result_count: number
  error?: string
}

export interface InterviewChatTurn {
  id: number
  turn_index: number
  speaker: 'assistant' | 'user'
  turn_type: 'question' | 'answer' | 'follow_up' | 'transition'
  content: string
  tool_calls: InterviewChatToolCall[]
  context_sources: string[]
  created_at: string
}

export interface InterviewChatSession {
  session_id: string
  label: string
  status: string
  phase: string
  created_at: string
  updated_at: string
  completed_at: string | null
  turn_count: number
  handoff_run_id: string | null
  preparation_status?: string | null
  rolling_score?: number | null
  limits?: {
    min_questions: number
    target_questions: number
    max_questions: number
  } | null
  primary_question_count?: number
}

export interface InterviewChatSessionDetail extends InterviewChatSession {
  job_id: number
  feedback?: InterviewChatFeedback | null
  thread_score_snapshot?: {
    score?: number
    rolling_score?: number
    category?: string
    question?: string
  } | null
  turns: InterviewChatTurn[]
}

export interface InterviewChatFeedback {
  overview: string
  what_went_well: string[]
  what_to_improve: string[]
  next_steps: string[]
}

export const listInterviewChatSessions = async (jobId: number): Promise<InterviewChatSession[]> => {
  const { data } = await api.get<InterviewChatSession[]>(`/interview-chat/jobs/${jobId}/sessions`)
  return data
}

export const createOrResumeInterviewChatSession = async (jobId: number): Promise<InterviewChatSessionDetail> => {
  const { data } = await api.post<{ session: InterviewChatSessionDetail }>(`/interview-chat/jobs/${jobId}/sessions`)
  return data.session
}

export const getInterviewChatSession = async (jobId: number, sessionId: string): Promise<InterviewChatSessionDetail> => {
  const { data } = await api.get<InterviewChatSessionDetail>(`/interview-chat/jobs/${jobId}/sessions/${sessionId}`)
  return data
}

export type InterviewChatStreamResult = {
  message: string
  phase: string
  turn_types: string[]
  tool_calls: InterviewChatToolCall[]
  context_sources: string[]
  preparation_status?: string
  rolling_score?: number | null
  thread_score_snapshot?: Record<string, unknown> | null
  primary_question_count?: number
  limits?: {
    min_questions: number
    target_questions: number
    max_questions: number
  } | null
}

export const streamInterviewChatTurn = async (
  jobId: number,
  sessionId: string,
  message: string | null,
  onToken: (token: string) => void,
  options?: { signal?: AbortSignal },
): Promise<InterviewChatStreamResult> => {
  const response = await fetch(`/api/interview-chat/jobs/${jobId}/sessions/${sessionId}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal: options?.signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`Interview stream failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let donePayload: InterviewChatStreamResult | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      if (!block.startsWith('data: ')) continue
      const payload = JSON.parse(block.slice(6)) as {
        type?: string
        delta?: string
        message?: string
        phase?: string
        turn_types?: string[]
        tool_calls?: InterviewChatToolCall[]
        context_sources?: string[]
        preparation_status?: string
        rolling_score?: number
        thread_score_snapshot?: Record<string, unknown> | null
        primary_question_count?: number
        limits?: {
          min_questions: number
          target_questions: number
          max_questions: number
        }
      }
      if (payload.type === 'token' && payload.delta) {
        onToken(payload.delta)
      }
      if (payload.type === 'done') {
        donePayload = {
          message: payload.message || '',
          phase: payload.phase || '',
          turn_types: payload.turn_types || [],
          tool_calls: payload.tool_calls || [],
          context_sources: payload.context_sources || [],
          preparation_status: payload.preparation_status,
          rolling_score: typeof payload.rolling_score === 'number' ? payload.rolling_score : null,
          thread_score_snapshot: payload.thread_score_snapshot ?? null,
          primary_question_count: payload.primary_question_count ?? 0,
          limits: payload.limits ?? null,
        }
      }
    }
  }

  if (!donePayload) {
    throw new Error('Interview stream ended without completion payload')
  }
  return donePayload
}

export const endInterviewChatSession = async (
  jobId: number,
  sessionId: string,
): Promise<{
  session_id: string
  status: string
  handoff_status: string
  handoff_run_id: string | null
  feedback: InterviewChatFeedback | null
}> => {
  const { data } = await api.post<{
    session_id: string
    status: string
    handoff_status: string
    handoff_run_id: string | null
    feedback: InterviewChatFeedback | null
  }>(
    `/interview-chat/jobs/${jobId}/sessions/${sessionId}/end`,
  )
  return data
}

export const deleteInterviewChatSession = async (
  jobId: number,
  sessionId: string,
): Promise<{ session_id: string; status: string }> => {
  const { data } = await api.delete<{ session_id: string; status: string }>(
    `/interview-chat/jobs/${jobId}/sessions/${sessionId}`,
  )
  return data
}

export const transcribeInterviewAudio = async (
  jobId: number,
  sessionId: string,
  audioBlob: Blob,
): Promise<{ transcript: string; latency_ms: number }> => {
  const form = new FormData()
  form.append('audio_file', audioBlob, 'interview-input.webm')
  try {
    const { data } = await api.post<{ transcript: string; latency_ms: number }>(
      `/interview-chat/jobs/${jobId}/sessions/${sessionId}/transcribe`,
      form,
      { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 },
    )
    return data
  } catch (error: unknown) {
    if (axios.isAxiosError(error)) {
      const detail = (error.response?.data as { detail?: string } | undefined)?.detail
      throw new Error(detail || `Transcription failed (${error.response?.status ?? 'network'})`)
    }
    throw error
  }
}

// ── Jobs (Phase 2) ────────────────────────────────────────────────────────────

export interface Job {
  id: number
  title?: string
  company?: string
  location?: string
  url?: string
  source?: string
  work_type?: string
  external_job_id?: string
  canonical_url?: string
  posted_at_raw?: string
  employment_type?: string
  job_function?: string
  industries?: string
  applicants_count?: string
  benefits?: string[]
  salary?: string
  company_logo?: string
  company_linkedin_url?: string
  company_website?: string
  company_address?: Record<string, string>
  company_employees_count?: number
  job_poster_name?: string
  job_poster_title?: string
  job_poster_profile_url?: string
  seniority?: string
  posted_at?: string
  fit_score?: number
  matched_keywords?: string[]
  missing_keywords?: string[]
  gap_analysis?: string
  reason?: string
  created_at: string
}

export interface InterviewResearchQuestion {
  question: string
  question_text?: string
  tool: string
  query: string
  source_url: string
  source_title: string
  source_type?: string
  query_used?: string
  reason?: string
  timestamp: string
  snippet: string
  confidence_score: number
  citations?: InterviewResearchCitation[]
}

export interface InterviewResearchCitation {
  source_url: string
  source_title: string
  snippet: string
  page_index?: number | null
  confidence: number
}

export interface InterviewResearchQuestionBank {
  behavioral: InterviewResearchQuestion[]
  technical: InterviewResearchQuestion[]
  system_design: InterviewResearchQuestion[]
  company_specific: InterviewResearchQuestion[]
  source_urls: string[]
}

export interface StudyCard {
  id: number
  front: string
  back: string
  last_reviewed_at: string | null
  ease_factor: number
  interval_days: number
}

export interface FlashcardSetResponse {
  card_set_id: number
  cards: StudyCard[]
  card_set: StudyCardSetSummary
  card_sets: FlashcardSetItem[]
  parent_card_set_id: number | null
  generation_diagnostics?: FlashcardGenerationDiagnostics
}

export type FlashcardGenerationJobStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface FlashcardGenerationDiagnostics {
  requested_cards: number
  llm_cards_parsed: number
  deduped_out: number
  fallback_cards_used: number
  fallback_used: boolean
}

export interface FlashcardGenerationStartResponse {
  job_id: string
  status: FlashcardGenerationJobStatus
}

export interface FlashcardGenerationJobResponse {
  job_id: string
  status: FlashcardGenerationJobStatus
  created_at: string
  updated_at: string
  error: string | null
  result: FlashcardSetResponse | null
}

export interface GenerateFlashcardsRequest {
  job_id?: number
  document_ids?: number[]
  name?: string
  topic?: string
  num_cards?: number
  generate_per_section?: boolean
}

export interface StudyCardReviewRequest {
  rating: 'easy' | 'hard'
}

export interface StudyCardSetSummary {
  id: number
  job_id: number | null
  parent_card_set_id: number | null
  name: string
  topic: string | null
  created_at: string | null
  card_count: number
  document_ids: number[]
  document_count: number
}

export interface StudyCardSetDetailResponse {
  card_set_id: number
  job_id: number | null
  parent_card_set_id: number | null
  name: string
  topic: string | null
  created_at: string | null
  cards: StudyCard[]
  document_ids: number[]
  document_count: number
}

export interface FlashcardSetItem {
  card_set_id: number
  cards: StudyCard[]
  card_set: StudyCardSetSummary
}

export interface StudyCardSetRenameRequest {
  name: string
}

export interface InterviewResearchSession {
  session_id: string
  role: string
  company: string
  status: string
  job_id: number
  question_bank: InterviewResearchQuestionBank
  fallback_used: boolean
  message: string
  metadata: Record<string, unknown>
  source_urls: string[]
  failure_reason: string | null
  stage: string | null
  processing_ms: number | null
  created_at: string
  updated_at: string
  started_at: string | null
  completed_at: string | null
}

export type InterviewResearchProgressItem = {
  stage: string
  tool: string
  query: string
  status: string
  latency_ms: number
  result_count: number
  rejected_count: number
  error: string
  metadata: Record<string, unknown>
  timestamp: string
}

export interface JobSearchResponse {
  status: string
  request_id: number | null
  source: string
  count: number
  stored_count: number
  message: string
}

export interface JobRescoreResponse {
  status: string
  run_id: string
  source?: string
  only_unscored: boolean
  total_jobs: number
  processed_jobs: number
  scored_count: number
  failed_count: number
  failed_job_ids: number[]
  message: string
}

export interface JobSearchRequest {
  query: string
  role?: string
  job?: string
  location?: string
  source?: string
  years_of_experience?: number | string
  num_records?: number
  return_raw?: boolean
}

export const searchJobs = async (
  request: JobSearchRequest
): Promise<JobSearchResponse> => {
  const { data } = await api.post<JobSearchResponse>('/jobs/search', request)
  return data
}

export const rescoreJobs = async (
  payload: { source?: string; only_unscored?: boolean } = {}
): Promise<JobRescoreResponse> => {
  const { data } = await api.post<JobRescoreResponse>('/jobs/rescore', payload)
  return data
}

export const getRescoreStatus = async (runId: string): Promise<JobRescoreResponse> => {
  const { data } = await api.get<JobRescoreResponse>(`/jobs/rescore/${runId}`)
  return data
}

export const getJobs = async (filters?: {
  min_score?: number
  max_score?: number
  score_status?: 'all' | 'scored' | 'unscored'
  work_type?: string
  seniority?: string
  source?: string
  sort_by?: string
  sort_dir?: 'asc' | 'desc'
  limit?: number
}): Promise<Job[]> => {
  const { data } = await api.get<Job[]>('/jobs/', { params: filters })
  return data
}

export const getJob = async (id: number): Promise<Job> => {
  const { data } = await api.get<Job>(`/jobs/${id}`)
  return data
}

export interface JobAnalysisResult {
  job_id: number
  run_id: string
  run_status: string
  run_state: string
  fit_score?: number
  matched_keywords: string[]
  missing_keywords: string[]
  gap_analysis?: string
  reason?: string
  rewrite_suggestions: string[]
  matched_keyword_evidence: ScoreEvidenceRecord[]
  missing_keyword_evidence: ScoreEvidenceRecord[]
  rewrite_suggestion_evidence: ScoreEvidenceRecord[]
  agent_plan?: ScoreAgentPlan
  failure_reason?: string
  failed_step?: string
}

export const analyzeJob = async (jobId: number): Promise<JobAnalysisResult> => {
  const { data } = await api.post<JobAnalysisResult>(`/jobs/${jobId}/analyze`, {}, { timeout: 180000 })
  return data
}

export const generateStudyFlashcards = async (request: GenerateFlashcardsRequest): Promise<FlashcardSetResponse> => {
  const { data } = await api.post<FlashcardSetResponse>('/study/flashcards', request, { timeout: 120000 })
  return data
}

export const generateStudyFlashcardsAsync = async (
  request: GenerateFlashcardsRequest,
): Promise<FlashcardGenerationStartResponse> => {
  const { data } = await api.post<FlashcardGenerationStartResponse>('/study/flashcards/async', request)
  return data
}

export const getStudyFlashcardsJob = async (jobId: string): Promise<FlashcardGenerationJobResponse> => {
  const { data } = await api.get<FlashcardGenerationJobResponse>(`/study/flashcards/jobs/${encodeURIComponent(jobId)}`)
  return data
}

export const reviewStudyCard = async (
  cardId: number,
  request: StudyCardReviewRequest,
): Promise<StudyCard> => {
  const { data } = await api.patch<StudyCard>(`/study/cards/${cardId}/review`, request)
  return data
}

export const listStudyCardSets = async (limit = 20): Promise<StudyCardSetSummary[]> => {
  const { data } = await api.get<StudyCardSetSummary[]>('/study/card-sets', { params: { limit } })
  return data
}

export const getStudyCardSet = async (cardSetId: number): Promise<StudyCardSetDetailResponse> => {
  const { data } = await api.get<StudyCardSetDetailResponse>(`/study/card-sets/${cardSetId}`)
  return data
}

export const renameStudyCardSet = async (
  cardSetId: number,
  payload: StudyCardSetRenameRequest,
): Promise<StudyCardSetSummary> => {
  const { data } = await api.patch<StudyCardSetSummary>(`/study/card-sets/${cardSetId}`, payload)
  return data
}

export const deleteStudyCardSet = async (cardSetId: number): Promise<void> => {
  await api.delete(`/study/card-sets/${cardSetId}`)
}

// ── Practice (Phase 1) ─────────────────────────────────────────────────────────
export interface PracticeQuestion {
  id: number
  title: string
  url: string | null
  prompt?: string | null
  difficulty: string | null
  acceptance: string | null
  frequency: string | null
  source_file: string
  source_window: string
  is_ai_generated: boolean
  is_solved?: boolean
}

export interface PracticeQuestionsResponse {
  session_id: string
  company_slug: string
  questions: PracticeQuestion[]
}

export interface PracticeSyncRequest {
  company_slug?: string
  preferred_window?: string
}

export interface PracticeSyncResponse {
  commit: string
  inserted: number
  updated: number
  retired: number
  companies: Array<{
    company_slug: string
    inserted: number
    updated: number
    retired: number
  }>
}

export interface PracticeMarkSolvedRequest {
  session_id: string
  question_id: number
}

export interface PracticeMarkSolvedResponse {
  session_id: string
  question_id: number
  status: string
}

export interface PracticeUnmarkSolvedResponse {
  session_id: string
  question_id: number
  status: string
}

export interface PracticeUnmarkSolvedRequest {
  session_id: string
  question_id: number
}

export interface PracticeDiscardRequest {
  session_id: string
  question_id: number
}

export interface PracticeDiscardResponse {
  session_id: string
  question_id: number
  status: string
}

export interface PracticeNextRequest {
  session_id: string
  solved_question_id: number
  difficulty_delta?: number | null
  language?: string | null
  technique?: string | null
  complexity?: string | null
  time_pressure_minutes?: number | null
  pattern?: string | null
}

export interface PracticeChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface PracticeInterviewChatRequest {
  session_id: string
  question_id: number
  message: string
  language?: string | null
  interview_history?: PracticeChatMessage[]
  solution_text?: string | null
}

export interface PracticeInterviewChatResponse {
  session_id: string
  question_id: number
  interviewer_reply: string
}

export interface ConstraintMetadata {
  difficulty_delta?: number | null
  language?: string | null
  technique?: string | null
  complexity?: string | null
  time_pressure_minutes?: number | null
  pattern?: string | null
}

export interface PracticeNextResponse {
  base_question_id: number
  base_question_link: string
  transformed_prompt: string
  constraint_metadata: ConstraintMetadata
  reason: string
  next_question: PracticeQuestion | null
}

export interface PracticeSessionResponse {
  session_id: string
  company_slug: string
  question_ids: number[]
  status: string
}

export const getPracticeQuestions = async (
  company: string,
  options: {
    job_id: number
    session_id?: string
    difficulty?: string
    source_window?: string
    limit?: number
    includeSolved?: boolean
  },
): Promise<PracticeQuestionsResponse> => {
  const query = {
    job_id: options.job_id,
    session_id: options.session_id,
    difficulty: options.difficulty,
    source_window: options.source_window,
    limit: options.limit,
    include_solved: options.includeSolved,
  }
  const { data } = await api.get<PracticeQuestionsResponse>(`/practice/company/${encodeURIComponent(company)}/questions`, {
    params: query,
  })
  return data
}

export const markPracticeQuestionSolved = async (payload: PracticeMarkSolvedRequest): Promise<PracticeMarkSolvedResponse> => {
  const { data } = await api.post<PracticeMarkSolvedResponse>('/practice/mark-solved', payload)
  return data
}

export const unmarkPracticeQuestionSolved = async (payload: PracticeUnmarkSolvedRequest): Promise<PracticeUnmarkSolvedResponse> => {
  const { data } = await api.post<PracticeUnmarkSolvedResponse>('/practice/unmark-solved', payload)
  return data
}

export const discardPracticeQuestion = async (payload: PracticeDiscardRequest): Promise<PracticeDiscardResponse> => {
  const { data } = await api.post<PracticeDiscardResponse>('/practice/discard', payload)
  return data
}

export const getPracticeNextQuestion = async (payload: PracticeNextRequest): Promise<PracticeNextResponse> => {
  const { data } = await api.post<PracticeNextResponse>('/practice/next', payload)
  return data
}

export const askPracticeInterviewer = async (payload: PracticeInterviewChatRequest): Promise<PracticeInterviewChatResponse> => {
  const { data } = await api.post<PracticeInterviewChatResponse>('/practice/interview-chat', payload)
  return data
}

export const syncPracticeRepo = async (payload: PracticeSyncRequest): Promise<PracticeSyncResponse> => {
  const { data } = await api.post<PracticeSyncResponse>('/practice/sync', payload)
  return data
}

export const getPracticeSession = async (sessionId: string): Promise<PracticeSessionResponse> => {
  const { data } = await api.get<PracticeSessionResponse>(`/practice/sessions/${sessionId}`)
  return data
}

export default api
