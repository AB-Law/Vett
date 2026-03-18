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
  has_anthropic_key: boolean
  has_openai_key: boolean
  has_azure_key: boolean
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

export const buildInterviewResearchStreamUrl = (jobId: number): string =>
  `/api/jobs/${jobId}/interview-research/stream`

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
}

export interface InterviewResearchQuestionBank {
  behavioral: InterviewResearchQuestion[]
  technical: InterviewResearchQuestion[]
  system_design: InterviewResearchQuestion[]
  company_specific: InterviewResearchQuestion[]
  source_urls: string[]
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
