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
  agent_plan?: ScoreAgentPlan
  job_title?: string
  company?: string
  llm_provider?: string
  llm_model?: string
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

export const scoreJD = async (req: ScoreRequest): Promise<ScoreResult> => {
  const { data } = await api.post<ScoreResult>('/score/', req)
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
  created_at: string
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
