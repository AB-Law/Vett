import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { ArrowLeft, AlertCircle, BookOpen, CheckCircle2, ExternalLink, Loader2, MessageCircle, XCircle, Upload, FileText } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { isAxiosError } from 'axios'
import {
  getJob,
  type Job,
  analyzeJob,
  type JobAnalysisResult,
  getPracticeQuestions,
  type PracticeQuestion,
  markPracticeQuestionSolved,
  unmarkPracticeQuestionSolved,
  getPracticeNextQuestion,
  discardPracticeQuestion,
  askPracticeInterviewer,
  getJobInterviewDocuments,
  uploadJobInterviewDocument,
  type InterviewKnowledgeDocument,
  type InterviewResearchProgressItem,
  buildInterviewResearchStreamUrl,
  cancelInterviewResearchSession,
  getInterviewResearchSession,
  type InterviewResearchSession,
} from '../lib/api'
import { formatDate, scoreColor } from '../lib/utils'
import toast from 'react-hot-toast'

type ConstraintSettings = {
  difficultyDelta: 'easier' | 'same' | 'harder' | string
  language: string
  technique: string
  complexity: string
  timePressureMinutes: string
  pattern: string
}

function CitationList({ citations, kind }: { citations: Array<{ section_id?: string; phrase_id?: string; line_start?: number; line_end?: number; snippet?: string }>; kind: 'cv' | 'jd' }) {
  if (!citations || citations.length === 0) return null
  return (
    <div className="mt-1 space-y-1">
      {citations.map((c, i) => (
        <div key={i} className="rounded bg-surface-secondary border border-border px-2 py-1 text-[11px]">
          <span className="font-medium text-text-secondary mr-1">
            {kind === 'cv' ? `CV §${c.section_id || i + 1}` : `JD #${c.phrase_id || i + 1}`}
          </span>
          {(c.line_start || c.line_end) && (
            <span className="text-text-muted mr-1">L{c.line_start}{c.line_end && c.line_end !== c.line_start ? `–${c.line_end}` : ''}</span>
          )}
          {c.snippet && <span className="italic text-text-primary">"{c.snippet}"</span>}
        </div>
      ))}
    </div>
  )
}

function EvidenceBlock({ records, label }: { records: Array<{ value: string; cv_citations: any[]; jd_phrase_citations: any[]; evidence_missing_reason?: string }>; label: string }) {
  if (!records || records.length === 0) return null
  return (
    <div className="mb-4">
      <div className="text-xs font-semibold text-text-primary mb-2">{label}</div>
      <div className="space-y-2">
        {records.map((rec, i) => (
          <div key={i} className="border border-border rounded p-2">
            <div className="text-xs font-medium text-text-primary mb-1">"{rec.value}"</div>
            {rec.cv_citations?.length > 0 && (
              <div>
                <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-0.5">CV evidence</div>
                <CitationList citations={rec.cv_citations} kind="cv" />
              </div>
            )}
            {rec.jd_phrase_citations?.length > 0 && (
              <div className="mt-1">
                <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-0.5">JD evidence</div>
                <CitationList citations={rec.jd_phrase_citations} kind="jd" />
              </div>
            )}
            {rec.evidence_missing_reason && (
              <div className="text-[11px] text-amber-600 mt-1 italic">{rec.evidence_missing_reason}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

const difficultyTagClass = (difficulty: string): string => {
  const normalized = difficulty.toLowerCase()
  if (normalized.includes('easy')) return 'bg-sage-100 text-sage-700 border border-sage-200'
  if (normalized.includes('hard')) return 'bg-red-50 text-red-700 border border-red-200'
  return 'bg-amber-100 text-amber-700 border border-amber-200'
}

const buildCoachUrl = (
  question: PracticeQuestion,
  sessionId: string,
  jobId: number,
  coachLanguage: string,
): string => {
  const params = new URLSearchParams({
    questionId: String(question.id),
    sessionId,
    title: question.title,
    difficulty: question.difficulty || 'unknown',
    language: coachLanguage,
    prompt: question.prompt || '',
    url: question.url || '',
    jobId: String(jobId),
  })
  return `/practice/coach?${params.toString()}`
}

export default function JobDetails() {
  const { id } = useParams<{ id: string }>()
  const [job, setJob] = useState<Job | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState<string>('')
  const [questionsLoading, setQuestionsLoading] = useState(false)
  const [questionRows, setQuestionRows] = useState<PracticeQuestion[]>([])
  const [questionsError, setQuestionsError] = useState('')
  const [followUpPrompt, setFollowUpPrompt] = useState('')
  const [followUpReason, setFollowUpReason] = useState('')
  const [followUpLink, setFollowUpLink] = useState<string>('')
  const [activeGeneratingQuestionId, setActiveGeneratingQuestionId] = useState<number | null>(null)
  const [followUpError, setFollowUpError] = useState('')
  const [followUpGeneratedQuestion, setFollowUpGeneratedQuestion] = useState<PracticeQuestion | null>(null)
  const [questionWindow, setQuestionWindow] = useState<'all' | 'older-than-six-months' | 'six-months' | 'three-months' | 'thirty-days'>(
    'all',
  )
  const [interviewDocuments, setInterviewDocuments] = useState<InterviewKnowledgeDocument[]>([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [uploadingDocument, setUploadingDocument] = useState(false)
  const [prepEventLog, setPrepEventLog] = useState<string[]>([])
  const [prepResearchSession, setPrepResearchSession] = useState<InterviewResearchSession | null>(null)
  const [prepInterviewLoading, setPrepInterviewLoading] = useState(false)
  const [prepInterviewError, setPrepInterviewError] = useState('')
  const [prepProgressTimeline, setPrepProgressTimeline] = useState<InterviewResearchProgressItem[]>([])
  const [prepActiveSessionId, setPrepActiveSessionId] = useState<string>('')
  const [prepLiveQuestionSnapshot, setPrepLiveQuestionSnapshot] = useState<{
    behavioral: number
    technical: number
    system_design: number
    company_specific: number
  } | null>(null)
  const prepResearchEventSourceRef = useRef<EventSource | null>(null)
  const isStoppingPrepInterviewStreamRef = useRef(false)

  const getQuestionSourceWindow = (windowFilter: typeof questionWindow): string => {
    if (windowFilter === 'older-than-six-months') {
      return 'one-year'
    }
    return windowFilter
  }
  const [analysisResult, setAnalysisResult] = useState<JobAnalysisResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState('')

  const [constraints, setConstraints] = useState<ConstraintSettings>({
    difficultyDelta: 'same',
    language: '',
    technique: '',
    complexity: '',
    timePressureMinutes: '',
    pattern: '',
  })

  useEffect(() => {
    const numericId = id ? Number(id) : Number.NaN

    if (!Number.isInteger(numericId) || numericId <= 0) {
      setError('Invalid job id')
      setLoading(false)
      return
    }

    const fetchJob = async () => {
      setLoading(true)
      setError('')
      try {
        const found = await getJob(numericId)
        setJob(found)
      } catch {
        setError('Could not load job details')
        toast.error('Could not load job details', { position: 'top-right' })
      } finally {
        setLoading(false)
      }
    }

    void fetchJob()
  }, [id])

  useEffect(() => {
    let cancelled = false

    const fetchQuestions = async () => {
      if (!job?.company) {
        setQuestionRows([])
        setQuestionsError('')
        setQuestionsLoading(false)
        return
      }

      setQuestionsLoading(true)
      setQuestionsError('')
      try {
        const response = await getPracticeQuestions(job.company, {
          job_id: job.id,
          session_id: sessionId || undefined,
          source_window: getQuestionSourceWindow(questionWindow),
          limit: 8,
          includeSolved: true,
        })
        if (!cancelled) {
          setSessionId(response.session_id)
          setQuestionRows(response.questions)
        }
      } catch (error: unknown) {
        if (cancelled) return
        setQuestionRows([])
        if (error instanceof Error && error.message.includes('No company folder')) {
          setQuestionsError('No company entry found in the linked interview-questions dataset yet.')
        } else {
          setQuestionsError('Failed to load interview questions right now.')
        }
      } finally {
        if (!cancelled) {
          setQuestionsLoading(false)
        }
      }
    }

    void fetchQuestions()
    return () => {
      cancelled = true
    }
  }, [job?.company, job?.id, sessionId, questionWindow])

  const refreshQuestions = async (): Promise<void> => {
    if (!job?.company || !job.id) return
    setQuestionsLoading(true)
    try {
      const response = await getPracticeQuestions(job.company, {
        job_id: job.id,
        session_id: sessionId || undefined,
        source_window: getQuestionSourceWindow(questionWindow),
        limit: 8,
        includeSolved: true,
      })
      setSessionId(response.session_id)
      setQuestionRows(response.questions)
      setQuestionsError('')
    } catch (error: unknown) {
      setQuestionsError(error instanceof Error ? error.message : 'Failed to load interview questions right now.')
    } finally {
      setQuestionsLoading(false)
    }
  }

  const stopInterviewResearchStream = (): void => {
    isStoppingPrepInterviewStreamRef.current = true
    if (prepResearchEventSourceRef.current) {
      prepResearchEventSourceRef.current.close()
      prepResearchEventSourceRef.current = null
    }
  }

  const fetchResearchSession = async (sessionId: string): Promise<void> => {
    if (!job) return
    try {
      const session = await getInterviewResearchSession(job.id, sessionId)
      setPrepResearchSession(session)
      const metadata = session.metadata as Record<string, unknown>
      const timeline = metadata?.progress_timeline
      if (Array.isArray(timeline)) {
        setPrepProgressTimeline(timeline as InterviewResearchProgressItem[])
      }
      setPrepInterviewError('')
      await refreshQuestions()
    } catch (error: unknown) {
      setPrepInterviewError(error instanceof Error ? error.message : 'Could not load interview research results')
    }
  }

  const handleCancelPrepInterview = async (): Promise<void> => {
    if (!job?.id || !prepActiveSessionId) return
    try {
      const response = await cancelInterviewResearchSession(job.id, prepActiveSessionId)
      if (response.status !== 'cancel_requested') {
        throw new Error(
          response.status === 'not_found'
            ? 'Interview prep run is no longer active.'
            : `Unexpected cancel response: ${response.status}`,
        )
      }
      setPrepInterviewError('Cancellation requested. Waiting for server acknowledgement…')
      stopInterviewResearchStream()
      setPrepInterviewLoading(false)
    } catch (error: unknown) {
      setPrepInterviewError(error instanceof Error ? error.message : 'Could not cancel interview prep run')
    }
  }

  const handlePrepInterview = (): void => {
    if (!job) return
    stopInterviewResearchStream()
    isStoppingPrepInterviewStreamRef.current = false
    setPrepInterviewLoading(true)
    setPrepInterviewError('')
    setPrepEventLog([])
    setPrepResearchSession(null)
    setPrepProgressTimeline([])
    setPrepActiveSessionId('')
    setPrepLiveQuestionSnapshot(null)

    const eventSource = new EventSource(buildInterviewResearchStreamUrl(job.id))
    prepResearchEventSourceRef.current = eventSource

    eventSource.onmessage = (event: MessageEvent<string>): void => {
      try {
        const payload = JSON.parse(event.data) as {
          type?: string
          stage?: string
          message?: string
          status?: string
          tool?: string
          query?: string
          latency_ms?: number
          result_count?: number
          rejected_count?: number
          error?: string
          timestamp?: string
          payload?: { session_id?: string; metadata?: Record<string, unknown> }
        }
        const message = payload.message || `${payload.type || 'status'}${payload.stage ? ` (${payload.stage})` : ''}`
        const nextSessionId = payload.payload?.session_id || (payload as { session_id?: string }).session_id
        if (nextSessionId) {
          setPrepActiveSessionId(nextSessionId)
        }
        if (payload.type === 'status') {
          const statusEvent: InterviewResearchProgressItem = {
            stage: payload.stage || '',
            tool: payload.tool || '',
            query: payload.query || '',
            status: payload.status || 'ok',
            latency_ms: payload.latency_ms || 0,
            result_count: payload.result_count || 0,
            rejected_count: payload.rejected_count || 0,
            error: payload.error || '',
            metadata: {},
            timestamp: payload.timestamp || new Date().toISOString(),
          }
          setPrepProgressTimeline((current) => [...current, statusEvent].slice(-60))
        }
        setPrepEventLog((current) => [...current, message].slice(-20))
        if (payload.type === 'done' && nextSessionId) {
          const doneMetadata = payload.payload?.metadata as Record<string, unknown> | undefined
          const snapshot = doneMetadata?.question_bank_snapshot as
            | { behavioral?: number; technical?: number; system_design?: number; company_specific?: number }
            | undefined
          if (snapshot) {
            setPrepLiveQuestionSnapshot({
              behavioral: snapshot.behavioral || 0,
              technical: snapshot.technical || 0,
              system_design: snapshot.system_design || 0,
              company_specific: snapshot.company_specific || 0,
            })
          }
          void fetchResearchSession(nextSessionId)
          setPrepInterviewLoading(false)
          stopInterviewResearchStream()
        }
        if (payload.type === 'error') {
          setPrepInterviewError(payload.message || 'Interview research failed.')
          setPrepInterviewLoading(false)
          stopInterviewResearchStream()
        }
      } catch {
        setPrepEventLog((current) => [...current, 'Received malformed progress update.'].slice(-20))
      }
    }

    eventSource.onerror = (): void => {
      if (isStoppingPrepInterviewStreamRef.current) {
        isStoppingPrepInterviewStreamRef.current = false
        return
      }
      setPrepInterviewError('Interview prep stream disconnected. Try again.')
      setPrepInterviewLoading(false)
      stopInterviewResearchStream()
    }
  }

  const markSolved = async (questionId: number): Promise<void> => {
    if (!sessionId || !job) return
    const solvedQuestion = questionRows.find((question) => question.id === questionId) ?? null
    try {
      await markPracticeQuestionSolved({ session_id: sessionId, question_id: questionId })
      setQuestionRows((current) =>
        current.map((question) => (question.id === questionId ? { ...question, is_solved: true } : question)),
      )
      if (followUpGeneratedQuestion?.id === questionId) {
        setFollowUpGeneratedQuestion(null)
      }
      if (solvedQuestion) {
        await generateFollowUpTwist(solvedQuestion)
      }
      await refreshQuestions()
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Could not mark question as solved'
      toast.error(msg, { position: 'top-right' })
    }
  }

  const unmarkSolved = async (questionId: number): Promise<void> => {
    if (!sessionId) return
    try {
      await unmarkPracticeQuestionSolved({ session_id: sessionId, question_id: questionId })
      setQuestionRows((current) =>
        current.map((question) => (question.id === questionId ? { ...question, is_solved: false } : question)),
      )
      await refreshQuestions()
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Could not restore solved question'
      toast.error(msg, { position: 'top-right' })
    }
  }

  const generateFollowUpTwist = async (question: PracticeQuestion): Promise<void> => {
    if (activeGeneratingQuestionId !== null) return
    if (!sessionId || !job) return
    setActiveGeneratingQuestionId(question.id)
    setFollowUpError('')
    setFollowUpGeneratedQuestion(null)
    try {
      const jdContext = job.gap_analysis ? `\nJD context: ${job.gap_analysis}` : '\nJD context: N/A'
      const response = await askPracticeInterviewer({
        session_id: sessionId,
        question_id: question.id,
        message:
          `User has solved "${question.title}" for ${job.company || 'this company'} role ` +
          `and is preparing for ${job.title || 'the selected role'}.${jdContext}\n` +
          'Prepare one high-value interview twist/variant that might be asked as a follow-up.',
        language: constraints.language || null,
        interview_history: [],
      })
      setFollowUpLink(question.url || '')
      setFollowUpPrompt(response.interviewer_reply)
      setFollowUpReason(`Twist prepared for ${question.title} (role: ${job.title || 'N/A'}).`)
    } catch {
      setFollowUpError('Could not generate the interview twist right now.')
      setFollowUpPrompt('')
      setFollowUpReason('')
    } finally {
      setActiveGeneratingQuestionId(null)
    }
  }

  const mapDifficultyDelta = (value: ConstraintSettings['difficultyDelta']): number | null => {
    if (value === 'same') return 0
    if (value === 'easier') return -1
    if (value === 'harder') return 1
    if (value === 'much-harder') return 2
    if (value === 'much-easier') return -2
    return null
  }

  const generateFollowUp = async (questionId: number): Promise<void> => {
    if (activeGeneratingQuestionId !== null) return
    if (!sessionId || !job) return
    setActiveGeneratingQuestionId(questionId)
    setFollowUpError('')
    setFollowUpGeneratedQuestion(null)

    try {
      const payload = {
        session_id: sessionId,
        solved_question_id: questionId,
        difficulty_delta: mapDifficultyDelta(constraints.difficultyDelta),
        language: constraints.language || null,
        technique: constraints.technique || null,
        complexity: constraints.complexity || null,
        time_pressure_minutes: Number.parseInt(constraints.timePressureMinutes, 10) || null,
        pattern: constraints.pattern || null,
      }

      const result = await getPracticeNextQuestion(payload)
      setFollowUpLink(result.base_question_link)
      setFollowUpPrompt(result.transformed_prompt)
      setFollowUpReason(result.reason)
      setFollowUpGeneratedQuestion(result.next_question ?? null)
      await refreshQuestions()
      setActiveGeneratingQuestionId(null)
    } catch (error: unknown) {
      if (isAxiosError(error) && error.response?.data && typeof error.response.data === 'object') {
        const detail = (error.response.data as { detail?: string }).detail
        setFollowUpError(detail || 'Could not generate the next practice prompt')
      } else {
        setFollowUpError(error instanceof Error ? error.message : 'Could not generate the next practice prompt')
      }
      setActiveGeneratingQuestionId(null)
      toast.error('Could not generate follow-up prompt', { position: 'top-right' })
    } finally {
      setActiveGeneratingQuestionId(null)
    }
  }

  const handleAnalyze = async (): Promise<void> => {
    if (!job) return
    setAnalyzing(true)
    setAnalysisError('')
    setAnalysisResult(null)
    try {
      const result = await analyzeJob(job.id)
      setAnalysisResult(result)
      // Refresh job to pick up updated fit_score etc
      const updated = await getJob(job.id)
      setJob(updated)
    } catch {
      setAnalysisError('Deep analysis failed. Please try again.')
    } finally {
      setAnalyzing(false)
    }
  }

  const updateConstraint = (key: keyof ConstraintSettings, value: string): void => {
    setConstraints((current) => ({ ...current, [key]: value }))
  }

  const discardQuestion = async (questionId: number): Promise<void> => {
    if (!sessionId) {
      return
    }
    try {
      await discardPracticeQuestion({ session_id: sessionId, question_id: questionId })
      await refreshQuestions()
      if (followUpGeneratedQuestion?.id === questionId) {
        setFollowUpGeneratedQuestion(null)
      }
      toast.success('Question hidden from your current session.', { position: 'top-right' })
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Could not hide question'
      toast.error(msg, { position: 'top-right' })
    }
  }

  const documentStatusClass = (status: string): string => {
    const normalized = status.toLowerCase()
    if (normalized.includes('embed')) {
      return 'bg-sage-100 text-sage-700 border border-sage-200'
    }
    if (normalized.includes('processing')) {
      return 'bg-amber-100 text-amber-700 border border-amber-200'
    }
    if (normalized.includes('failed')) {
      return 'bg-red-100 text-red-700 border border-red-200'
    }
    return 'bg-gray-100 text-text-muted border border-border'
  }

  const handleInterviewDocsDrop = useCallback(
    async (files: File[]) => {
      const file = files[0]
      if (!file || !job?.id) return
      setUploadingDocument(true)
      try {
        const newDoc = await uploadJobInterviewDocument(job.id, file)
        setInterviewDocuments((current) => [newDoc, ...current])
        toast.success('Interview document uploaded')
      } catch (err: unknown) {
        const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Upload failed'
        toast.error(detail)
      } finally {
        setUploadingDocument(false)
      }
    },
    [job?.id],
  )

  const {
    getRootProps: getInterviewDocsRootProps,
    getInputProps: getInterviewDocsInputProps,
    isDragActive: isInterviewDocsDragActive,
  } = useDropzone({
    onDrop: handleInterviewDocsDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/markdown': ['.md', '.markdown'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    disabled: uploadingDocument,
  })

  useEffect(() => {
    if (!job?.id) {
      setInterviewDocuments([])
      return
    }

    let cancelled = false

    const fetchDocuments = async () => {
      setDocumentsLoading(true)
      try {
        const docs = await getJobInterviewDocuments(job.id)
        if (!cancelled) {
          setInterviewDocuments(docs)
        }
      } catch {
        if (!cancelled) {
          setInterviewDocuments([])
        }
      } finally {
        if (!cancelled) {
          setDocumentsLoading(false)
        }
      }
    }

    void fetchDocuments()
    return () => {
      cancelled = true
    }
  }, [job?.id])

  useEffect(() => {
    return () => {
      stopInterviewResearchStream()
    }
  }, [])

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="flex items-center justify-center py-16 text-sm text-text-muted">Loading job details…</div>
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="card p-6 text-sm text-text-muted">{error || 'Job not found'}</div>
        <Link to="/jobs" className="inline-flex items-center gap-2 mt-3 text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to jobs
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      <div className="mb-6">
        <Link to="/jobs" className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to jobs
        </Link>
      </div>

      <div className="card p-5">
        <div className="mb-4">
          <h1 className="text-2xl font-semibold text-text-primary">{job.title || 'Untitled job'}</h1>
          <p className="text-sm text-text-secondary mt-1">{job.company || '—'}</p>
        </div>

        {typeof job.fit_score === 'number' && (
          <div className="flex items-center gap-2 mb-4">
            <span className="text-2xl font-bold text-text-primary">{Math.round(job.fit_score)}%</span>
            <span className={scoreColor(job.fit_score)}>fit score</span>
          </div>
        )}

        {(job.matched_keywords ?? []).length > 0 && (
          <div className="mb-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-sage-500" />
              <span className="text-xs font-semibold text-text-primary">Matched keywords</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {(job.matched_keywords ?? []).map((kw) => (
                <span key={kw} className="keyword-matched text-xs">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}

        {(job.missing_keywords ?? []).length > 0 && (
          <div className="mb-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <XCircle className="w-3.5 h-3.5 text-red-400" />
              <span className="text-xs font-semibold text-text-primary">Missing keywords</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {(job.missing_keywords ?? []).map((kw) => (
                <span key={kw} className="keyword-missing text-xs">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}

        {job.gap_analysis && (
          <div className="mb-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <BookOpen className="w-3.5 h-3.5 text-text-secondary" />
              <span className="text-xs font-semibold text-text-primary">Gap Analysis</span>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed">{job.gap_analysis}</p>
          </div>
        )}

        {job.reason && (
          <div className="mb-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-text-secondary" />
              <span className="text-xs font-semibold text-text-primary">Why this score</span>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed">{job.reason}</p>
          </div>
        )}

        <div className="card p-4 mb-4">
          <div className="flex items-center gap-2 mb-3">
            <Upload className="w-3.5 h-3.5 text-text-secondary" />
            <span className="text-xs font-semibold text-text-primary">Interview Documents</span>
          </div>
          <div
            {...getInterviewDocsRootProps()}
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              isInterviewDocsDragActive ? 'border-sage-400 bg-sage-50' : 'border-border hover:border-sage-300 hover:bg-gray-50'
            } ${uploadingDocument ? 'opacity-60 pointer-events-none' : ''}`}
          >
            <input {...getInterviewDocsInputProps()} />
            <FileText className="w-6 h-6 text-text-muted mx-auto mb-2" />
            <p className="text-xs font-medium text-text-secondary mb-1">
              {isInterviewDocsDragActive ? 'Drop interview docs here…' : 'Drop or upload PDF / DOCX / DOC / MD / TXT'}
            </p>
            <p className="text-[11px] text-text-muted">These documents are used as job-level interview context.</p>
            {uploadingDocument && <p className="text-xs text-sage-600 mt-2 font-medium">Uploading…</p>}
          </div>
          <div className="mt-3">
            {documentsLoading ? (
              <p className="text-xs text-text-muted">Loading interview documents…</p>
            ) : interviewDocuments.length === 0 ? (
              <p className="text-xs text-text-muted">No job-specific interview documents uploaded yet.</p>
            ) : (
              <div className="space-y-2">
                {interviewDocuments.map((doc) => (
                  <div key={doc.id} className="border border-border rounded p-2">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <div className="text-text-primary font-medium truncate">{doc.source_filename}</div>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] ${documentStatusClass(doc.status)}`}>
                        {doc.status}
                      </span>
                    </div>
                    {doc.error_message && <p className="text-[11px] text-red-600 mt-1">{doc.error_message}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Deep analysis trigger */}
        <div className="mt-3 pt-3 border-t border-border">
          <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => void handlePrepInterview()}
                disabled={prepInterviewLoading}
                className="btn-primary flex items-center gap-2 text-xs py-1.5 px-3 disabled:opacity-60"
              >
                {prepInterviewLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
                {prepInterviewLoading ? 'Preparing interview questions…' : 'Prep Interview'}
              </button>
              {job?.id ? (
                <Link
                  to={`/jobs/${job.id}/interview-prep`}
                  className="rounded border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 px-3 py-1.5 text-xs"
                >
                  Start Interview Prep
                </Link>
              ) : null}
            </div>
            {prepInterviewLoading ? (
              <button
                type="button"
                onClick={() => void handleCancelPrepInterview()}
                className="rounded border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 px-3 py-1.5 text-xs"
              >
                Cancel run
              </button>
            ) : null}
            {prepResearchSession?.status ? <span className="text-[11px] text-text-muted">Last status: {prepResearchSession.status}</span> : null}
          </div>

          {prepInterviewError && <p className="text-xs text-red-500 mb-2">{prepInterviewError}</p>}

          {prepEventLog.length > 0 ? (
            <div className="mb-2 text-xs text-text-muted space-y-1">
              {prepEventLog.map((eventLine, index) => (
                <p key={`${eventLine}-${index}`} className="leading-relaxed">
                  {eventLine}
                </p>
              ))}
            </div>
          ) : null}

          {prepProgressTimeline.length > 0 ? (
            <div className="mb-2 border border-border rounded p-2">
              <div className="text-[11px] font-semibold text-text-primary mb-1">Structured timeline</div>
              <div className="space-y-1 max-h-44 overflow-auto">
                {prepProgressTimeline.slice(-12).map((item, idx) => (
                  <div key={`${item.stage}-${idx}-${item.timestamp}`} className="text-[11px] text-text-secondary flex items-start justify-between gap-2">
                    <span>
                      <span className="font-medium text-text-primary">{item.stage || 'status'}</span>
                      {item.tool ? ` · ${item.tool}` : ''}
                      {item.status ? ` · ${item.status}` : ''}
                    </span>
                    <span className="text-text-muted">{item.result_count > 0 ? `${item.result_count} hits` : ''}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className="mt-3 pt-3 border-t border-border">
          {!analysisResult && (
            <button
              type="button"
              onClick={() => void handleAnalyze()}
              disabled={analyzing}
              className="btn-primary flex items-center gap-2 text-xs py-1.5 px-3 disabled:opacity-60"
            >
              {analyzing ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
              {analyzing ? 'Running deep analysis…' : 'Run deep analysis'}
            </button>
          )}
          {analyzing && (
            <p className="text-xs text-text-muted mt-2">This may take 30–60 seconds. Role analysis → scoring → action plan.</p>
          )}
          {analysisError && (
            <p className="text-xs text-red-500 mt-2">{analysisError}</p>
          )}
        </div>

        {/* Deep analysis results */}
        {analysisResult && (
          <div className="mt-4 pt-3 border-t border-border space-y-4">
            <div className="text-xs font-semibold text-text-primary flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5" />
              Deep analysis results
              <button
                type="button"
                onClick={() => void handleAnalyze()}
                disabled={analyzing}
                className="ml-auto text-[10px] text-text-muted hover:text-text-primary disabled:opacity-50"
              >
                {analyzing ? 'Re-running…' : 'Re-run'}
              </button>
            </div>

            {/* Rewrite suggestions */}
            {analysisResult.rewrite_suggestions?.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-primary mb-1.5">Rewrite suggestions</div>
                <ul className="space-y-1">
                  {analysisResult.rewrite_suggestions.map((s, i) => (
                    <li key={i} className="text-xs text-text-secondary flex gap-2">
                      <span className="text-text-muted shrink-0">{i + 1}.</span>
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Agent plan */}
            {analysisResult.agent_plan && (
              <div>
                <div className="text-xs font-semibold text-text-primary mb-1.5">Action plan</div>
                {analysisResult.agent_plan.skills_to_fix_first?.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-1">Skills to fix first</div>
                    <div className="flex flex-wrap gap-1">
                      {analysisResult.agent_plan.skills_to_fix_first.map((s: string, i: number) => (
                        <span key={i} className="keyword-missing text-xs">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
                {analysisResult.agent_plan.concrete_edit_actions?.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-1">CV edits</div>
                    <ul className="space-y-1">
                      {analysisResult.agent_plan.concrete_edit_actions.map((a: string, i: number) => (
                        <li key={i} className="text-xs text-text-secondary flex gap-2">
                          <span className="text-text-muted shrink-0">{i + 1}.</span>{a}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {analysisResult.agent_plan.interview_topics_to_prioritize?.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-1">Interview topics</div>
                    <div className="flex flex-wrap gap-1">
                      {analysisResult.agent_plan.interview_topics_to_prioritize.map((t: string, i: number) => (
                        <span key={i} className="keyword-matched text-xs">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
                {analysisResult.agent_plan.study_order?.length > 0 && (
                  <div>
                    <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-1">Study order</div>
                    <ol className="space-y-0.5">
                      {analysisResult.agent_plan.study_order.map((s: string, i: number) => (
                        <li key={i} className="text-xs text-text-secondary flex gap-2">
                          <span className="text-text-muted shrink-0 font-medium">{i + 1}.</span>{s}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}

            {/* Evidence sections */}
            <EvidenceBlock
              records={analysisResult.matched_keyword_evidence}
              label="Matched keyword evidence"
            />
            <EvidenceBlock
              records={analysisResult.missing_keyword_evidence}
              label="Missing keyword evidence"
            />
            <EvidenceBlock
              records={analysisResult.rewrite_suggestion_evidence}
              label="Rewrite suggestion evidence"
            />
          </div>
        )}

        {prepResearchSession && (
          <div className="mb-4 border-t border-border pt-3">
            <div className="text-xs font-semibold text-text-primary mb-2">AI-prepared interview research</div>
            <p className="text-[11px] text-text-muted mb-2">
              {prepResearchSession.message}
            </p>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 mb-2">
              <div className="p-2 border border-border rounded text-[11px]">
                Behavioral: {prepLiveQuestionSnapshot?.behavioral ?? prepResearchSession.question_bank.behavioral.length}
              </div>
              <div className="p-2 border border-border rounded text-[11px]">
                Technical: {prepLiveQuestionSnapshot?.technical ?? prepResearchSession.question_bank.technical.length}
              </div>
              <div className="p-2 border border-border rounded text-[11px]">
                System design: {prepLiveQuestionSnapshot?.system_design ?? prepResearchSession.question_bank.system_design.length}
              </div>
              <div className="p-2 border border-border rounded text-[11px]">
                Company specific: {prepLiveQuestionSnapshot?.company_specific ?? prepResearchSession.question_bank.company_specific.length}
              </div>
            </div>
            <div className="space-y-3 mb-3">
              {[
                { label: 'Behavioral', items: prepResearchSession.question_bank.behavioral },
                { label: 'Technical', items: prepResearchSession.question_bank.technical },
                { label: 'System design', items: prepResearchSession.question_bank.system_design },
                { label: 'Company specific', items: prepResearchSession.question_bank.company_specific },
              ].map(({ label, items }) => {
                const questions = items
                if (!questions.length) return null
                return (
                  <div key={label}>
                    <div className="text-[11px] font-semibold text-text-primary mb-1">{label}</div>
                    <ul className="space-y-1.5">
                      {questions.map((item, index) => (
                        <li key={`${label}-${index}-${item.source_url}`} className="text-[11px] text-text-secondary">
                          <div className="text-text-primary">
                            {item.question_text || item.question}
                          </div>
                          {(item.reason || item.source_title) && (
                            <div className="text-[10px] text-text-muted">
                              {[item.reason, item.source_title].filter(Boolean).join(' · ')}
                            </div>
                          )}
                          {item.citations?.length ? (
                            <div className="mt-1 space-y-1">
                              {item.citations.slice(0, 2).map((citation, cidx) => (
                                <div key={`${citation.source_url}-${cidx}`} className="text-[10px] text-text-muted border border-border rounded px-1.5 py-1">
                                  <div className="font-medium text-text-secondary">
                                    {citation.source_title || citation.source_url || 'Source'}
                                  </div>
                                  {citation.snippet ? <div className="italic">"{citation.snippet}"</div> : null}
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                )
              })}
            </div>
            {prepResearchSession.question_bank.source_urls.length > 0 && (
              <div className="text-xs">
                <div className="font-semibold text-text-primary mb-1">Provenance</div>
                <ul className="space-y-1">
                  {prepResearchSession.question_bank.source_urls.map((url) => (
                    <li key={url}>
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sky-700 underline break-all"
                      >
                        {url}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {job.company && (
          <div className="mb-4 border-t border-border pt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <BookOpen className="w-3.5 h-3.5 text-text-secondary" />
              <span className="text-xs font-semibold text-text-primary">Interview questions to practice</span>
            </div>

            <div className="card p-3 mb-3">
              <div className="text-xs text-text-secondary font-semibold mb-2">Practice constraints</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                <select
                  className="field-input"
                  value={constraints.difficultyDelta}
                  onChange={(event) => updateConstraint('difficultyDelta', event.target.value)}
                >
                  <option value="much-easier">Much easier</option>
                  <option value="easier">Easier</option>
                  <option value="same">Same</option>
                  <option value="harder">Harder</option>
                  <option value="much-harder">Much harder</option>
                </select>
                <input
                  className="field-input"
                  value={constraints.language}
                  onChange={(event) => updateConstraint('language', event.target.value)}
                  placeholder="Language (eg. Python)"
                />
                <input
                  className="field-input"
                  value={constraints.technique}
                  onChange={(event) => updateConstraint('technique', event.target.value)}
                  placeholder="Technique (eg. hashmap)"
                />
                <input
                  className="field-input"
                  value={constraints.complexity}
                  onChange={(event) => updateConstraint('complexity', event.target.value)}
                  placeholder="Complexity (eg. O(n))"
                />
                <input
                  className="field-input"
                  value={constraints.timePressureMinutes}
                  type="number"
                  min="1"
                  onChange={(event) => updateConstraint('timePressureMinutes', event.target.value)}
                  placeholder="Time pressure minutes"
                />
                <input
                  className="field-input"
                  value={constraints.pattern}
                  onChange={(event) => updateConstraint('pattern', event.target.value)}
                  placeholder="Pattern / scenario / scale (eg. two-pointers, 10m users)"
                />
              </div>
            </div>
            <div className="card p-3 mb-3">
              <div className="text-xs text-text-secondary font-semibold mb-2">Question window</div>
              <select
                className="field-input"
                value={questionWindow}
                onChange={(event) =>
                  setQuestionWindow(
                    event.target.value as
                      | 'all'
                      | 'older-than-six-months'
                      | 'six-months'
                      | 'three-months'
                      | 'thirty-days',
                  )
                }
              >
                <option value="all">All</option>
                <option value="older-than-six-months">More than 6 months</option>
                <option value="six-months">6 months</option>
                <option value="three-months">3 months</option>
                <option value="thirty-days">30 days</option>
              </select>
            </div>

            {questionsLoading ? (
              <p className="text-xs text-text-muted flex items-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" />
                Loading company question set from the local practice store…
              </p>
            ) : (
              <div>
                <div className="space-y-2">
                  {questionRows.length > 0 ? (
                    questionRows.map((question) => {
                      return (
                        <div key={`${question.id}-${question.url}`} className="border rounded border-border p-2">
                          <a
                            href={question.url || '#'}
                            target={question.url ? '_blank' : undefined}
                            rel={question.url ? 'noopener noreferrer' : undefined}
                            className="font-medium text-sky-700 hover:text-sky-900 text-xs block"
                          >
                            {question.title}
                          </a>
                          <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-text-muted">
                            <span className={`px-1.5 py-0.5 rounded-full ${difficultyTagClass(question.difficulty || '')}`}>
                              {question.difficulty || 'Unknown'}
                            </span>
                            {question.is_ai_generated && (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded-full bg-purple-100 text-purple-700">
                                AI Generated
                              </span>
                            )}
                            {question.acceptance && <span>Acceptance: {question.acceptance}</span>}
                            {question.frequency && <span>Asked: {question.frequency}</span>}
                            <span className="ml-auto text-[10px] text-text-muted">source: {question.source_window}</span>
                          </div>
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            <button
                              type="button"
                              onClick={() => {
                                if (question.is_solved) return
                                void markSolved(question.id)
                              }}
                              disabled={activeGeneratingQuestionId === question.id || question.is_solved}
                              className={`btn-secondary py-1 px-2 text-[11px] disabled:opacity-60 ${question.is_solved ? 'bg-sage-100 text-sage-700 border border-sage-200' : ''}`}
                            >
                              {question.is_solved
                                ? 'Solved'
                                : activeGeneratingQuestionId === question.id
                                  ? 'Generating twist…'
                                  : 'Mark solved'}
                            </button>
                            <Link
                              to={buildCoachUrl(question, sessionId, job?.id || 0, constraints.language)}
                              className="rounded border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 px-2 py-1 text-[11px]"
                            >
                              <MessageCircle className="inline w-3 h-3 mr-1 align-text-bottom" />
                              Coach this question
                            </Link>
                            {question.is_solved ? (
                              <button
                                type="button"
                                onClick={() => void unmarkSolved(question.id)}
                                disabled={activeGeneratingQuestionId === question.id}
                                className="rounded border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 px-2 py-1 text-[11px]"
                              >
                                Unhide
                              </button>
                            ) : null}
                            <button
                              type="button"
                              onClick={() => void generateFollowUp(question.id)}
                              disabled={activeGeneratingQuestionId !== null}
                              className="btn-primary py-1 px-2 text-[11px] disabled:opacity-60"
                            >
                              {activeGeneratingQuestionId === question.id ? 'Generating…' : 'Generate next'}
                            </button>
                            {question.is_ai_generated && (
                              <button
                                type="button"
                                onClick={() => void discardQuestion(question.id)}
                                className="rounded border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 px-2 py-1 text-[11px]"
                              >
                                Delete
                              </button>
                            )}
                          </div>
                        </div>
                      )
                    })
                  ) : questionsError ? (
                    <p className="text-xs text-text-muted">{questionsError}</p>
                  ) : (
                    <p className="text-xs text-text-muted">
                      No interview questions found for <span className="font-semibold text-text-primary">{job.company}</span> in this dataset yet.
                    </p>
                  )}

                  {(followUpError || followUpPrompt || followUpGeneratedQuestion) && (
                    <div className="mt-3 card p-3 text-xs text-text-secondary">
                      {followUpError ? (
                        <p className="text-red-600">{followUpError}</p>
                      ) : (
                        <>
                          <p className="font-semibold text-text-primary mb-1">Generated Interview Twist</p>
                          {followUpLink && (
                            <p className="mb-2 text-text-muted">
                                Solved question link:{' '}
                              <a href={followUpLink} target="_blank" rel="noopener noreferrer" className="text-sky-700 underline">
                                open
                              </a>
                            </p>
                          )}
                          {followUpReason && <p className="text-[11px] text-text-muted mb-2">{followUpReason}</p>}
                          <p className="leading-relaxed text-[13px]">{followUpPrompt}</p>
                          {followUpGeneratedQuestion?.title ? (
                            <div className="mt-2 border-t border-border pt-2">
                              <div className="flex items-center gap-1.5 text-text-secondary">
                                <span className="inline-flex items-center rounded-full bg-purple-100 text-purple-700 px-2 py-0.5">
                                  AI-generated
                                </span>
                                <span className="text-text-secondary">Next question candidate: {followUpGeneratedQuestion.title}</span>
                              </div>
                              {followUpGeneratedQuestion.url && (
                                <p className="mt-1">
                                  <span className="text-text-muted">Candidate link:</span>{' '}
                                  <a
                                    href={followUpGeneratedQuestion.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-sky-700 underline"
                                  >
                                    open
                                  </a>
                                </p>
                              )}
                              {followUpGeneratedQuestion.difficulty && (
                                <p className="mt-1">
                                  Difficulty: {followUpGeneratedQuestion.difficulty}
                                </p>
                              )}
                              <div className="mt-2 flex flex-wrap gap-2">
                                <button
                                  type="button"
                                  onClick={() => void generateFollowUpTwist(followUpGeneratedQuestion)}
                                  disabled={activeGeneratingQuestionId !== null}
                                  className="rounded border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 px-2 py-1 text-[11px]"
                                >
                                  {activeGeneratingQuestionId !== null ? 'Generating…' : 'Prepare another twist'}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void discardQuestion(followUpGeneratedQuestion.id)}
                                  className="rounded border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 px-2 py-1 text-[11px]"
                                >
                                  Delete generated question
                                </button>
                              </div>
                            </div>
                          ) : null}
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="mb-4 border-t border-border pt-3">
          <div className="text-xs text-text-secondary">
            <p>
              <span className="font-semibold text-text-primary">Posted:</span>{' '}
              {job.posted_at ? formatDate(job.posted_at) : job.posted_at_raw || '—'}
            </p>
            <p>
              <span className="font-semibold text-text-primary">Employment:</span>{' '}
              {job.employment_type || job.work_type || '—'}
            </p>
            <p>
              <span className="font-semibold text-text-primary">Job function:</span> {job.job_function || '—'}
            </p>
            <p>
              <span className="font-semibold text-text-primary">Industries:</span> {job.industries || '—'}
            </p>
            {job.applicants_count && (
              <p>
                <span className="font-semibold text-text-primary">Applicants:</span> {job.applicants_count}
              </p>
            )}
            {job.salary && (
              <p>
                <span className="font-semibold text-text-primary">Salary:</span> {job.salary}
              </p>
            )}
          </div>

          {job.external_job_id && <p className="text-[10px] text-text-muted mt-2 break-all">External ID: {job.external_job_id}</p>}

          {job.company_address && (
            <p className="text-[10px] text-text-muted mt-1">
              {[
                job.company_address['addressLocality'],
                job.company_address['addressRegion'],
                job.company_address['addressCountry'],
              ]
                .filter(Boolean)
                .join(', ') || 'Company address available'}
            </p>
          )}

          {(job.company_website || job.company_linkedin_url) && (
            <div className="mt-2 flex flex-wrap gap-2">
              {job.company_website && (
                <a href={job.company_website} target="_blank" rel="noopener noreferrer" className="underline text-[10px] text-sky-600">
                  Company site
                </a>
              )}
              {job.company_linkedin_url && (
                <a
                  href={job.company_linkedin_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline text-[10px] text-sky-600"
                >
                  Company LinkedIn
                </a>
              )}
            </div>
          )}

          {job.benefits && job.benefits.length > 0 && (
            <p className="text-[10px] text-text-muted mt-2">Benefits: {job.benefits.join(', ')}</p>
          )}
        </div>

        {job.url && (
          <a
            href={job.url}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary flex items-center justify-center gap-2 w-full mt-2"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Apply Now
          </a>
        )}
      </div>
    </div>
  )
}
