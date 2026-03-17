import { useEffect, useState } from 'react'
import { ArrowLeft, BookOpen, CheckCircle2, ExternalLink, Loader2, MessageCircle, XCircle } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { isAxiosError } from 'axios'
import {
  getJob,
  type Job,
  getPracticeQuestions,
  type PracticeQuestion,
  markPracticeQuestionSolved,
  unmarkPracticeQuestionSolved,
  getPracticeNextQuestion,
  discardPracticeQuestion,
  askPracticeInterviewer,
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

  const getQuestionSourceWindow = (windowFilter: typeof questionWindow): string => {
    if (windowFilter === 'older-than-six-months') {
      return 'one-year'
    }
    return windowFilter
  }
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
