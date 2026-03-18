import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Play, Square, MessageCircle, Sparkles, Trash2 } from 'lucide-react'
import toast from 'react-hot-toast'

import {
  createOrResumeInterviewChatSession,
  deleteInterviewChatSession,
  endInterviewChatSession,
  getInterviewChatSession,
  getJob,
  listInterviewChatSessions,
  streamInterviewChatTurn,
  type InterviewChatFeedback,
  type InterviewChatSession,
  type InterviewChatSessionDetail,
  type Job,
} from '../lib/api'

export default function InterviewPrep() {
  const { id } = useParams<{ id: string }>()
  const [job, setJob] = useState<Job | null>(null)
  const [sessions, setSessions] = useState<InterviewChatSession[]>([])
  const [activeSession, setActiveSession] = useState<InterviewChatSessionDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [sending, setSending] = useState(false)
  const [draft, setDraft] = useState('')
  const [streamingText, setStreamingText] = useState('')
  const [latestFeedback, setLatestFeedback] = useState<InterviewChatFeedback | null>(null)
  const transcriptRef = useRef<HTMLDivElement | null>(null)
  const autoStartedRef = useRef(false)

  const numericId = useMemo(() => Number(id || 0), [id])

  const refreshSessions = async (): Promise<void> => {
    if (!Number.isInteger(numericId) || numericId <= 0) return
    const list = await listInterviewChatSessions(numericId)
    setSessions(list)
  }

  const loadSessionDetail = async (sessionId: string): Promise<void> => {
    const detail = await getInterviewChatSession(numericId, sessionId)
    setActiveSession(detail)
    setLatestFeedback(detail.feedback || null)
  }

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!Number.isInteger(numericId) || numericId <= 0) {
        setLoading(false)
        return
      }
      setLoading(true)
      try {
        const [jobResult, sessionList] = await Promise.all([getJob(numericId), listInterviewChatSessions(numericId)])
        if (cancelled) return
        setJob(jobResult)
        setSessions(sessionList)
      } catch {
        if (!cancelled) {
          toast.error('Could not load interview prep context')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [numericId])

  useEffect(() => {
    autoStartedRef.current = false
  }, [numericId])

  const streamTurn = async (message: string | null): Promise<void> => {
    if (!activeSession || sending) return
    setSending(true)
    setStreamingText('')
    try {
      await streamInterviewChatTurn(numericId, activeSession.session_id, message, (token) => {
        setStreamingText((current) => current + token)
      })
      await loadSessionDetail(activeSession.session_id)
      await refreshSessions()
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Interview stream failed')
    } finally {
      setStreamingText('')
      setSending(false)
    }
  }

  const activeTurnCount = activeSession?.turns.length ?? 0
  const isPreparing =
    activeSession?.status === 'active' &&
    (activeSession.phase === 'preparing' ||
      (activeSession.preparation_status !== undefined &&
        activeSession.preparation_status !== null &&
        activeSession.preparation_status !== 'ready' &&
        (activeSession.turns?.length ?? 0) === 0))

  const getAssistantTurnLabel = (turnType: string): string => {
    if (turnType === 'question') return 'Question'
    if (turnType === 'follow_up') return 'Follow-up'
    if (turnType === 'transition') return 'Transition'
    return ''
  }

  useEffect(() => {
    if (!transcriptRef.current) return
    transcriptRef.current.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [activeSession?.session_id, activeTurnCount, streamingText])

  const startOrResume = async (): Promise<void> => {
    if (sending) return
    try {
      const session = await createOrResumeInterviewChatSession(numericId)
      setActiveSession(session)
      setLatestFeedback(session.feedback || null)
      await refreshSessions()
      if (!session.turns.length) {
        setSending(true)
        setStreamingText('')
        await streamInterviewChatTurn(numericId, session.session_id, null, (token) => {
          setStreamingText((current) => current + token)
        })
        const updated = await getInterviewChatSession(numericId, session.session_id)
        setActiveSession(updated)
        setLatestFeedback(updated.feedback || null)
      }
    } catch {
      toast.error('Could not start interview prep')
    } finally {
      setStreamingText('')
      setSending(false)
    }
  }

  useEffect(() => {
    if (loading || sending || !job || activeSession || autoStartedRef.current) return
    autoStartedRef.current = true
    void startOrResume()
  }, [loading, sending, job, activeSession])

  const sendMessage = async (): Promise<void> => {
    if (!draft.trim()) return
    const value = draft.trim()
    setDraft('')
    await streamTurn(value)
  }

  const endInterview = async (): Promise<void> => {
    if (!activeSession || sending) return
    try {
      const result = await endInterviewChatSession(numericId, activeSession.session_id)
      await loadSessionDetail(activeSession.session_id)
      setLatestFeedback(result.feedback || null)
      await refreshSessions()
      if (result.handoff_status === 'triggered') {
        toast.success('Interview ended and scoring handoff triggered')
      } else {
        toast('Interview ended')
      }
    } catch {
      toast.error('Could not end interview')
    }
  }

  const removeSession = async (sessionId: string): Promise<void> => {
    if (sending) return
    const proceed = globalThis.confirm('Delete this interview session and transcript permanently?')
    if (!proceed) return
    try {
      await deleteInterviewChatSession(numericId, sessionId)
      if (activeSession?.session_id === sessionId) {
        setActiveSession(null)
        setLatestFeedback(null)
      }
      await refreshSessions()
      toast.success('Interview session deleted')
    } catch {
      toast.error('Could not delete interview session')
    }
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="flex items-center justify-center py-16 text-sm text-text-muted">Loading interview prep…</div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8 space-y-4">
      <div className="flex items-center gap-3">
        <Link to={`/jobs/${numericId}`} className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to Job
        </Link>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4">
        <aside className="card p-4 h-fit lg:sticky lg:top-6">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div>
              <h1 className="text-base font-semibold text-text-primary">Interview Prep</h1>
              <p className="text-xs text-text-secondary mt-1">
                {job?.title || 'Role'} at {job?.company || 'Company'}
              </p>
            </div>
            <button type="button" className="btn-primary text-xs py-1.5 px-3 shrink-0" onClick={() => void startOrResume()} disabled={sending}>
              {sending ? <Loader2 className="inline w-3 h-3 mr-1 animate-spin" /> : <Play className="inline w-3 h-3 mr-1" />}
              Resume
            </button>
          </div>

          <div className="space-y-2">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Sessions</p>
            {sessions.length > 0 ? (
              <div className="space-y-1.5">
                {sessions.map((session) => {
                  const isActive = activeSession?.session_id === session.session_id
                  return (
                    <div
                      key={session.session_id}
                      className={`rounded-xl border px-2 py-2 transition ${
                        isActive
                          ? 'border-blue-300 bg-blue-50/70'
                          : 'border-border bg-surface hover:bg-surface-secondary'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            void (async () => {
                              try {
                                await loadSessionDetail(session.session_id)
                              } catch (error: unknown) {
                                setActiveSession(null)
                                setLatestFeedback(null)
                                setStreamingText('')
                                toast.error(error instanceof Error ? error.message : 'Could not load interview session')
                              }
                            })()
                          }}
                          className="flex-1 text-left px-1"
                        >
                          <div className="text-xs font-medium text-text-primary">{session.label}</div>
                          <div className="mt-1 flex items-center justify-between text-[11px] text-text-muted">
                            <span className="capitalize">{session.status}</span>
                            <span>{session.turn_count} turns</span>
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={() => void removeSession(session.session_id)}
                          className="mt-0.5 rounded p-1 text-text-muted hover:text-red-600 hover:bg-red-50"
                          title="Delete session"
                          aria-label={`Delete ${session.label}`}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-xs text-text-muted">No interview rounds yet.</p>
            )}
          </div>
        </aside>

        <section className="card p-0 overflow-hidden">
          {activeSession ? (
            <>
              <div className="sticky top-0 z-10 px-4 py-3 border-b border-border bg-white/95 backdrop-blur flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-text-primary">{activeSession.label}</div>
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-text-secondary">
                    <span className="rounded-full border border-border px-2 py-0.5 capitalize">{activeSession.status}</span>
                    <span className="rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 capitalize">{activeSession.phase.split('_').join(' ')}</span>
                    {activeSession.limits ? (
                      <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5">
                        Q {activeSession.primary_question_count ?? 0}/{activeSession.limits.max_questions}
                      </span>
                    ) : null}
                    {typeof activeSession.rolling_score === 'number' ? (
                      <span className="rounded-full border border-indigo-200 bg-indigo-50 px-2 py-0.5">
                        Score {Math.round(activeSession.rolling_score)}
                      </span>
                    ) : null}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void endInterview()}
                  disabled={activeSession.status !== 'active' || sending}
                  className="rounded-lg border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 px-2.5 py-1.5 text-[11px] disabled:opacity-60"
                >
                  <Square className="inline w-3 h-3 mr-1" />
                  End Interview
                </button>
              </div>

              {isPreparing ? (
                <div className="h-[560px] flex flex-col items-center justify-center gap-3 text-center px-6">
                  <Loader2 className="w-7 h-7 animate-spin text-blue-600" />
                  <p className="text-sm font-medium text-text-primary">Preparing interviewer…</p>
                  <p className="text-xs text-text-muted max-w-md">
                    Building role context, selecting focus areas, and creating a dynamic interview strategy.
                  </p>
                </div>
              ) : (
                <>

              {latestFeedback && activeSession.status === 'completed' ? (
                <div className="mx-4 mt-4 rounded-xl border border-blue-200 bg-blue-50/60 p-3 text-sm">
                  <div className="text-sm font-semibold text-text-primary mb-1">Interview Feedback</div>
                  <p className="text-xs text-text-secondary mb-3">{latestFeedback.overview}</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-green-700 mb-1">What Went Well</p>
                      <ul className="space-y-1 text-xs text-text-secondary">
                        {latestFeedback.what_went_well.map((item) => (
                          <li key={`good-${item}`}>- {item}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-amber-700 mb-1">What To Improve</p>
                      <ul className="space-y-1 text-xs text-text-secondary">
                        {latestFeedback.what_to_improve.map((item) => (
                          <li key={`improve-${item}`}>- {item}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-blue-700 mb-1">Next Steps</p>
                      <ul className="space-y-1 text-xs text-text-secondary">
                        {latestFeedback.next_steps.map((item) => (
                          <li key={`next-${item}`}>- {item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              ) : null}

              <div ref={transcriptRef} className="h-[560px] overflow-auto px-4 py-4 bg-[linear-gradient(180deg,#fafafa_0%,#ffffff_100%)]">
                <div className="space-y-3">
                  {activeSession.turns.map((turn) => {
                    const isAssistant = turn.speaker === 'assistant'
                    const assistantLabel = getAssistantTurnLabel(turn.turn_type)
                    let speakerMeta = 'You · Answer'
                    if (isAssistant) {
                      speakerMeta = assistantLabel ? `Interviewer · ${assistantLabel}` : 'Interviewer'
                    }
                    return (
                      <div key={turn.id} className={`flex ${isAssistant ? 'justify-start' : 'justify-end'}`}>
                        <div
                          className={`max-w-[80%] rounded-xl px-3 py-2 border text-sm shadow-sm ${
                            isAssistant
                              ? 'bg-white border-border'
                              : 'bg-blue-600 text-white border-blue-600'
                          }`}
                        >
                          <div className={`mb-1 text-[10px] uppercase tracking-wide ${isAssistant ? 'text-text-muted' : 'text-blue-100'}`}>
                            {speakerMeta}
                          </div>
                          <div className={`whitespace-pre-wrap ${isAssistant ? 'text-text-primary' : 'text-white'}`}>{turn.content}</div>
                        </div>
                      </div>
                    )
                  })}

                  {streamingText ? (
                    <div className="flex justify-start">
                      <div className="max-w-[80%] rounded-xl px-3 py-2 border border-blue-200 bg-blue-50 shadow-sm text-sm">
                        <div className="mb-1 text-[10px] uppercase tracking-wide text-blue-700 flex items-center gap-1">
                          <Sparkles className="w-3 h-3" />
                          Interviewer · Thinking
                        </div>
                        <div className="text-text-primary whitespace-pre-wrap">{streamingText}</div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="border-t border-border p-3 bg-white">
                <div className="flex gap-2 items-end">
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    className="field-input flex-1 min-h-[48px] max-h-40 resize-y"
                    placeholder={activeSession.status === 'active' ? 'Write your answer…' : 'Interview is complete'}
                    disabled={activeSession.status !== 'active' || sending}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault()
                        void sendMessage()
                      }
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => void sendMessage()}
                    disabled={!draft.trim() || activeSession.status !== 'active' || sending}
                    className="btn-primary h-10 py-1.5 px-3 text-xs disabled:opacity-60"
                  >
                    {sending ? <Loader2 className="inline w-3 h-3 mr-1 animate-spin" /> : <MessageCircle className="inline w-3 h-3 mr-1" />}
                    Send
                  </button>
                </div>
              </div>
                </>
              )}
            </>
          ) : (
            <div className="h-[560px] flex items-center justify-center text-sm text-text-muted">
              Preparing your interview session…
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
