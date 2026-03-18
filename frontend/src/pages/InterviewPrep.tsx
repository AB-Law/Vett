import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Play, Square, MessageCircle } from 'lucide-react'
import toast from 'react-hot-toast'

import {
  createOrResumeInterviewChatSession,
  endInterviewChatSession,
  getInterviewChatSession,
  getJob,
  listInterviewChatSessions,
  streamInterviewChatTurn,
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

  const numericId = useMemo(() => Number(id || 0), [id])

  const refreshSessions = async (): Promise<void> => {
    if (!Number.isInteger(numericId) || numericId <= 0) return
    const list = await listInterviewChatSessions(numericId)
    setSessions(list)
  }

  const loadSessionDetail = async (sessionId: string): Promise<void> => {
    const detail = await getInterviewChatSession(numericId, sessionId)
    setActiveSession(detail)
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

  const startOrResume = async (): Promise<void> => {
    if (sending) return
    try {
      const session = await createOrResumeInterviewChatSession(numericId)
      setActiveSession(session)
      await refreshSessions()
      if (!session.turns.length) {
        setSending(true)
        setStreamingText('')
        await streamInterviewChatTurn(numericId, session.session_id, null, (token) => {
          setStreamingText((current) => current + token)
        })
        const updated = await getInterviewChatSession(numericId, session.session_id)
        setActiveSession(updated)
      }
    } catch {
      toast.error('Could not start interview prep')
    } finally {
      setStreamingText('')
      setSending(false)
    }
  }

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

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="flex items-center justify-center py-16 text-sm text-text-muted">Loading interview prep…</div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10 space-y-4">
      <div className="flex items-center gap-3">
        <Link to={`/jobs/${numericId}`} className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to Job
        </Link>
      </div>

      <div className="card p-4">
        <div className="flex items-center justify-between gap-3 mb-2">
          <div>
            <h1 className="text-lg font-semibold text-text-primary">Interview Prep</h1>
            <p className="text-xs text-text-secondary">
              {job?.title || 'Role'} at {job?.company || 'Company'}
            </p>
          </div>
          <button type="button" className="btn-primary text-xs py-1.5 px-3" onClick={() => void startOrResume()} disabled={sending}>
            {sending ? <Loader2 className="inline w-3 h-3 mr-1 animate-spin" /> : <Play className="inline w-3 h-3 mr-1" />}
            Start Interview Prep
          </button>
        </div>

        {sessions.length > 0 ? (
          <div className="border border-border rounded p-2 text-xs space-y-1">
            {sessions.map((session) => (
              <button
                key={session.session_id}
                type="button"
                onClick={() => void loadSessionDetail(session.session_id)}
                className="w-full text-left rounded border border-border px-2 py-1 hover:bg-surface-secondary"
              >
                <span className="font-medium text-text-primary">{session.label}</span>
                <span className="text-text-muted ml-2">{session.status} · {session.turn_count} turns</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-xs text-text-muted">No interview rounds yet.</p>
        )}
      </div>

      {activeSession ? (
        <div className="card p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="text-xs text-text-secondary">
              <span className="font-semibold text-text-primary">{activeSession.label}</span> · {activeSession.status} · {activeSession.phase}
            </div>
            <button
              type="button"
              onClick={() => void endInterview()}
              disabled={activeSession.status !== 'active' || sending}
              className="rounded border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 px-2 py-1 text-[11px] disabled:opacity-60"
            >
              <Square className="inline w-3 h-3 mr-1" />
              End Interview
            </button>
          </div>

          <div className="border border-border rounded p-2 max-h-[420px] overflow-auto space-y-2">
            {activeSession.turns.map((turn) => (
              <div key={turn.id} className={`rounded px-2 py-1 text-xs ${turn.speaker === 'assistant' ? 'bg-surface-secondary border border-border' : 'bg-sky-50 border border-sky-100'}`}>
                <div className="text-[10px] text-text-muted mb-0.5">
                  {turn.speaker} · {turn.turn_type}
                </div>
                <div className="text-text-primary whitespace-pre-wrap">{turn.content}</div>
              </div>
            ))}
            {streamingText ? (
              <div className="rounded px-2 py-1 text-xs bg-surface-secondary border border-border">
                <div className="text-[10px] text-text-muted mb-0.5">assistant · streaming</div>
                <div className="text-text-primary whitespace-pre-wrap">{streamingText}</div>
              </div>
            ) : null}
          </div>

          <div className="mt-3 flex gap-2">
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              className="field-input flex-1"
              placeholder={activeSession.status === 'active' ? 'Type your answer...' : 'Interview is complete'}
              disabled={activeSession.status !== 'active' || sending}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void sendMessage()
                }
              }}
            />
            <button
              type="button"
              onClick={() => void sendMessage()}
              disabled={!draft.trim() || activeSession.status !== 'active' || sending}
              className="btn-primary py-1.5 px-3 text-xs disabled:opacity-60"
            >
              {sending ? <Loader2 className="inline w-3 h-3 mr-1 animate-spin" /> : <MessageCircle className="inline w-3 h-3 mr-1" />}
              Send
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
