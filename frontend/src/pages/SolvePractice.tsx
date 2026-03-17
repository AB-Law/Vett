import { type KeyboardEvent, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Bot, Send, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import {
  askPracticeInterviewer,
  markPracticeQuestionSolved,
  type PracticeChatMessage,
} from '../lib/api'

type Message = {
  role: 'assistant' | 'user'
  content: string
}

const quickPrompts = [
  'Give me a structured way to explain this problem in an interview.',
  'What are the most likely follow-up questions?',
  'Help me build a 2-minute verbal plan for this question.',
]

function normalizeLanguage(value: string | null): string {
  if (!value) return ''
  const normalized = value.toLowerCase().trim()
  if (normalized === 'python3' || normalized === 'python') return 'Python'
  return value.trim()
}

function normalizeText(value: string | null): string {
  return (value || '').trim()
}

export default function SolvePractice() {
  const [searchParams] = useSearchParams()
  const questionId = Number(searchParams.get('questionId') || searchParams.get('question_id'))
  const sessionId = normalizeText(searchParams.get('sessionId') || searchParams.get('session_id'))
  const jobId = normalizeText(searchParams.get('jobId'))
  const questionTitle = normalizeText(searchParams.get('title')) || 'Practice question'
  const sourceUrl = normalizeText(searchParams.get('url'))
  const difficulty = normalizeText(searchParams.get('difficulty')) || 'unknown'
  const languageFromQuery = normalizeLanguage(searchParams.get('language'))
  const promptFromQuery = normalizeText(searchParams.get('prompt'))

  const backHref = jobId ? `/jobs/${jobId}` : '/jobs'
  const hasQuestion = Number.isFinite(questionId) && questionId > 0 && Number.isInteger(questionId)
  const hasSession = Boolean(sessionId)

  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Got it. I can coach this question without any code execution. Ask for clarification, interview framing, or practice follow-ups.',
    },
  ])
  const [chatInput, setChatInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isMarkingSolved, setIsMarkingSolved] = useState(false)

  const problemDescription = useMemo(() => {
    const sourceLine = sourceUrl ? `\nSource: ${sourceUrl}\n` : ''
    const base = `Interview prompt: ${questionTitle}.`
    const difficultyLine = difficulty ? `\nDifficulty tag: ${difficulty}.` : ''
    return `${base}${difficultyLine}${sourceLine}\n\n${promptFromQuery || 'No full statement available in this context yet. Open the source link for the full prompt.'}`
  }, [promptFromQuery, questionTitle, sourceUrl, difficulty])

  const markIfSolved = async () => {
    if (!hasSession || !hasQuestion || isMarkingSolved) return

    setIsMarkingSolved(true)
    try {
      await markPracticeQuestionSolved({ session_id: sessionId, question_id: questionId })
      toast.success('Marked as solved. You can still continue discussion.')
    } catch {
      toast.error('Could not mark this question as solved.')
    } finally {
      setIsMarkingSolved(false)
    }
  }

  const sendToBot = async () => {
    const message = chatInput.trim()
    if (!message || !hasSession || !hasQuestion || isSending) return

    const userMessage: Message = { role: 'user', content: message }
    setChatInput('')
    setMessages((previous) => [...previous, userMessage])
    setIsSending(true)

    try {
      const conversation = [...messages, userMessage].map<PracticeChatMessage>((item) => ({
        role: item.role,
        content: item.content,
      }))

      const reply = await askPracticeInterviewer({
        session_id: sessionId,
        question_id: questionId,
        message,
        language: languageFromQuery || null,
        interview_history: conversation,
      })

      setMessages((previous) => [...previous, { role: 'assistant', content: reply.interviewer_reply }])
    } catch {
      setMessages((previous) => [...previous, { role: 'assistant', content: 'I could not reach the AI mentor right now. Try again in a moment.' }])
    } finally {
      setIsSending(false)
    }
  }

  const prefillPrompt = (prompt: string) => {
    setChatInput(prompt)
  }

  if (!hasQuestion || !hasSession) {
    return (
      <div className="max-w-5xl mx-auto px-8 py-10">
        <Link to="/jobs" className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to jobs
        </Link>
        <div className="mt-4 card p-5">
          <p className="text-sm text-text-secondary">
            Missing question/session context. Open this page from Job Details using the coach action.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="mb-4">
        <Link to={backHref} className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to job
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4">
        <section className="card p-4 space-y-4">
          <header>
            <h1 className="text-lg font-semibold text-text-primary">{questionTitle}</h1>
            <p className="text-xs text-text-muted mt-1">Difficulty: {difficulty || 'unknown'} · Interview coaching workspace</p>
            {sourceUrl ? (
              <a href={sourceUrl} target="_blank" rel="noreferrer" className="text-xs text-sky-700 underline mt-1 inline-block">
                Open source link
              </a>
            ) : null}
            {languageFromQuery ? <p className="text-xs text-text-muted mt-1">Language context: {languageFromQuery}</p> : null}
          </header>

          <div className="border border-border rounded bg-slate-50 p-3 text-xs text-text-secondary">
            <p className="font-semibold text-text-primary mb-1">Prompt / context</p>
            <pre className="whitespace-pre-wrap leading-relaxed">{problemDescription}</pre>
          </div>

          <div className="flex flex-wrap gap-2">
            <button onClick={markIfSolved} disabled={isMarkingSolved} className="btn-primary">
              {isMarkingSolved ? 'Saving…' : 'Mark solved'}
            </button>
            <p className="text-[11px] text-text-muted flex items-center">
              <span className="rounded bg-sage-100 text-sage-800 px-2 py-1">No code execution or test generation on this page.</span>
            </p>
          </div>
        </section>

        <aside className="card p-0 flex flex-col">
          <header className="px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-sage-600" />
              <h2 className="text-sm font-semibold text-text-primary">AI coach</h2>
            </div>
            <p className="text-xs text-text-muted mt-1">Coach your reasoning and interview delivery.</p>
          </header>

          <div className="flex-1 p-3 overflow-y-auto space-y-2 max-h-[58vh]">
            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`rounded-lg px-2.5 py-1.5 text-xs ${
                  message.role === 'user'
                    ? 'bg-gray-100 text-text-primary'
                    : 'bg-sage-50 text-text-primary border border-sage-100'
                }`}
              >
                <p className="font-semibold text-[11px] mb-1">{message.role === 'assistant' ? 'AI Coach' : 'You'}</p>
                <p>{message.content}</p>
              </div>
            ))}
          </div>

          <div className="px-3 pb-3 space-y-2 border-t border-border">
            <div className="flex flex-wrap gap-1.5">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => prefillPrompt(prompt)}
                  className="text-[11px] rounded border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 px-2 py-1"
                >
                  {prompt}
                </button>
              ))}
            </div>

            <textarea
              className="input min-h-20 max-h-20 text-xs resize-none"
              placeholder="Ask for hints, corner cases, communication style, or expected interview follow-ups..."
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault()
                  void sendToBot()
                }
              }}
            />

            <button
              onClick={() => {
                void sendToBot()
              }}
              disabled={isSending || !chatInput.trim()}
              className="btn-primary w-full flex items-center justify-center gap-1.5"
            >
              <Send className="w-3.5 h-3.5" />
              {isSending ? 'Sending…' : 'Ask AI coach'}
            </button>

            <button
              onClick={() =>
                setMessages((previous) => [
                  ...previous,
                  {
                    role: 'assistant',
                    content: `Current coach context includes language: ${languageFromQuery || 'not specified'}. Ask for mock follow-up practice for this tone.`,
                  },
                ])
              }
              className="btn-secondary w-full flex items-center justify-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Seed coach with current context
            </button>
          </div>
        </aside>
      </div>
    </div>
  )
}
