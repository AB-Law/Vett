import { type KeyboardEvent, useEffect, useMemo, useRef, type UIEvent, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft,
  Bot,
  CalendarClock,
  Copy,
  Download,
  Play,
  RefreshCw,
  Send,
  Sparkles,
} from 'lucide-react'
import toast from 'react-hot-toast'
import {
  askPracticeInterviewer,
  generatePracticeTestCases,
  markPracticeQuestionSolved,
  generatePracticeSolutionTemplate,
  runPracticeCode,
  reviewPracticeSolution,
  type PracticeChatMessage,
  type PracticeRunCaseItem,
  type PracticeTestCaseItem,
  type PracticeTestCaseResponse,
} from '../lib/api'

type Tab = 'description' | 'tests' | 'output'
type Message = {
  role: 'assistant' | 'user'
  content: string
}

const languageOptions = [
  { value: 'python3', label: 'Python 3' },
]

const starterTemplateByLanguage: Record<string, string> = {
  python3: `class Solution:\n    def solve(self, nums, target):\n        # Write your approach\n        raise NotImplementedError\n`,
}

function normalizeLanguage(value: string | null): string {
  if (!value) return 'python3'
  if (value.toLowerCase() === 'python3' || value.toLowerCase() === 'python') return 'python3'
  return 'python3'
}

function normalizeText(value: string | null): string {
  return (value || '').trim()
}

const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'for',
  'finally',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
])

const INDENT_UNIT = '    '
const BUILTIN_VALUES = new Set(['self', 'len', 'print', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple'])
const fallbackTemplate = starterTemplateByLanguage.python3 || `class Solution:\n    def solve(self, nums, target):\n        # Write your approach\n        raise NotImplementedError\n`

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

function wrapToken(value: string, style: string): string {
  return `<span style="${style}">${escapeHtml(value)}</span>`
}

function highlightPythonCode(codeText: string): string {
  const tokens: string[] = []
  let index = 0

  while (index < codeText.length) {
    const character = codeText[index]

    if (character === '#') {
      const lineBreakAt = codeText.indexOf('\n', index)
      const end = lineBreakAt === -1 ? codeText.length : lineBreakAt
      tokens.push(wrapToken(codeText.slice(index, end), 'color:#6e7781;'))
      index = end
      continue
    }

    if (codeText.startsWith("'''", index) || codeText.startsWith('"""', index)) {
      const quote = codeText.slice(index, index + 3)
      const closing = codeText.indexOf(quote, index + 3)
      const end = closing === -1 ? codeText.length : closing + 3
      tokens.push(wrapToken(codeText.slice(index, end), 'color:#50a14f;'))
      index = end
      continue
    }

    if (character === '"' || character === "'") {
      let next = index + 1
      let escaped = false
      while (next < codeText.length) {
        const current = codeText[next]
        if (escaped) {
          escaped = false
          next += 1
          continue
        }
        if (current === '\\') {
          escaped = true
          next += 1
          continue
        }
        if (current === character) {
          next += 1
          break
        }
        next += 1
      }
      tokens.push(wrapToken(codeText.slice(index, next), 'color:#50a14f;'))
      index = next
      continue
    }

    if (/[A-Za-z_]/.test(character)) {
      let next = index + 1
      while (next < codeText.length && /[A-Za-z0-9_]/.test(codeText[next])) {
        next += 1
      }
      const word = codeText.slice(index, next)
      if (PYTHON_KEYWORDS.has(word)) {
        tokens.push(wrapToken(word, 'color:#c678dd;font-weight:600;'))
      } else if (BUILTIN_VALUES.has(word)) {
        tokens.push(wrapToken(word, 'color:#56b6c2;'))
      } else {
        tokens.push(escapeHtml(word))
      }
      index = next
      continue
    }

    if (/[0-9]/.test(character)) {
      let next = index + 1
      while (next < codeText.length && /[0-9a-zA-Z_+\.\-eE]/.test(codeText[next])) {
        next += 1
      }
      tokens.push(wrapToken(codeText.slice(index, next), 'color:#d19a66;'))
      index = next
      continue
    }

    if (/[=+\-*/%<>!&|^~:(),.;{}\[\]]/.test(character)) {
      tokens.push(wrapToken(character, 'color:#e06c75;'))
      index += 1
      continue
    }

    if (character === '\n' || character === '\t') {
      tokens.push(escapeHtml(character))
      index += 1
      continue
    }

    tokens.push(escapeHtml(character))
    index += 1
  }

  return tokens.join('')
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

  const [language, setLanguage] = useState<string>(languageFromQuery)
  const [activeTab, setActiveTab] = useState<Tab>('description')
  const [code, setCode] = useState<string>(starterTemplateByLanguage[languageFromQuery] || fallbackTemplate)
  const [baseTemplate, setBaseTemplate] = useState<string>(starterTemplateByLanguage[languageFromQuery] || fallbackTemplate)
  const [isTemplateLoading, setIsTemplateLoading] = useState(false)
  const [problemPromptFromTemplate, setProblemPromptFromTemplate] = useState('')
  const [hasUserEditedCode, setHasUserEditedCode] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content:
        'Got it. I can review your code, generate edge-case tests, and suggest hints as you solve.',
    },
  ])
  const [chatInput, setChatInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isReviewing, setIsReviewing] = useState(false)
  const [tests, setTests] = useState<PracticeTestCaseItem[]>([])
  const [isGeneratingTests, setIsGeneratingTests] = useState(false)
  const [runOutput, setRunOutput] = useState<string[]>([
    'No run yet. Use Run to validate against AI-generated checks.',
  ])
  const [isRunning, setIsRunning] = useState(false)
  const [isMarkingSolved, setIsMarkingSolved] = useState(false)
  const codeTextareaRef = useRef<HTMLTextAreaElement>(null)
  const codeHighlightRef = useRef<HTMLPreElement>(null)
  const highlightedCode = useMemo(
    () => (language === 'python3' ? highlightPythonCode(code) : escapeHtml(code)),
    [language, code],
  )

  const problemDescription = useMemo(() => {
    if (promptFromQuery) return promptFromQuery
    if (problemPromptFromTemplate) return problemPromptFromTemplate
    const base = `Solve "${questionTitle}".`
    const sourceLine = sourceUrl ? `\nSource: ${sourceUrl}\n` : ''
    return `${base}${sourceLine}\n\n`
      + 'No full statement is available in this workspace yet. Open the source link for full context.'
  }, [promptFromQuery, problemPromptFromTemplate, questionTitle, sourceUrl])

  const hasQuestion = Number.isFinite(questionId) && questionId > 0 && Number.isInteger(questionId)
  const hasSession = !!sessionId

  useEffect(() => {
    if (!hasSession || !hasQuestion) {
      return
    }

    let cancelled = false
    const requestTemplate = async (): Promise<void> => {
      setProblemPromptFromTemplate('')
      setIsTemplateLoading(true)
      try {
        const templateResponse = await generatePracticeSolutionTemplate({
          session_id: sessionId,
          question_id: questionId,
          language,
          question_prompt: promptFromQuery || null,
        })
        if (cancelled) {
          return
        }
        setBaseTemplate(templateResponse.template)
        setProblemPromptFromTemplate(templateResponse.problem_prompt || '')
        if (!hasUserEditedCode && templateResponse.template.trim()) {
          setCode(templateResponse.template)
        }
      } catch {
        setBaseTemplate(fallbackTemplate)
        setProblemPromptFromTemplate(promptFromQuery || '')
      } finally {
        if (!cancelled) {
          setIsTemplateLoading(false)
        }
      }
    }
    void requestTemplate()
    return () => {
      cancelled = true
    }
  }, [hasSession, hasQuestion, sessionId, questionId, language, promptFromQuery, hasUserEditedCode])

  useEffect(() => {
    setBaseTemplate(fallbackTemplate)
    if (!hasUserEditedCode) {
      setCode(fallbackTemplate)
    }
  }, [language, hasUserEditedCode])

  const markIfSolved = async () => {
    if (!hasSession || !hasQuestion || isMarkingSolved) return
    setIsMarkingSolved(true)
    try {
      await markPracticeQuestionSolved({ session_id: sessionId, question_id: questionId })
      toast.success('Marked as solved. You can still continue iterating.')
    } catch {
      toast.error('Could not mark this question as solved.')
    } finally {
      setIsMarkingSolved(false)
    }
  }

  const insertCodeAtCursor = (valueToInsert: string, selectionStart: number, selectionEnd: number) => {
    setCode((currentCode) => {
      const nextCode = `${currentCode.slice(0, selectionStart)}${valueToInsert}${currentCode.slice(selectionEnd)}`
      requestAnimationFrame(() => {
        const textarea = codeTextareaRef.current
        if (textarea) {
          const cursor = selectionStart + valueToInsert.length
          textarea.selectionStart = cursor
          textarea.selectionEnd = cursor
        }
      })
      return nextCode
    })
  }

  const handleCodeKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Tab') {
      event.preventDefault()
      const textarea = event.currentTarget
      insertCodeAtCursor(INDENT_UNIT, textarea.selectionStart, textarea.selectionEnd)
      return
    }

    if (event.key === 'Enter') {
      event.preventDefault()
      const textarea = event.currentTarget
      const cursor = textarea.selectionStart
      const beforeCursor = code.slice(0, cursor)
      const lineStart = beforeCursor.lastIndexOf('\n') + 1
      const currentLine = beforeCursor.slice(lineStart)
      const currentIndent = currentLine.match(/^[ \t]*/)?.[0] || ''
      const dedentedIndent = currentIndent
      const nextIndent = currentLine.trim().endsWith(':') ? `${dedentedIndent}${INDENT_UNIT}` : dedentedIndent
      insertCodeAtCursor(`\n${nextIndent}`, cursor, textarea.selectionEnd)
    }
  }

  const syncCodeScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    const textarea = event.currentTarget
    if (codeHighlightRef.current) {
      codeHighlightRef.current.scrollTop = textarea.scrollTop
      codeHighlightRef.current.scrollLeft = textarea.scrollLeft
    }
  }

  const generateTests = async () => {
    if (!hasSession || !hasQuestion) {
      toast.error('Missing session or question id. Open this page from a job question first.')
      return
    }
    setIsGeneratingTests(true)
    setRunOutput(['Generating AI test cases...'])
    try {
      const response: PracticeTestCaseResponse = await generatePracticeTestCases({
        session_id: sessionId,
        question_id: questionId,
        language,
        count: 8,
      })
      setTests(response.test_cases)
      setActiveTab('tests')
      setRunOutput([
        `Generated ${response.test_cases.length} test cases for ${questionTitle}.`,
        ...(response.llm_provider ? [`Model provider: ${response.llm_provider}`] : []),
      ])
      toast.success('AI test cases generated.')
    } catch {
      toast.error('Could not generate test cases right now.')
      setRunOutput(['Failed to generate AI test cases.'])
    } finally {
      setIsGeneratingTests(false)
    }
  }

  const runCode = async () => {
    if (!hasQuestion || !hasSession) {
      return
    }
    setIsRunning(true)
    const selectedTests = tests.map((test): PracticeRunCaseItem => ({
      name: test.name,
      input: test.input,
      expected_output: test.expected_output,
      rationale: test.rationale,
    }))
    setActiveTab('output')
    setRunOutput(['Running python code...'])

    if (!code.trim()) {
      setRunOutput(['No solution text found. Write code in the editor before running.'])
      setIsRunning(false)
      return
    }

    try {
      const response = await runPracticeCode({
        session_id: sessionId,
        question_id: questionId,
        language,
        code,
        tests: selectedTests,
      })
      const output: string[] = [
        `Python runtime status: ${response.status}`,
        response.summary,
        `Cases passed: ${response.passed} / ${response.total}`,
      ]
      if (response.output.length) {
        output.push('Output:')
        output.push(...response.output)
      }
      if (response.results.length === 0) {
        output.push('No structured test results returned. Ensure at least one test case is present.')
      } else {
        output.push('Case results:')
        response.results.forEach((item, index) => {
          const heading = `${index + 1}) ${item.name}`
          if (item.passed === true) {
            output.push(`${heading}: PASS`)
          } else if (item.passed === false) {
            output.push(`${heading}: FAIL`)
          } else {
            output.push(`${heading}: INCOMPLETE`)
          }
          output.push(`  input: ${item.input}`)
          output.push(`  expected: ${item.expected_output}`)
          output.push(`  actual: ${item.actual_output ?? 'null'}`)
          if (item.error) {
            output.push(`  error: ${item.error}`)
          }
        })
      }
      setRunOutput(output)
    } catch {
      setRunOutput([
        'Could not execute code right now.',
        tests.length
          ? 'Try again after regenerating test cases if inputs are malformed.'
          : 'No test cases yet, so execution is currently limited to runtime sanity checks.',
      ])
      toast.error('Python runtime execution failed.')
    } finally {
      setIsRunning(false)
    }
  }

  const reviewCode = async () => {
    if (!hasSession || !hasQuestion || isReviewing) return
    setIsReviewing(true)
    setActiveTab('output')
    setRunOutput(['Running review on written code...'])
    try {
      const review = await reviewPracticeSolution({
        session_id: sessionId,
        question_id: questionId,
        solution_text: code,
        language,
      })
      setRunOutput((previous) => [
        ...previous,
        `Review summary: ${review.review_summary}`,
        `Strengths: ${review.strengths.join(', ') || 'None listed.'}`,
        `Concerns: ${review.concerns.join(', ') || 'None listed.'}`,
        `Variant check: ${review.follow_up_prompt}`,
      ])
      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: `Review complete for ${questionTitle}: ${review.review_summary}`,
        },
      ])
      toast.success('Review completed.')
    } catch {
      toast.error('Review failed. Try again in a bit.')
      setRunOutput((previous) => [...previous, 'Review failed.'])
    } finally {
      setIsReviewing(false)
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
      const history = messages
        .map<PracticeChatMessage>((item): PracticeChatMessage => ({
          role: item.role,
          content: item.content,
        }))
        .concat(userMessage)
      const reply = await askPracticeInterviewer({
        session_id: sessionId,
        question_id: questionId,
        message,
        language,
        interview_history: history,
        solution_text: code,
      })
      setMessages((previous) => [...previous, { role: 'assistant', content: reply.interviewer_reply }])
    } catch {
      setMessages((previous) => [...previous, { role: 'assistant', content: 'Could not reach interviewer bot right now.' }])
    } finally {
      setIsSending(false)
    }
  }

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    toast.success('Code copied')
  }

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${questionTitle.replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '') || 'solution'}.py`
    link.click()
    URL.revokeObjectURL(link.href)
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
            Missing question/session context. Open this page from Job Details using the Solve button.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="mb-4">
        <Link to={backHref} className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
          <ArrowLeft className="w-4 h-4" />
          Back to job
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_390px] gap-4">
        {/* Main editor + tabs */}
        <div className="card p-0 flex flex-col">
          <header className="border-b border-border px-4 py-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h1 className="text-lg font-semibold text-text-primary">{questionTitle}</h1>
                <p className="text-xs text-text-muted mt-1">
                  <span className="inline-flex items-center gap-1">
                    <CalendarClock className="w-3.5 h-3.5" />
                    Difficulty: {difficulty || 'unknown'} · LeetCode-style workspace
                  </span>
                </p>
                {sourceUrl ? (
                  <a href={sourceUrl} target="_blank" rel="noreferrer" className="text-xs text-sky-700 underline">
                    Open source link
                  </a>
                ) : null}
              </div>
              <div className="flex items-center gap-2">
                <label className="label mb-0">Language</label>
                <select
                  className="input w-32 py-1.5"
                  value={language}
                  onChange={(e) => {
                    const nextLanguage = e.target.value
                    if (nextLanguage !== language) {
                      setLanguage(nextLanguage)
                      const nextTemplate = fallbackTemplate
                      setBaseTemplate(nextTemplate)
                      setCode(nextTemplate)
                      setHasUserEditedCode(false)
                    }
                  }}
                >
                  {languageOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                {isTemplateLoading ? (
                  <span className="text-[11px] text-text-muted">AI starter loading…</span>
                ) : null}
              </div>
            </div>
          </header>

          <div className="border-b border-border px-4">
            <div className="flex items-center gap-2 py-2">
              {(['description', 'tests', 'output'] as Tab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-2 py-1.5 text-xs rounded-md ${
                    activeTab === tab
                      ? 'bg-sage-600 text-white'
                      : 'text-text-muted hover:text-text-primary hover:bg-gray-100'
                  }`}
                >
                  {tab === 'description' ? 'Description' : tab === 'tests' ? 'Tests' : 'Run / Output'}
                </button>
              ))}
            </div>
          </div>

          <div className="px-4 py-3 h-[34vh] overflow-y-auto bg-slate-50 border-b border-border text-sm leading-relaxed">
            {activeTab === 'description' ? (
              <pre className="whitespace-pre-wrap text-text-secondary text-xs">{problemDescription}</pre>
            ) : activeTab === 'tests' ? (
              <div className="space-y-2">
                {tests.length === 0 ? (
                  <p className="text-xs text-text-muted">No test cases yet. Use Generate test cases.</p>
                ) : (
                  tests.map((test, idx) => (
                    <div key={`${test.name}-${idx}`} className="border border-border rounded-lg p-2.5">
                      <p className="font-medium text-xs text-text-primary">{test.name || `Case ${idx + 1}`}</p>
                      <p className="text-xs text-text-muted mt-1">Input: {test.input}</p>
                      <p className="text-xs text-text-muted mt-1">Expected: {test.expected_output}</p>
                      {test.rationale ? <p className="text-xs text-text-muted mt-1">Why: {test.rationale}</p> : null}
                    </div>
                  ))
                )}
              </div>
            ) : (
              <pre className="text-xs whitespace-pre-wrap text-text-secondary">{runOutput.join('\n')}</pre>
            )}
          </div>

          <div className="p-3 border-b border-border">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={runCode}
                disabled={isRunning}
                className="btn-secondary flex items-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5" />
                {isRunning ? 'Running...' : 'Run'}
              </button>
              <button onClick={generateTests} disabled={isGeneratingTests} className="btn-secondary">
                {isGeneratingTests ? 'Generating…' : 'Generate test cases'}
              </button>
              <button onClick={reviewCode} disabled={isReviewing} className="btn-secondary">
                {isReviewing ? 'Reviewing…' : 'AI review'}
              </button>
              <button onClick={markIfSolved} disabled={isMarkingSolved} className="btn-primary">
                {isMarkingSolved ? 'Saving…' : 'Mark solved'}
              </button>
            </div>
          </div>

          <div className="p-3">
            <div className="relative rounded border border-border bg-white overflow-hidden min-h-[260px] h-64">
              <pre
                ref={codeHighlightRef}
                aria-hidden="true"
                className="absolute inset-0 m-0 overflow-auto p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-text-primary"
                dangerouslySetInnerHTML={{ __html: highlightedCode }}
              />
              <textarea
                ref={codeTextareaRef}
                className="absolute inset-0 m-0 w-full h-full resize-none p-3 bg-transparent border-0 outline-none focus:ring-0 font-mono text-xs leading-relaxed"
                style={{
                  color: 'transparent',
                  caretColor: '#0f172a',
                  tabSize: INDENT_UNIT.length,
                  WebkitTextFillColor: 'transparent',
                }}
                value={code}
                onKeyDown={handleCodeKeyDown}
                onScroll={syncCodeScroll}
                onChange={(event) => {
                  setHasUserEditedCode(true)
                  setCode(event.target.value)
                }}
                spellCheck={false}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
              />
            </div>
            <div className="flex items-center gap-2 mt-2">
              <button onClick={copyCode} className="btn-secondary text-xs flex items-center gap-1">
                <Copy className="w-3 h-3" />
                Copy code
              </button>
              <button onClick={downloadCode} className="btn-secondary text-xs flex items-center gap-1">
                <Download className="w-3 h-3" />
                Download .py
              </button>
              <button
                onClick={() => {
                  setCode(baseTemplate)
                  setHasUserEditedCode(false)
                }}
                className="btn-secondary text-xs flex items-center gap-1"
              >
                <RefreshCw className="w-3 h-3" />
                Reset
              </button>
            </div>
          </div>
        </div>

        {/* AI bot right side */}
        <aside className="card p-0 flex flex-col">
          <header className="px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-sage-600" />
              <h2 className="text-sm font-semibold text-text-primary">AI mentor</h2>
            </div>
            <p className="text-xs text-text-muted mt-1">Ask anything while coding. It sees your written draft.</p>
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
                <p className="font-semibold text-[11px] mb-1">{message.role === 'assistant' ? 'AI Bot' : 'You'}</p>
                <p>{message.content}</p>
              </div>
            ))}
          </div>

          <div className="p-3 border-t border-border space-y-2">
            <textarea
              className="input min-h-20 max-h-20 text-xs resize-none"
              placeholder="Ask for hints, edge cases, or implementation guidance..."
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault()
                  void sendToBot()
                }
              }}
            />
            <button
              onClick={() => void sendToBot()}
              disabled={isSending || !chatInput.trim()}
              className="btn-primary w-full flex items-center justify-center gap-1.5"
            >
              <Send className="w-3.5 h-3.5" />
              {isSending ? 'Sending…' : 'Ask AI bot'}
            </button>
            <button
              onClick={() => {
                setMessages((previous) => [
                  ...previous,
                  { role: 'assistant', content: `Current language: ${language}. I can help with this draft.` },
                ])
              }}
              className="btn-secondary w-full flex items-center justify-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Seed bot with current language
            </button>
          </div>
        </aside>
      </div>
    </div>
  )
}
