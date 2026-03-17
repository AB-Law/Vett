import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Zap,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Lightbulb,
  Save,
  AlertCircle,
} from 'lucide-react'
import { scoreJD, type ScoreResult } from '../lib/api'
import { scoreColorHex } from '../lib/utils'
import toast from 'react-hot-toast'

export default function ScoreJD() {
  const [jd, setJd] = useState('')
  const [jobTitle, setJobTitle] = useState('')
  const [company, setCompany] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScoreResult | null>(null)
  const [suggestionsOpen, setSuggestionsOpen] = useState(true)
  const [savedId, setSavedId] = useState<number | null>(null)

  const handleScore = async () => {
    if (!jd.trim()) {
      toast.error('Please paste a job description first')
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const res = await scoreJD({ job_description: jd, job_title: jobTitle || undefined, company: company || undefined })
      setResult(res)
      if (res.id) setSavedId(res.id)
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      if (detail?.includes('No CV')) {
        toast.error(
          () => (
            <span>
              No CV loaded.{' '}
              <Link to="/cv" className="underline font-medium">
                Upload one first
              </Link>
            </span>
          ),
          { duration: 5000 }
        )
      } else {
        toast.error(detail || 'Scoring failed. Check your LLM config in Settings.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-serif text-text-primary">Score Job Description</h1>
        <p className="text-text-secondary text-sm mt-1">
          Paste a JD and get your fit score instantly
        </p>
      </div>

      <div className="flex gap-6">
        {/* Left – Input */}
        <div className="flex-1 min-w-0">
          <div className="card p-5">
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div>
                <label className="label">Job Title (optional)</label>
                <input
                  className="input"
                  placeholder="e.g. Senior Software Engineer"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                />
              </div>
              <div>
                <label className="label">Company (optional)</label>
                <input
                  className="input"
                  placeholder="e.g. Acme Corp"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                />
              </div>
            </div>

            <label className="label">Job Description *</label>
            <textarea
              className="input font-mono text-xs leading-relaxed resize-none"
              style={{ minHeight: '360px' }}
              placeholder="Paste the full job description here…"
              value={jd}
              onChange={(e) => setJd(e.target.value)}
            />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-text-muted">
                {jd.length.toLocaleString()} characters
              </span>
              <button
                onClick={handleScore}
                disabled={loading}
                className="btn-primary flex items-center gap-2 px-5"
              >
                {loading ? (
                  <>
                    <SpinnerIcon />
                    Scoring…
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Score My CV
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right – Results */}
        <div className="w-[380px] shrink-0">
          {result ? (
            <ResultPanel result={result} savedId={savedId} suggestionsOpen={suggestionsOpen} setSuggestionsOpen={setSuggestionsOpen} />
          ) : (
            <div className="card flex flex-col items-center justify-center py-20 text-center px-8">
              <div className="w-12 h-12 rounded-full bg-sage-50 flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-sage-400" />
              </div>
              <p className="text-sm font-medium text-text-secondary mb-1">Results will appear here</p>
              <p className="text-xs text-text-muted">
                Paste a job description and click "Score My CV"
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResultPanel({
  result,
  savedId,
  suggestionsOpen,
  setSuggestionsOpen,
}: {
  result: ScoreResult
  savedId: number | null
  suggestionsOpen: boolean
  setSuggestionsOpen: (v: boolean) => void
}) {
  const score = Math.round(result.fit_score)
  const color = scoreColorHex(score)

  return (
    <div className="flex flex-col gap-4">
      {/* Score gauge */}
      <div className="card p-5 text-center">
        <p className="text-xs text-text-muted uppercase tracking-wide font-medium mb-3">Fit Score</p>
        <ScoreGauge score={score} color={color} />
        {result.llm_provider && (
          <p className="text-xs text-text-muted mt-3">
            via {result.llm_provider} · {result.llm_model}
          </p>
        )}
        {savedId && (
          <div className="flex items-center justify-center gap-1.5 mt-2 text-xs text-sage-600">
            <Save className="w-3.5 h-3.5" />
            Saved to history
          </div>
        )}
      </div>

      {/* Matched keywords */}
      {result.matched_keywords.length > 0 && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4 text-sage-500" />
            <span className="text-sm font-semibold text-text-primary">
              Matched ({result.matched_keywords.length})
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.matched_keywords.map((kw) => (
              <span key={kw} className="keyword-matched">{kw}</span>
            ))}
          </div>
        </div>
      )}

      {/* Missing keywords */}
      {result.missing_keywords.length > 0 && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <XCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-semibold text-text-primary">
              Missing ({result.missing_keywords.length})
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.missing_keywords.map((kw) => (
              <span key={kw} className="keyword-missing">{kw}</span>
            ))}
          </div>
        </div>
      )}

      {/* Gap Analysis */}
      {result.gap_analysis && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="w-4 h-4 text-text-secondary" />
            <span className="text-sm font-semibold text-text-primary">Gap Analysis</span>
          </div>
          <p className="text-sm text-text-secondary leading-relaxed">{result.gap_analysis}</p>
        </div>
      )}

      {/* Agent Plan */}
      {result.agent_plan && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-text-secondary" />
            <span className="text-sm font-semibold text-text-primary">Agentic Upgrade Plan</span>
          </div>
          {result.agent_plan.skills_to_fix_first.length > 0 && (
            <div className="mb-3">
              <h3 className="text-xs font-semibold text-text-primary mb-1">Skills to fix first</h3>
              <ol className="list-decimal list-inside text-sm text-text-secondary space-y-1">
                {result.agent_plan.skills_to_fix_first.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ol>
            </div>
          )}
          {result.agent_plan.interview_topics_to_prioritize.length > 0 && (
            <div className="mb-3">
              <h3 className="text-xs font-semibold text-text-primary mb-1">Interview topics to prioritize</h3>
              <ul className="list-disc list-inside text-sm text-text-secondary space-y-1">
                {result.agent_plan.interview_topics_to_prioritize.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          {result.agent_plan.study_order.length > 0 && (
            <div className="mb-3">
              <h3 className="text-xs font-semibold text-text-primary mb-1">Study order</h3>
              <ol className="list-decimal list-inside text-sm text-text-secondary space-y-1">
                {result.agent_plan.study_order.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ol>
            </div>
          )}
          {result.agent_plan.concrete_edit_actions.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-text-primary mb-1">Concrete edit actions</h3>
              <ul className="list-disc list-inside text-sm text-text-secondary space-y-1">
                {result.agent_plan.concrete_edit_actions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Rewrite Suggestions */}
      {result.rewrite_suggestions.length > 0 && (
        <div className="card p-4">
          <button
            onClick={() => setSuggestionsOpen(!suggestionsOpen)}
            className="flex items-center justify-between w-full"
          >
            <div className="flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-amber-500" />
              <span className="text-sm font-semibold text-text-primary">
                Rewrite Suggestions ({result.rewrite_suggestions.length})
              </span>
            </div>
            {suggestionsOpen ? (
              <ChevronUp className="w-4 h-4 text-text-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-text-muted" />
            )}
          </button>
          {suggestionsOpen && (
            <ul className="mt-3 space-y-2">
              {result.rewrite_suggestions.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                  <span className="text-sage-500 font-semibold shrink-0">{i + 1}.</span>
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* No CV warning */}
      {score === 0 && result.matched_keywords.length === 0 && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200">
          <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-700">
            Low confidence result. Make sure your CV is loaded and the LLM provider is configured correctly.
          </p>
        </div>
      )}
    </div>
  )
}

function ScoreGauge({ score, color }: { score: number; color: string }) {
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (score / 100) * circumference

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width="140" height="140" className="-rotate-90">
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke="#E8E5DF"
          strokeWidth="10"
        />
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.8s ease' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-3xl font-bold text-text-primary" style={{ color }}>
          {score}%
        </span>
        <span className="text-xs text-text-muted">fit score</span>
      </div>
    </div>
  )
}

function SpinnerIcon() {
  return (
    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
