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
import { scoreJD, type ScoreEvidenceRecord, type ScoreResult } from '../lib/api'
import { scoreColorHex } from '../lib/utils'
import toast from 'react-hot-toast'

function normalizeEvidenceValue(value: string): string {
  return value.trim().toLowerCase()
}

function citationRange(citation: { line_start?: number; line_end?: number }): string {
  if (!citation.line_start && !citation.line_end) {
    return ''
  }
  if (!citation.line_start || !citation.line_end) {
    return `${citation.line_start || citation.line_end}`
  }
  if (citation.line_start === citation.line_end) {
    return `${citation.line_start}`
  }
  return `${citation.line_start}-${citation.line_end}`
}

function attachEvidence(
  values: string[],
  evidenceRows: ScoreEvidenceRecord[],
): Array<{ value: string; evidence?: ScoreEvidenceRecord }> {
  const buckets = new Map<string, ScoreEvidenceRecord[]>()
  for (const row of evidenceRows) {
    const key = normalizeEvidenceValue(row.value || '')
    if (!key) {
      continue
    }
    const rows = buckets.get(key) || []
    rows.push(row)
    buckets.set(key, rows)
  }

  return values.map((value) => {
    const key = normalizeEvidenceValue(value)
    const rows = buckets.get(key)
    if (rows && rows.length > 0) {
      return { value, evidence: rows.shift() }
    }
    return { value }
  })
}

export default function ScoreJD() {
  const [jd, setJd] = useState('')
  const [jobTitle, setJobTitle] = useState('')
  const [company, setCompany] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScoreResult | null>(null)
  const [suggestionsOpen, setSuggestionsOpen] = useState(true)
  const [savedId, setSavedId] = useState<number | null>(null)
  const [showTimeline, setShowTimeline] = useState(true)

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
            <>
              <ResultPanel result={result} savedId={savedId} suggestionsOpen={suggestionsOpen} setSuggestionsOpen={setSuggestionsOpen} />
              <RunTimelinePanel result={result} isOpen={showTimeline} onToggle={() => setShowTimeline((prev) => !prev)} />
            </>
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
  const matchedKeywordRows = attachEvidence(result.matched_keywords, result.matched_keyword_evidence || [])
  const missingKeywordRows = attachEvidence(result.missing_keywords, result.missing_keyword_evidence || [])
  const rewriteSuggestionRows = attachEvidence(result.rewrite_suggestions, result.rewrite_suggestion_evidence || [])

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
      {matchedKeywordRows.length > 0 && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4 text-sage-500" />
            <span className="text-sm font-semibold text-text-primary">
              Matched ({matchedKeywordRows.length})
            </span>
          </div>
          <div className="space-y-2">
            {matchedKeywordRows.map((item, i) => {
              const citations = item.evidence?.cv_citations || []
              const phraseCitations = item.evidence?.jd_phrase_citations || []
              const hasEvidence = citations.length > 0 || phraseCitations.length > 0

              return (
                <div key={`${item.value}-${i}`} className="p-2 rounded-lg border border-sage-100 bg-sage-50">
                  <div className="flex items-start justify-between gap-2">
                    <span className="keyword-matched">{item.value}</span>
                    {hasEvidence && (
                      <span className="text-[10px] text-sage-700">
                        {citations.length > 0 ? `${citations.length} CV` : ''}
                        {citations.length > 0 && phraseCitations.length > 0 ? ' · ' : ''}
                        {phraseCitations.length > 0 ? `${phraseCitations.length} JD` : ''}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 space-y-1">
                    {citations.map((citation, idx) => {
                      const lines = citationRange(citation)
                      return (
                        <div key={`${item.value}-cv-${idx}`} className="text-xs text-text-secondary">
                          <span className="font-semibold">CV:</span>{' '}
                          {citation.section_id ? `section ${citation.section_id}` : 'CV evidence'}
                          {lines ? ` (lines ${lines})` : ''}{' '}
                          {citation.snippet ? `— ${citation.snippet}` : ''}
                        </div>
                      )
                    })}
                    {phraseCitations.map((citation, idx) => {
                      const lines = citationRange(citation)
                      return (
                        <div key={`${item.value}-jd-${idx}`} className="text-xs text-text-secondary">
                          <span className="font-semibold">JD:</span>{' '}
                          {citation.phrase_id ? `phrase ${citation.phrase_id}` : 'JD evidence'}
                          {lines ? ` (lines ${lines})` : ''}{' '}
                          {citation.snippet ? `— ${citation.snippet}` : ''}
                        </div>
                      )
                    })}
                    {item.evidence?.evidence_missing_reason && !hasEvidence && (
                      <p className="text-xs text-amber-700">
                        {item.evidence.evidence_missing_reason}
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Missing keywords */}
      {missingKeywordRows.length > 0 && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <XCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-semibold text-text-primary">
              Missing ({missingKeywordRows.length})
            </span>
          </div>
          <div className="space-y-2">
            {missingKeywordRows.map((item, i) => {
              const citations = item.evidence?.cv_citations || []
              const phraseCitations = item.evidence?.jd_phrase_citations || []
              const hasEvidence = citations.length > 0 || phraseCitations.length > 0

              return (
                <div key={`${item.value}-${i}`} className="p-2 rounded-lg border border-red-100 bg-red-50">
                  <div className="flex items-start justify-between gap-2">
                    <span className="keyword-missing">{item.value}</span>
                    {hasEvidence && (
                      <span className="text-[10px] text-red-700">
                        {citations.length > 0 ? `${citations.length} CV` : ''}
                        {citations.length > 0 && phraseCitations.length > 0 ? ' · ' : ''}
                        {phraseCitations.length > 0 ? `${phraseCitations.length} JD` : ''}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 space-y-1">
                    {citations.map((citation, idx) => {
                      const lines = citationRange(citation)
                      return (
                        <div key={`${item.value}-cv-${idx}`} className="text-xs text-text-secondary">
                          <span className="font-semibold">CV:</span>{' '}
                          {citation.section_id ? `section ${citation.section_id}` : 'CV evidence'}
                          {lines ? ` (lines ${lines})` : ''}{' '}
                          {citation.snippet ? `— ${citation.snippet}` : ''}
                        </div>
                      )
                    })}
                    {phraseCitations.map((citation, idx) => {
                      const lines = citationRange(citation)
                      return (
                        <div key={`${item.value}-jd-${idx}`} className="text-xs text-text-secondary">
                          <span className="font-semibold">JD:</span>{' '}
                          {citation.phrase_id ? `phrase ${citation.phrase_id}` : 'JD evidence'}
                          {lines ? ` (lines ${lines})` : ''}{' '}
                          {citation.snippet ? `— ${citation.snippet}` : ''}
                        </div>
                      )
                    })}
                    {item.evidence?.evidence_missing_reason && !hasEvidence && (
                      <p className="text-xs text-amber-700">
                        {item.evidence.evidence_missing_reason}
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
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

      {result.reason && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-text-secondary" />
            <span className="text-sm font-semibold text-text-primary">Why this score</span>
          </div>
          <p className="text-sm text-text-secondary leading-relaxed">{result.reason}</p>
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
      {rewriteSuggestionRows.length > 0 && (
        <div className="card p-4">
          <button
            onClick={() => setSuggestionsOpen(!suggestionsOpen)}
            className="flex items-center justify-between w-full"
          >
            <div className="flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-amber-500" />
              <span className="text-sm font-semibold text-text-primary">
                Rewrite Suggestions ({rewriteSuggestionRows.length})
              </span>
            </div>
            {suggestionsOpen ? (
              <ChevronUp className="w-4 h-4 text-text-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-text-muted" />
            )}
          </button>
          {suggestionsOpen && (
            <ul className="mt-3 space-y-3">
              {rewriteSuggestionRows.map((item, i) => {
                const citations = item.evidence?.cv_citations || []
                const phraseCitations = item.evidence?.jd_phrase_citations || []
                const hasEvidence = citations.length > 0 || phraseCitations.length > 0

                return (
                  <li key={`${item.value}-${i}`} className="p-2 rounded-lg border border-amber-100 bg-amber-50">
                    <div className="flex items-start gap-2">
                      <span className="text-sage-500 font-semibold shrink-0 mt-0.5">{i + 1}.</span>
                      <div>
                        <p className="text-sm text-text-secondary leading-relaxed">{item.value}</p>
                        {item.evidence?.evidence_missing_reason && !hasEvidence && (
                          <p className="text-xs text-amber-700 mt-1">
                            {item.evidence.evidence_missing_reason}
                          </p>
                        )}
                        {citations.length > 0 && (
                          <p className="text-xs text-text-secondary mt-1">
                            CV: {citations.length} citation{citations.length > 1 ? 's' : ''}
                          </p>
                        )}
                        {phraseCitations.length > 0 && (
                          <p className="text-xs text-text-secondary">
                            JD: {phraseCitations.length} citation{phraseCitations.length > 1 ? 's' : ''}
                          </p>
                        )}
                        {citations.map((citation, index) => {
                          const lines = citationRange(citation)
                          return (
                            <div key={`${item.value}-cv-${index}`} className="text-xs text-text-secondary">
                              {citation.section_id ? `section ${citation.section_id}` : 'CV evidence'}
                              {lines ? ` (lines ${lines})` : ''}{' '}
                              {citation.snippet ? `— ${citation.snippet}` : ''}
                            </div>
                          )
                        })}
                        {phraseCitations.map((citation, index) => {
                          const lines = citationRange(citation)
                          return (
                            <div key={`${item.value}-jd-${index}`} className="text-xs text-text-secondary">
                              {citation.phrase_id ? `phrase ${citation.phrase_id}` : 'JD evidence'}
                              {lines ? ` (lines ${lines})` : ''}{' '}
                              {citation.snippet ? `— ${citation.snippet}` : ''}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  </li>
                )
              })}
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

function RunTimelinePanel({
  result,
  isOpen,
  onToggle,
}: {
  result: ScoreResult
  isOpen: boolean
  onToggle: () => void
}) {
  if (!result.run) return null

  const transitions = result.run_transitions || []
  const artifacts = result.run_artifacts || []

  return (
    <div className="card p-4">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between text-sm font-semibold text-text-primary mb-3"
      >
        <span>Execution timeline</span>
        {isOpen ? <ChevronUp className="w-4 h-4 text-text-muted" /> : <ChevronDown className="w-4 h-4 text-text-muted" />}
      </button>
      <div className="text-xs text-text-secondary mb-2">
        Run {result.run.id.slice(0, 10)} • {result.run_status || result.run.status}
      </div>
      {isOpen && (
        <div className="space-y-2">
          <div className="text-xs text-text-secondary">
            {result.run.attempt_count} attempts • {transitions.length} transitions • {artifacts.length} artifacts
          </div>
          <div className="text-xs font-medium text-text-primary">
            Status: {result.run_status || result.run.status} • Current: {result.run.current_state}
          </div>
          {result.run.failure_reason && (
            <div className="text-xs text-red-500">Error: {result.run.failure_reason}</div>
          )}
          <div className="space-y-1.5">
            {transitions.length > 0 ? transitions.map((t) => (
              <div
                key={t.id}
                className={`px-2 py-1 rounded text-[11px] border ${
                  t.failure_reason ? 'bg-red-50 border-red-200 text-red-700' : 'bg-sage-50 border-sage-200 text-text-primary'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span>{t.trigger}</span>
                  <span>{t.next_state}</span>
                </div>
                {t.failure_reason && <div className="mt-0.5">{t.failure_reason}</div>}
              </div>
            )) : (
              <p className="text-text-muted">No transitions recorded yet.</p>
            )}
          </div>
          <div className="text-xs font-medium text-text-primary">
            Artifacts
          </div>
          <div className="space-y-1.5">
            {artifacts.length > 0 ? artifacts.map((artifact) => (
              <div key={artifact.id} className="px-2 py-1 rounded border border-slate-200 text-[11px]">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{artifact.step}</span>
                  <span>attempt {artifact.attempt}</span>
                </div>
                <p className="text-text-muted mt-0.5">
                  {artifact.payload ? Object.keys(artifact.payload).length : 0} payload fields
                </p>
              </div>
            )) : (
              <p className="text-text-muted">No artifacts recorded yet.</p>
            )}
          </div>
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
