import { useEffect, useRef, useState } from 'react'
import {
  Search,
  MapPin,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  RefreshCw,
  X,
  AlertTriangle,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getJobs, searchJobs, getRescoreStatus, rescoreJobs, type Job } from '../lib/api'
import { scoreColor, formatDate } from '../lib/utils'
import toast from 'react-hot-toast'

export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [searching, setSearching] = useState(false)
  const [rescoring, setRescoring] = useState(false)
  const [query, setQuery] = useState('')
  const [job, setJob] = useState('')
  const [location, setLocation] = useState('')
  const [source, setSource] = useState('all')
  const [numRecords, setNumRecords] = useState('25')
  const [yearsOfExperience, setYearsOfExperience] = useState('')
  const [returnRaw, setReturnRaw] = useState(false)
  const [rescoreRunId, setRescoreRunId] = useState<string | null>(null)
  const [rescoreMessage, setRescoreMessage] = useState('')
  const rescorePollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const navigate = useNavigate()

  // Filters
  const [minScore, setMinScore] = useState<number | undefined>(undefined)
  const [maxScore, setMaxScore] = useState<number | undefined>(undefined)
  const [scoreFilter, setScoreFilter] = useState<'all' | 'scored' | 'unscored'>('all')
  const [workType, setWorkType] = useState('')
  const [seniority, setSeniority] = useState('')
  const [sortBy, setSortBy] = useState<'title' | 'company' | 'location' | 'score' | 'source' | 'posted_at' | 'created_at'>(
    'created_at',
  )
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

  const reloadJobs = async () => {
    setLoading(true)
    try {
      const refreshed = await getJobs({
        min_score: minScore,
        max_score: maxScore,
        score_status: scoreFilter === 'all' ? undefined : scoreFilter,
        work_type: workType || undefined,
        seniority: seniority || undefined,
        source: source === 'all' ? undefined : source,
        sort_by: sortBy === 'score' ? 'fit_score' : sortBy,
        sort_dir: sortDirection,
      })
      setJobs(refreshed)
    } catch {
      toast.error('Failed to load jobs', { position: 'top-right' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reloadJobs()
  }, [minScore, maxScore, scoreFilter, workType, seniority, sortBy, sortDirection, source])

  useEffect(() => {
    return () => {
      if (rescorePollRef.current) {
        clearInterval(rescorePollRef.current)
      }
    }
  }, [])

  const stopRescorePolling = () => {
    if (rescorePollRef.current) {
      clearInterval(rescorePollRef.current)
      rescorePollRef.current = null
    }
  }

  const handleSearch = async () => {
    const q = query.trim()
    const jobText = job.trim()
    const parsedNumRecords = Number.parseInt(numRecords, 10)
    const years = yearsOfExperience.trim() ? Number(yearsOfExperience) : undefined

    if (!q && !jobText) {
      toast.error('Enter a role or job title')
      return
    }

    if (!Number.isFinite(parsedNumRecords) || parsedNumRecords < 10) {
      toast.error('Number of records must be at least 10', { position: 'top-right', duration: 5000 })
      return
    }

    if (years !== undefined && !Number.isFinite(years)) {
      toast.error('Years of experience must be a number', { position: 'top-right', duration: 5000 })
      return
    }

    setSearching(true)
    try {
      const res = await searchJobs({
        query: q,
        role: q,
        job: jobText || undefined,
        location: location || undefined,
        source,
        years_of_experience: years,
        num_records: parsedNumRecords,
        return_raw: returnRaw,
      })
      toast(res.message || 'Phase 2 feature — enable Celery to scrape jobs', {
        icon: '⚠️',
        duration: 5000,
        position: 'top-right',
      })
      await reloadJobs()
    } catch (error: unknown) {
      const responseData = (error as { response?: { data?: unknown } })?.response?.data
      let message = (error as Error)?.message || 'Search failed'

      if (responseData && typeof responseData === 'object' && 'detail' in responseData) {
        const detail = (responseData as { detail?: unknown }).detail
        if (typeof detail === 'string') {
          message = detail
        } else if (Array.isArray(detail) && detail.length > 0) {
          const first = detail[0]
          if (typeof first === 'string') {
            message = first
          } else if (first && typeof first === 'object' && 'msg' in first) {
            message = String((first as { msg?: unknown }).msg)
          }
        }
      }

      toast.error(String(message), { position: 'top-right', duration: 6000 })
    } finally {
      setSearching(false)
    }
  }

  const handleRescoreAll = async () => {
    stopRescorePolling()
    setRescoring(true)

    try {
      const res = await rescoreJobs({
        source: source === 'all' ? undefined : source,
      })
      setRescoreRunId(res.run_id)
      setRescoreMessage(res.message)

      if (!res.run_id || res.total_jobs === 0) {
        await reloadJobs()
        toast.info(res.message || `No jobs matched the selected filters.`)
        setRescoring(false)
        setRescoreRunId(null)
        return
      }

      const pollRun = async () => {
        try {
          const status = await getRescoreStatus(res.run_id)
          await reloadJobs()
          setRescoreMessage(status.message)
          if (status.status === 'completed' || status.status === 'failed') {
            stopRescorePolling()
            setRescoring(false)
            setRescoreRunId(null)
            toast.success(`Rescoring complete. ${status.scored_count} scored, ${status.failed_count} failed.`, {
              position: 'top-right',
            })
            return true
          }
          return false
        } catch {
          stopRescorePolling()
          setRescoring(false)
          setRescoreRunId(null)
          toast.error('Rescore progress check failed.')
          return true
        }
      }

      const isDone = await pollRun()
      if (!isDone) {
        rescorePollRef.current = setInterval(() => {
          void pollRun()
        }, 2000)
      }
    } catch {
      stopRescorePolling()
      setRescoring(false)
      setRescoreRunId(null)
      setRescoreMessage('')
      toast.error('Rescoring failed. Upload a CV first.', { position: 'top-right' })
    }
  }

  const handleSelectJob = (id: number) => {
    navigate(`/jobs/${id}`)
  }

  const handleSort = (nextSortBy: typeof sortBy) => {
    if (sortBy === nextSortBy) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortBy(nextSortBy)
    setSortDirection('desc')
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-serif text-text-primary">Jobs</h1>
        <p className="text-text-secondary text-sm mt-1">Find and score jobs automatically</p>
      </div>

      {/* Jobs scraping + storage notice */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-sky-50 border border-sky-200 mb-6">
        <AlertTriangle className="w-4 h-4 text-sky-500 shrink-0 mt-0.5" />
        <p className="text-sm text-sky-700">
          Search writes scraped jobs to the database for later processing. This keeps raw snapshots for traceability.
        </p>
      </div>

      {/* Search bar */}
      <div className="card p-4 mb-4">
        <div className="flex flex-wrap gap-3">
          <div className="flex-1 min-w-[180px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              className="input pl-9"
              placeholder="Role or keywords…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <div className="relative">
            <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              className="input pl-9 w-44"
              placeholder="Location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>
          <select className="input w-40" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="all">All Sources</option>
            <option value="linkedin">LinkedIn</option>
            <option value="indeed">Indeed</option>
            <option value="naukri">Naukri</option>
          </select>
          <button onClick={handleSearch} disabled={searching} className="btn-primary flex items-center gap-2">
            {searching ? <SpinnerIcon /> : <Search className="w-4 h-4" />}
            Search Jobs
          </button>
          <button
            onClick={handleRescoreAll}
            disabled={rescoring || loading}
            className="btn-primary flex items-center gap-2"
          >
            {rescoring ? <SpinnerIcon /> : <RefreshCw className="w-4 h-4" />}
            Rescore all jobs
          </button>
        </div>
      {(rescoring || rescoreMessage) && (
        <p className="mt-3 text-xs text-text-muted">
          {rescoreRunId ? `Rescore run ${rescoreRunId.slice(0, 8)}: ` : ''}{rescoreMessage || 'Rescore started.'}
        </p>
      )}

        {/* Advanced scrape options */}
        <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-border items-end">
          <div className="relative flex-1 min-w-[180px]">
            <input
              className="input"
              placeholder="Optional alternate job title"
              value={job}
              onChange={(e) => setJob(e.target.value)}
            />
          </div>
          <div className="relative">
            <input
              className="input w-36"
              type="number"
              min={10}
              placeholder="Records"
              value={numRecords}
              onChange={(e) => setNumRecords(e.target.value)}
            />
          </div>
          <div className="relative">
            <input
              className="input w-40"
              type="number"
              min={0}
              placeholder="Years exp"
              value={yearsOfExperience}
              onChange={(e) => setYearsOfExperience(e.target.value)}
            />
          </div>
          <label className="inline-flex items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={returnRaw}
              onChange={(e) => setReturnRaw(e.target.checked)}
            />
            Store raw payload
          </label>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mt-3 pt-3 border-t border-border items-center">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted font-medium">Score</span>
            <input
              type="number"
              min={0}
              max={100}
              className="input w-20 py-1 text-xs"
              placeholder="Min"
              value={minScore === undefined ? '' : minScore}
              onChange={(e) => setMinScore(e.target.value === '' ? undefined : Number(e.target.value))}
            />
            <span className="text-xs text-text-muted">to</span>
            <input
              type="number"
              min={0}
              max={100}
              className="input w-20 py-1 text-xs"
              placeholder="Max"
              value={maxScore === undefined ? '' : maxScore}
              onChange={(e) => setMaxScore(e.target.value === '' ? undefined : Number(e.target.value))}
            />
          </div>
          <select
            className="input w-44 text-xs py-1"
            value={scoreFilter}
            onChange={(e) => setScoreFilter(e.target.value as 'all' | 'scored' | 'unscored')}
          >
            <option value="all">Any score status</option>
            <option value="scored">Only scored</option>
            <option value="unscored">Only unscored</option>
          </select>
          <select className="input w-36 text-xs py-1" value={workType} onChange={(e) => setWorkType(e.target.value)}>
            <option value="">Any Work Type</option>
            <option value="remote">Remote</option>
            <option value="hybrid">Hybrid</option>
            <option value="onsite">Onsite</option>
          </select>
          <select className="input w-36 text-xs py-1" value={seniority} onChange={(e) => setSeniority(e.target.value)}>
            <option value="">Any Seniority</option>
            <option value="junior">Junior</option>
            <option value="mid">Mid</option>
            <option value="senior">Senior</option>
            <option value="lead">Lead</option>
          </select>
          {(minScore !== undefined || maxScore !== undefined || scoreFilter !== 'all' || workType || seniority) && (
            <button
              onClick={() => {
                setMinScore(undefined)
                setMaxScore(undefined)
                setScoreFilter('all')
                setWorkType('')
                setSeniority('')
              }}
              className="text-xs text-text-muted hover:text-red-500 flex items-center gap-1"
            >
              <X className="w-3 h-3" /> Clear filters
            </button>
          )}
        </div>
      </div>

      <div>
        {/* Jobs table */}
        <div className="card overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-16 text-text-muted text-sm">Loading…</div>
          ) : jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <Search className="w-8 h-8 text-text-muted mb-3" />
              <p className="text-sm font-medium text-text-secondary mb-1">No jobs yet</p>
              <p className="text-xs text-text-muted">Search for a role to populate this list</p>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-border">
                  {[
                    { label: 'Job Title', key: 'title' as const },
                    { label: 'Company', key: 'company' as const },
                    { label: 'Location', key: 'location' as const },
                    { label: 'Score', key: 'score' as const },
                    { label: 'Source', key: 'source' as const },
                    { label: 'Posted', key: 'posted_at' as const },
                    { label: '', key: null as const },
                  ].map((h) => {
                    if (!h.key) {
                      return <th key={h.label} className="text-left px-4 py-3 text-text-muted font-medium text-xs uppercase tracking-wide" />
                    }
                    return (
                      <th
                        key={h.label}
                        className="text-left px-4 py-3 text-text-muted font-medium text-xs uppercase tracking-wide"
                      >
                        <button
                          type="button"
                          onClick={() => handleSort(h.key)}
                          className="inline-flex items-center gap-1 hover:text-text-primary"
                          title={`Sort by ${h.label}`}
                        >
                          {h.label}
                          {sortBy === h.key &&
                            (sortDirection === 'asc' ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            ))}
                        </button>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.id}
                    className="border-b border-border last:border-0 hover:bg-gray-50/50 transition-colors cursor-pointer"
                    onClick={() => handleSelectJob(job.id)}
                  >
                    <td className="px-4 py-3 font-medium text-text-primary max-w-[200px] truncate">
                      {job.title || '—'}
                    </td>
                    <td className="px-4 py-3 text-text-secondary">{job.company || '—'}</td>
                    <td className="px-4 py-3 text-text-muted text-xs">{job.location || '—'}</td>
                    <td className="px-4 py-3">
                      {typeof job.fit_score === 'number' ? (
                        <span className={scoreColor(job.fit_score)}>{Math.round(job.fit_score)}%</span>
                      ) : (
                        <span className="text-text-muted text-xs">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-text-muted capitalize">{job.source || '—'}</td>
                    <td className="px-4 py-3 text-xs text-text-muted">
                      {job.posted_at ? formatDate(job.posted_at) : '—'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <ChevronRight className="w-4 h-4 text-text-muted" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
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
