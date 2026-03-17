import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { FileText, BarChart2, TrendingUp, ChevronRight } from 'lucide-react'
import { getCV, getJobs, type CV, type Job } from '../lib/api'
import { formatDate, scoreColor } from '../lib/utils'
import toast from 'react-hot-toast'

export default function Dashboard() {
  const [cv, setCv] = useState<CV | null>(null)
  const [topRecentJobs, setTopRecentJobs] = useState<Job[]>([])
  const [allScoredJobs, setAllScoredJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [cvData, topJobs, scoredJobs] = await Promise.all([
          getCV(),
          getJobs({
            score_status: 'scored',
            sort_by: 'fit_score',
            sort_dir: 'desc',
            limit: 3,
          }),
          getJobs({
            score_status: 'scored',
            limit: 10000,
          }),
        ])
        setCv(cvData)
        setTopRecentJobs(topJobs)
        setAllScoredJobs(scoredJobs)
      } catch {
        toast.error('Failed to load dashboard data')
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [])

  const scoredEntries = allScoredJobs.filter((job) => typeof job.fit_score === 'number')
  const avgScore = scoredEntries.length > 0
    ? Math.round(scoredEntries.reduce((s, job) => s + (job.fit_score || 0), 0) / scoredEntries.length)
    : null

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-text-muted text-sm">Loading…</div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-serif text-text-primary">Dashboard</h1>
        <p className="text-text-secondary text-sm mt-1">
          Your local-first career intelligence platform
        </p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard
          icon={<FileText className="w-4 h-4 text-sage-600" />}
          label="CV Loaded"
          value={cv ? cv.filename : '—'}
          sub={cv ? formatDate(cv.created_at) : 'No CV uploaded yet'}
          accent={cv ? 'green' : 'gray'}
        />
        <StatCard
          icon={<BarChart2 className="w-4 h-4 text-sage-600" />}
          label="Jobs Scored"
          value={String(scoredEntries.length)}
          sub="in jobs"
          accent="green"
        />
        <StatCard
          icon={<TrendingUp className="w-4 h-4 text-sage-600" />}
          label="Avg Fit Score"
          value={avgScore !== null ? `${avgScore}%` : '—'}
          sub={avgScore !== null ? 'across all scored JDs' : 'No scores yet'}
          accent={avgScore !== null ? (avgScore >= 70 ? 'green' : avgScore >= 50 ? 'amber' : 'red') : 'gray'}
        />
      </div>

      {/* Recent Scores */}
      <div className="card overflow-hidden">
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <h2 className="section-title">Recent Scores</h2>
            {topRecentJobs.length > 0 && (
              <Link to="/jobs" className="text-xs text-sage-600 hover:text-sage-700 font-medium flex items-center gap-1">
                View jobs <ChevronRight className="w-3 h-3" />
              </Link>
            )}
        </div>

        {topRecentJobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="w-12 h-12 rounded-full bg-sage-50 flex items-center justify-center mb-4">
              <ClipboardIcon />
            </div>
            <p className="text-text-secondary text-sm font-medium mb-1">No scores yet</p>
            <p className="text-text-muted text-xs mb-4">
              Search and score jobs to see top results
            </p>
            <Link to="/score" className="btn-primary">
              Score a JD
            </Link>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-border">
                <th className="text-left px-5 py-3 text-text-muted font-medium text-xs uppercase tracking-wide">
                  Job Title
                </th>
                <th className="text-left px-4 py-3 text-text-muted font-medium text-xs uppercase tracking-wide">
                  Company
                </th>
                <th className="text-left px-4 py-3 text-text-muted font-medium text-xs uppercase tracking-wide">
                  Fit Score
                </th>
                <th className="text-left px-4 py-3 text-text-muted font-medium text-xs uppercase tracking-wide">
                  Date
                </th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {topRecentJobs.map((item) => (
                <tr key={item.id} className="border-b border-border last:border-0 hover:bg-gray-50/50 transition-colors">
                  <td className="px-5 py-3 font-medium text-text-primary">
                    {item.title || <span className="text-text-muted italic">Untitled</span>}
                  </td>
                  <td className="px-4 py-3 text-text-secondary">
                    {item.company || '—'}
                  </td>
                  <td className="px-4 py-3">
                    {typeof item.fit_score === 'number' ? (
                      <span className={scoreColor(item.fit_score)}>
                        {Math.round(item.fit_score)}%
                      </span>
                    ) : (
                      <span className="text-text-muted text-xs">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-text-muted">
                    {formatDate(item.posted_at || item.created_at)}
                  </td>
                  <td className="px-4 py-3" />
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub: string
  accent: 'green' | 'amber' | 'red' | 'gray'
}) {
  const dotColors: Record<string, string> = {
    green: 'bg-sage-500',
    amber: 'bg-amber-400',
    red: 'bg-red-400',
    gray: 'bg-gray-300',
  }

  return (
    <div className="card px-5 py-4">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <span className="text-xs font-medium text-text-muted uppercase tracking-wide">{label}</span>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-xl font-semibold text-text-primary truncate">{value}</span>
        <div className={`w-2 h-2 rounded-full mb-1 shrink-0 ${dotColors[accent]}`} />
      </div>
      <p className="text-xs text-text-muted mt-1">{sub}</p>
    </div>
  )
}

function ClipboardIcon() {
  return (
    <svg className="w-6 h-6 text-sage-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
    </svg>
  )
}
