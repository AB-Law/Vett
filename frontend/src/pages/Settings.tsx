import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Settings2, CheckCircle2, XCircle, Loader2, Trash2, Download, Database, Upload, FileText } from 'lucide-react'
import {
  getSettings,
  updateSettings,
  testConnection,
  clearHistory,
  getEmbeddingProgress,
  getInterviewDocumentProgress,
  uploadInterviewDocument,
  type AppSettings,
  type InterviewKnowledgeDocument,
  type InterviewKnowledgeDocumentProgress,
} from '../lib/api'
import toast from 'react-hot-toast'

type Provider = 'claude' | 'openai' | 'azure_openai' | 'ollama' | 'lm_studio'
type InterviewDocumentWithProgress = {
  id: number
  owner_type: 'global' | 'job'
  job_id: number | null
  source_filename: string
  content_type: string
  status: string
  error_message: string | null
  parser_version: string | null
  source_ref: string | null
  created_at: string
  created_by_user_id: string | null
  total_chunks: number
  embedded_chunks: number
  parsed_word_count: number
  progress_percent: number
}

const PROVIDERS: { id: Provider; label: string }[] = [
  { id: 'claude', label: 'Claude' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'azure_openai', label: 'Azure OpenAI' },
  { id: 'ollama', label: 'Ollama' },
  { id: 'lm_studio', label: 'LM Studio' },
]

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [embeddingProgress, setEmbeddingProgress] = useState<{ total: number; embedded: number; percent: number } | null>(null)
  const [interviewDocuments, setInterviewDocuments] = useState<InterviewDocumentWithProgress[]>([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [uploadingDocument, setUploadingDocument] = useState(false)
  const documentsPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const progressPercent = useCallback((embeddedChunks: number, totalChunks: number): number => {
    if (totalChunks <= 0) return 0
    const value = (embeddedChunks / totalChunks) * 100
    if (!Number.isFinite(value)) return 0
    return Math.round(value * 10) / 10
  }, [])

  const toDocumentWithProgress = useCallback(
    (doc: InterviewKnowledgeDocument | InterviewKnowledgeDocumentProgress): InterviewDocumentWithProgress => ({
      id: doc.id,
      owner_type: doc.owner_type,
      job_id: doc.job_id,
      source_filename: doc.source_filename,
      content_type: 'content_type' in doc ? doc.content_type : '',
      status: doc.status,
      error_message: doc.error_message,
      parser_version: 'parser_version' in doc ? doc.parser_version : null,
      source_ref: 'source_ref' in doc ? doc.source_ref : null,
      created_at: doc.created_at,
      created_by_user_id: doc.created_by_user_id,
      total_chunks: doc.total_chunks,
      embedded_chunks: doc.embedded_chunks,
      parsed_word_count: doc.parsed_word_count,
      progress_percent:
        'progress_percent' in doc ? doc.progress_percent : progressPercent(doc.embedded_chunks, doc.total_chunks),
    }),
    [progressPercent],
  )

  const stopDocumentPolling = (): void => {
    if (documentsPollRef.current) {
      clearInterval(documentsPollRef.current)
      documentsPollRef.current = null
    }
  }

  const isDocumentInProgress = useCallback((status: string): boolean => {
    const normalized = status.toLowerCase()
    return normalized.includes('pending') || normalized.includes('processing')
  }, [])

  const refreshInterviewDocuments = useCallback(async () => {
    try {
      const [ep, docs] = await Promise.all([getEmbeddingProgress(), getInterviewDocumentProgress()])
      setEmbeddingProgress(ep)
      const mappedDocs = docs.map(toDocumentWithProgress)
      setInterviewDocuments(mappedDocs)

      const shouldKeepPolling = mappedDocs.some((doc) => isDocumentInProgress(doc.status)) || ep.percent < 100
      if (!shouldKeepPolling) {
        stopDocumentPolling()
      }
    } catch {
      // Keep existing state and try again next interval if polling is still active.
    }
  }, [toDocumentWithProgress])

  const ensureDocumentPolling = useCallback((): void => {
    if (documentsPollRef.current) return
    documentsPollRef.current = setInterval(() => {
      void refreshInterviewDocuments()
    }, 2000)
  }, [refreshInterviewDocuments])

  // Pending changes
  const [activeProvider, setActiveProvider] = useState<Provider>('ollama')
  const [fields, setFields] = useState<Record<string, string>>({})

  useEffect(() => {
    setLoading(true)
    setDocumentsLoading(true)
    const load = async () => {
      try {
        const [s, ep, docs] = await Promise.all([getSettings(), getEmbeddingProgress(), getInterviewDocumentProgress()])
        const mappedDocs = docs.map(toDocumentWithProgress)

        setSettings(s)
        setActiveProvider(s.active_provider as Provider)
        setFields({
          claude_model: s.claude_model,
          openai_model: s.openai_model,
          azure_openai_endpoint: s.azure_openai_endpoint,
          azure_openai_deployment: s.azure_openai_deployment,
          azure_openai_api_version: s.azure_openai_api_version,
          ollama_base_url: s.ollama_base_url,
          ollama_model: s.ollama_model,
          lm_studio_base_url: s.lm_studio_base_url,
          lm_studio_model: s.lm_studio_model,
        })
        setEmbeddingProgress(ep)
        setInterviewDocuments(mappedDocs)
        if (mappedDocs.some((doc) => isDocumentInProgress(doc.status)) || ep.percent < 100) {
          ensureDocumentPolling()
        }
      } finally {
        setLoading(false)
        setDocumentsLoading(false)
      }
    }

    void load()

    return () => {
      stopDocumentPolling()
    }
  }, [])

  const set = (key: string, value: string) => setFields((f) => ({ ...f, [key]: value }))

  const statusClass = (status: string): string => {
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
      if (!file) return
      setUploadingDocument(true)
      try {
        const newDoc = await uploadInterviewDocument(file)
        const normalizedDoc = toDocumentWithProgress(newDoc)
        setInterviewDocuments((current) => [normalizedDoc, ...current.filter((doc) => doc.id !== normalizedDoc.id)])
        ensureDocumentPolling()
        await refreshInterviewDocuments()
        toast.success('Interview document uploaded')
      } catch (err: unknown) {
        const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Upload failed'
        toast.error(detail)
      } finally {
        setUploadingDocument(false)
      }
    },
    [ensureDocumentPolling, refreshInterviewDocuments, toDocumentWithProgress],
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

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateSettings({ active_provider: activeProvider, ...fields })
      toast.success('Settings saved')
      setTestResult(null)
    } catch {
      toast.error('Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  const handleTestConnection = async () => {
    // Save first, then test
    setSaving(true)
    try {
      await updateSettings({ active_provider: activeProvider, ...fields })
    } catch {
      toast.error('Failed to save settings before test')
      setSaving(false)
      return
    }
    setSaving(false)
    setTesting(true)
    try {
      const res = await testConnection()
      if (res.ok) {
        setTestResult({ ok: true, message: `Connected! Reply: "${res.reply}"` })
      } else {
        setTestResult({ ok: false, message: res.error || 'Connection failed' })
      }
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Connection failed'
      setTestResult({ ok: false, message: detail })
    } finally {
      setTesting(false)
    }
  }

  const handleClearHistory = async () => {
    if (!confirm('Clear all scoring history? This cannot be undone.')) return
    await clearHistory()
    toast.success('History cleared')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-text-muted text-sm">Loading…</div>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-serif text-text-primary">Settings</h1>
        <p className="text-text-secondary text-sm mt-1">Configure your LLM providers and preferences</p>
      </div>

      {/* LLM Provider */}
      <div className="card p-5 mb-5">
        <div className="flex items-center gap-2 mb-4">
          <Settings2 className="w-4 h-4 text-text-secondary" />
          <h2 className="section-title">LLM Provider</h2>
        </div>

        {/* Provider tabs */}
        <div className="flex gap-1 p-1 bg-gray-100 rounded-lg mb-5">
          {PROVIDERS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setActiveProvider(id)}
              className={`flex-1 py-1.5 text-xs font-medium rounded transition-colors ${
                activeProvider === id
                  ? 'bg-white text-sage-700 shadow-sm'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Provider-specific fields */}
        {activeProvider === 'claude' && (
          <ProviderFields>
            <ApiKeyField
              label="Anthropic API Key"
              placeholder="sk-ant-…"
              hasKey={settings?.has_anthropic_key ?? false}
              onChange={(v) => set('anthropic_api_key', v)}
            />
            <Field label="Model" value={fields.claude_model} onChange={(v) => set('claude_model', v)}
              placeholder="claude-3-5-sonnet-20241022" />
          </ProviderFields>
        )}

        {activeProvider === 'openai' && (
          <ProviderFields>
            <ApiKeyField
              label="OpenAI API Key"
              placeholder="sk-…"
              hasKey={settings?.has_openai_key ?? false}
              onChange={(v) => set('openai_api_key', v)}
            />
            <Field label="Model" value={fields.openai_model} onChange={(v) => set('openai_model', v)}
              placeholder="gpt-4o" />
          </ProviderFields>
        )}

        {activeProvider === 'azure_openai' && (
          <ProviderFields>
            <ApiKeyField
              label="Azure OpenAI API Key"
              placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
              hasKey={settings?.has_azure_key ?? false}
              onChange={(v) => set('azure_openai_api_key', v)}
            />
            <Field label="Endpoint" value={fields.azure_openai_endpoint}
              onChange={(v) => set('azure_openai_endpoint', v)}
              placeholder="https://YOUR_RESOURCE.openai.azure.com/" />
            <Field label="Deployment Name" value={fields.azure_openai_deployment}
              onChange={(v) => set('azure_openai_deployment', v)}
              placeholder="gpt-4o" />
            <Field label="API Version" value={fields.azure_openai_api_version}
              onChange={(v) => set('azure_openai_api_version', v)}
              placeholder="2024-02-01" />
          </ProviderFields>
        )}

        {activeProvider === 'ollama' && (
          <ProviderFields>
            <Field label="Base URL" value={fields.ollama_base_url}
              onChange={(v) => set('ollama_base_url', v)}
              placeholder="http://host.docker.internal:11434" />
            <Field label="Model" value={fields.ollama_model}
              onChange={(v) => set('ollama_model', v)}
              placeholder="llama3.2" />
          </ProviderFields>
        )}

        {activeProvider === 'lm_studio' && (
          <ProviderFields>
            <Field label="Base URL (OpenAI-compatible)" value={fields.lm_studio_base_url}
              onChange={(v) => set('lm_studio_base_url', v)}
              placeholder="http://host.docker.internal:1234/v1" />
            <Field label="Model Name" value={fields.lm_studio_model}
              onChange={(v) => set('lm_studio_model', v)}
              placeholder="local-model" />
          </ProviderFields>
        )}

        {/* Test result */}
        {testResult && (
          <div className={`flex items-start gap-2 p-3 rounded-lg mt-4 text-sm ${
            testResult.ok ? 'bg-sage-50 border border-sage-200 text-sage-700' : 'bg-red-50 border border-red-200 text-red-700'
          }`}>
            {testResult.ok
              ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
              : <XCircle className="w-4 h-4 shrink-0 mt-0.5" />}
            {testResult.message}
          </div>
        )}

        <div className="flex items-center gap-2 mt-5">
          <button onClick={handleTestConnection} disabled={testing || saving} className="btn-secondary flex items-center gap-2">
            {testing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            Test Connection
          </button>
        </div>
      </div>

      {/* General Settings */}
      <div className="card p-5 mb-5">
        <h2 className="section-title mb-4">General</h2>
        <div className="flex items-center justify-between py-2">
          <div>
            <p className="text-sm font-medium text-text-primary">Save scoring history</p>
            <p className="text-xs text-text-muted">Store JD scoring results in local DB</p>
          </div>
          <Toggle
            enabled={fields.save_history !== 'false'}
            onChange={(v) => set('save_history', v ? 'true' : 'false')}
          />
        </div>
        <div className="border-t border-border pt-3 mt-2">
          <label className="label">Default export format</label>
          <select
            className="input max-w-xs"
            value={fields.default_export_format || 'json'}
            onChange={(e) => set('default_export_format', e.target.value)}
          >
            <option value="json">JSON</option>
            <option value="markdown">Markdown</option>
            <option value="pdf">PDF</option>
          </select>
        </div>
      </div>

      {/* Practice Embeddings */}
      {embeddingProgress && (
        <div className="card p-5 mb-5">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-4 h-4 text-text-secondary" />
            <h2 className="section-title">Practice Question Embeddings</h2>
          </div>
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-text-secondary">{embeddingProgress.embedded} / {embeddingProgress.total} questions embedded</span>
            <span className="font-medium text-text-primary">{embeddingProgress.percent}%</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2">
            <div
              className="bg-sage-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${embeddingProgress.percent}%` }}
            />
          </div>
          {embeddingProgress.percent < 100 && (
            <p className="text-xs text-text-muted mt-2">Embedding is in progress and updates automatically.</p>
          )}
        </div>
      )}

      {/* Interview Documents */}
      <div className="card p-5 mb-5">
        <div className="flex items-center gap-2 mb-4">
          <Upload className="w-4 h-4 text-text-secondary" />
          <h2 className="section-title">Interview Documents</h2>
        </div>

        <div
          {...getInterviewDocsRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isInterviewDocsDragActive
              ? 'border-sage-400 bg-sage-50'
              : 'border-border hover:border-sage-300 hover:bg-gray-50'
          } ${uploadingDocument ? 'opacity-60 pointer-events-none' : ''}`}
        >
          <input {...getInterviewDocsInputProps()} />
          <FileText className="w-7 h-7 text-text-muted mx-auto mb-2" />
          <p className="text-sm font-medium text-text-secondary mb-1">
            {isInterviewDocsDragActive ? 'Drop interview docs here…' : 'Drop or upload PDF / DOCX / DOC / MD / TXT'}
          </p>
          <p className="text-xs text-text-muted">These documents are used as interview knowledge base references.</p>
          {uploadingDocument && <p className="text-xs text-sage-600 mt-2 font-medium">Uploading…</p>}
        </div>

        <div className="mt-4">
          {documentsLoading ? (
            <p className="text-xs text-text-muted">Loading interview documents…</p>
          ) : interviewDocuments.length === 0 ? (
            <p className="text-xs text-text-muted">No interview documents uploaded yet.</p>
          ) : (
            <div className="space-y-2">
              {interviewDocuments.map((doc) => (
                <div key={doc.id} className="border border-border rounded p-2">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <div className="text-text-primary font-medium truncate">{doc.source_filename}</div>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] ${statusClass(doc.status)}`}>
                      {doc.status}
                    </span>
                  </div>
                  {doc.error_message && <p className="text-[11px] text-red-600 mt-1">{doc.error_message}</p>}
                  <div className="mt-2">
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div
                        className="bg-sage-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(0, doc.progress_percent))}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-text-muted mt-1">
                      {doc.total_chunks > 0 ? `${doc.embedded_chunks} / ${doc.total_chunks} chunks` : 'No chunks yet'} ({doc.progress_percent.toFixed(1)}%)
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Data & Storage */}
      <div className="card p-5 mb-8">
        <h2 className="section-title mb-4">Data &amp; Storage</h2>
        <div className="flex flex-wrap gap-2">
          <button onClick={handleClearHistory} className="btn-danger flex items-center gap-2">
            <Trash2 className="w-3.5 h-3.5" />
            Clear all history
          </button>
          <button
            className="btn-secondary flex items-center gap-2 opacity-50 cursor-not-allowed"
            disabled
            title="Coming soon"
          >
            <Download className="w-3.5 h-3.5" />
            Export all data
          </button>
        </div>
        <p className="text-xs text-text-muted mt-3">
          Data is stored in a local SQLite database at <code className="font-mono bg-gray-100 px-1 rounded">/data/vett.db</code>
        </p>
      </div>

      {/* Save button */}
      <div className="flex justify-end">
        <button onClick={handleSave} disabled={saving} className="btn-primary flex items-center gap-2 px-6">
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
          Save Settings
        </button>
      </div>
    </div>
  )
}

function ProviderFields({ children }: { children: React.ReactNode }) {
  return <div className="space-y-3">{children}</div>
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <div>
      <label className="label">{label}</label>
      <input className="input" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
    </div>
  )
}

function ApiKeyField({
  label,
  placeholder,
  hasKey,
  onChange,
}: {
  label: string
  placeholder: string
  hasKey: boolean
  onChange: (v: string) => void
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <label className="label mb-0">{label}</label>
        {hasKey && (
          <span className="inline-flex items-center gap-1 text-xs text-sage-600 font-medium">
            <CheckCircle2 className="w-3 h-3" /> Key set
          </span>
        )}
      </div>
      <input
        type="password"
        className="input"
        placeholder={hasKey ? '••••••••••••••••••• (leave blank to keep)' : placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  )
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
        enabled ? 'bg-sage-500' : 'bg-gray-300'
      }`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
          enabled ? 'translate-x-4.5' : 'translate-x-0.5'
        }`}
      />
    </button>
  )
}
