import { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Trash2, RefreshCw, Copy, CheckCircle2, ShieldCheck } from 'lucide-react'
import { uploadCV, getCV, deleteCV, type CV } from '../lib/api'
import { formatFileSize, formatDate } from '../lib/utils'
import toast from 'react-hot-toast'

export default function MyCv() {
  const [cv, setCv] = useState<CV | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    getCV()
      .then(setCv)
      .finally(() => setLoading(false))
  }, [])

  const handleDrop = useCallback(async (files: File[]) => {
    const file = files[0]
    if (!file) return
    setUploading(true)
    try {
      const result = await uploadCV(file)
      setCv(result)
      toast.success('CV uploaded and parsed!')
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Upload failed'
      toast.error(msg)
    } finally {
      setUploading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/markdown': ['.md', '.markdown'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    disabled: uploading,
  })

  const handleDelete = async () => {
    if (!confirm('Remove your CV? This will delete all parsed data.')) return
    await deleteCV()
    setCv(null)
    toast.success('CV removed')
  }

  const handleCopy = () => {
    if (!cv) return
    navigator.clipboard.writeText(cv.parsed_text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-text-muted text-sm">Loading…</div>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-serif text-text-primary">My CV</h1>
        <p className="text-text-secondary text-sm mt-1">Manage your uploaded resume</p>
      </div>

      {/* Upload Zone */}
      <div className="card p-5 mb-6">
        <h2 className="section-title mb-4">Upload</h2>

        {cv ? (
          /* Loaded state */
          <div className="flex items-start gap-4 p-4 rounded-lg bg-sage-50 border border-sage-200">
            <div className="w-10 h-10 rounded-lg bg-sage-100 flex items-center justify-center shrink-0">
              <FileText className="w-5 h-5 text-sage-600" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium text-text-primary text-sm truncate">{cv.filename}</span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-sage-100 text-sage-700 border border-sage-200">
                  <CheckCircle2 className="w-3 h-3" />
                  Parsed
                </span>
              </div>
              <div className="flex items-center gap-3 mt-1 text-xs text-text-muted">
                <span>{formatFileSize(cv.file_size)}</span>
                <span>·</span>
                <span>{cv.file_type.toUpperCase()}</span>
                <span>·</span>
                <span>{formatDate(cv.created_at)}</span>
              </div>
            </div>
          </div>
        ) : (
          /* Drop zone */
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-sage-400 bg-sage-50'
                : 'border-border hover:border-sage-300 hover:bg-gray-50'
            } ${uploading ? 'opacity-60 pointer-events-none' : ''}`}
          >
            <input {...getInputProps()} />
            <Upload className="w-8 h-8 text-text-muted mx-auto mb-3" />
            <p className="text-sm font-medium text-text-secondary mb-1">
              {isDragActive ? 'Drop your CV here…' : 'Drop your CV here or click to browse'}
            </p>
            <p className="text-xs text-text-muted">PDF, DOCX, DOC, MD, TXT</p>
            {uploading && <p className="text-xs text-sage-600 mt-2 font-medium">Parsing…</p>}
          </div>
        )}

        {/* Actions */}
        {cv && (
          <div className="flex items-center gap-2 mt-4">
            <div {...getRootProps()}>
              <input {...getInputProps()} />
              <button className="btn-secondary flex items-center gap-2">
                <RefreshCw className="w-3.5 h-3.5" />
                Re-upload
              </button>
            </div>
            <button onClick={handleDelete} className="btn-danger flex items-center gap-2">
              <Trash2 className="w-3.5 h-3.5" />
              Clear CV
            </button>
          </div>
        )}
      </div>

      {/* Parsed Content Preview */}
      {cv && (
        <div className="card p-5 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="section-title">Parsed Content Preview</h2>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors"
            >
              {copied ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-sage-500" />
                  <span className="text-sage-600">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </>
              )}
            </button>
          </div>
          <div className="bg-gray-50 rounded-lg border border-border p-4 max-h-80 overflow-y-auto">
            <pre className="text-xs text-text-secondary font-mono whitespace-pre-wrap leading-relaxed">
              {cv.parsed_text}
            </pre>
          </div>
        </div>
      )}

      {/* Privacy note */}
      <div className="flex items-start gap-2 text-xs text-text-muted">
        <ShieldCheck className="w-4 h-4 text-sage-500 shrink-0 mt-0.5" />
        <span>
          Your CV is stored locally in a SQLite database and never sent to any external server —
          unless you configure a cloud LLM provider (Claude / OpenAI / Azure).
        </span>
      </div>
    </div>
  )
}
