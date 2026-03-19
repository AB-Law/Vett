import { useEffect, useMemo, useState } from 'react'
import ReactFlow, { Background, Controls, type Edge, type Node, Position } from 'reactflow'
import 'reactflow/dist/style.css'
import { Loader2, Network, RefreshCcw } from 'lucide-react'
import toast from 'react-hot-toast'

import {
  generateStudyMindMap,
  getInterviewDocuments,
  getJobInterviewDocuments,
  getStudyMindMap,
  type InterviewKnowledgeDocument,
  type StudyMindMapResponse,
} from '../lib/api'

type MapStatus = 'idle' | 'loading' | 'ready'

export default function StudyMindMap() {
  const [jobIdInput, setJobIdInput] = useState('')
  const [jobId, setJobId] = useState<number | null>(null)
  const [documents, setDocuments] = useState<InterviewKnowledgeDocument[]>([])
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null)
  const [mindMap, setMindMap] = useState<StudyMindMapResponse | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [status, setStatus] = useState<MapStatus>('idle')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isCurrentSelectionCached, setIsCurrentSelectionCached] = useState(false)
  const [error, setError] = useState('')

  const extractErrorMessage = (errorValue: unknown): string => {
    const typed = errorValue as { response?: { data?: { detail?: string } }; message?: string }
    return typed?.response?.data?.detail || typed?.message || 'Request failed'
  }

  const loadSelection = async (nextJobId: number, nextDocId: number | null) => {
    setStatus('loading')
    setError('')
    setMindMap(null)
    setSelectedNodeId(null)
    try {
      const data = await getStudyMindMap(nextJobId, nextDocId)
      setMindMap(data)
      setIsCurrentSelectionCached(true)
      setStatus('ready')
    } catch (err: unknown) {
      const message = extractErrorMessage(err)
      if (message.toLowerCase().includes('not found')) {
        setIsCurrentSelectionCached(false)
        setStatus('idle')
        return
      }
      setError(message)
      setStatus('idle')
    }
  }

  const loadDocuments = async (nextJobId: number) => {
    try {
      const [globalDocs, jobDocs] = await Promise.all([
        getInterviewDocuments(),
        getJobInterviewDocuments(nextJobId),
      ])
      const seen = new Set<number>()
      const merged = [...jobDocs, ...globalDocs].filter((doc) => {
        if (seen.has(doc.id)) return false
        seen.add(doc.id)
        return true
      })
      setDocuments(merged)
      if (selectedDocId && merged.every((doc) => doc.id !== selectedDocId)) {
        setSelectedDocId(null)
      }
    } catch {
      setDocuments([])
    }
  }

  useEffect(() => {
    if (!jobId) return
    void loadDocuments(jobId)
    void loadSelection(jobId, selectedDocId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, selectedDocId])

  const applyJobId = () => {
    const parsed = Number(jobIdInput)
    if (!Number.isInteger(parsed) || parsed <= 0) {
      toast.error('Enter a valid job id.')
      return
    }
    setJobId(parsed)
  }

  const handleGenerate = async () => {
    if (!jobId) {
      toast.error('Enter a job id first.')
      return
    }
    setIsGenerating(true)
    setError('')
    try {
      const payload = await generateStudyMindMap({ job_id: jobId, doc_id: selectedDocId ?? null })
      setMindMap(payload)
      setStatus('ready')
      setIsCurrentSelectionCached(true)
      setSelectedNodeId(payload.graph.nodes[0]?.id ?? null)
      toast.success(payload.cached ? 'Loaded cached mind map.' : 'Mind map generated.')
    } catch (err: unknown) {
      const message = extractErrorMessage(err)
      setError(message)
      toast.error(message)
    } finally {
      setIsGenerating(false)
    }
  }

  const flowNodes: Node[] = useMemo(() => {
    if (!mindMap) return []
    const groups = Array.from(new Set(mindMap.graph.nodes.map((node) => node.group || 'general')))
    const groupIndex = new Map(groups.map((group, index) => [group, index]))
    return mindMap.graph.nodes.map((node, index) => {
      const group = node.group || 'general'
      const row = Math.floor(index / 4)
      const col = index % 4
      return {
        id: node.id,
        data: { label: node.label },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        position: {
          x: col * 240 + (groupIndex.get(group) ?? 0) * 20,
          y: row * 130 + (groupIndex.get(group) ?? 0) * 20,
        },
        style: {
          borderRadius: 10,
          border: '1px solid #A1B39A',
          background: '#F4F9F2',
          color: '#29342F',
          fontSize: 12,
          padding: 10,
          width: 200,
        },
      }
    })
  }, [mindMap])

  const flowEdges: Edge[] = useMemo(() => {
    if (!mindMap) return []
    return mindMap.graph.edges.map((edge, index) => ({
      id: `${edge.source}-${edge.target}-${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#7F8F78' },
      labelStyle: { fontSize: 11, fill: '#4C5B51' },
    }))
  }, [mindMap])

  const selectedNodeSource = selectedNodeId && mindMap ? mindMap.node_sources[selectedNodeId] : ''
  const selectedNodeLabel = selectedNodeId && mindMap ? mindMap.graph.nodes.find((node) => node.id === selectedNodeId)?.label : ''
  const canRegenerate = Boolean(mindMap) && !isGenerating && !isCurrentSelectionCached

  return (
    <div className="study-page">
      <section className="study-header">
        <h1>Concept Mind Map</h1>
        <p>Generate a visual graph from interview documents for the selected job context.</p>
      </section>

      <section className="card p-4 space-y-3">
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <label className="label" htmlFor="mindmap-job-id">
              Job id
            </label>
            <input
              id="mindmap-job-id"
              className="input"
              value={jobIdInput}
              onChange={(event) => setJobIdInput(event.target.value)}
              placeholder="e.g. 42"
            />
          </div>
          <div>
            <label className="label" htmlFor="mindmap-doc-id">
              Document scope
            </label>
            <select
              id="mindmap-doc-id"
              className="input"
              value={selectedDocId ?? ''}
              onChange={(event) => setSelectedDocId(event.target.value ? Number(event.target.value) : null)}
              disabled={!jobId}
            >
              <option value="">All job documents (top-k)</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.source_filename} (#{doc.id})
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end gap-2">
            <button type="button" className="btn-secondary w-full" onClick={applyJobId}>
              Apply Job
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn-primary inline-flex items-center gap-2" onClick={() => void handleGenerate()} disabled={isGenerating}>
            {isGenerating ? <Loader2 className="spinner" /> : <Network className="w-4 h-4" />}
            {mindMap ? 'Refresh Mind Map' : 'Generate Mind Map'}
          </button>
          <button
            type="button"
            className="btn-secondary inline-flex items-center gap-2"
            onClick={() => void handleGenerate()}
            disabled={!canRegenerate}
            title={canRegenerate ? 'Document content changed. Regenerate available.' : 'Regeneration is available after content changes.'}
          >
            <RefreshCcw className="w-4 h-4" />
            Regenerate
          </button>
          {mindMap ? (
            <span className="text-xs text-text-muted self-center">
              {mindMap.graph.nodes.length} nodes, {mindMap.graph.edges.length} edges
            </span>
          ) : null}
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      {status === 'loading' ? (
        <section className="card p-4 flex items-center gap-2 text-sm text-text-secondary">
          <Loader2 className="spinner" />
          Loading cached mind map...
        </section>
      ) : null}

      {!mindMap && status !== 'loading' ? (
        <section className="empty-state">
          <h2>No cached map for this document state.</h2>
          <p>Generate one now. If you update or re-upload documents, regenerate becomes available for the new content.</p>
        </section>
      ) : null}

      {mindMap ? (
        <section className="grid gap-4 lg:grid-cols-[1fr_300px]">
          <article className="card p-3" style={{ height: 560 }}>
            <ReactFlow
              nodes={flowNodes}
              edges={flowEdges}
              fitView
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              proOptions={{ hideAttribution: true }}
            >
              <Background />
              <Controls />
            </ReactFlow>
          </article>
          <aside className="card p-4 space-y-3">
            <h2 className="section-title">Source Chunk</h2>
            {selectedNodeLabel ? <p className="text-sm font-medium text-text-primary">{selectedNodeLabel}</p> : null}
            <p className="text-xs text-text-muted">
              {selectedNodeSource || 'Click any node in the graph to inspect the supporting source chunk.'}
            </p>
          </aside>
        </section>
      ) : null}
    </div>
  )
}
