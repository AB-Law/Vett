import { useEffect, useMemo, useRef, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  type Edge,
  Handle,
  type Node,
  type NodeProps,
  Position,
} from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from '@dagrejs/dagre'
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Loader2, Network, Send } from 'lucide-react'
import toast from 'react-hot-toast'

import {
  chatMindMap,
  getLatestMindMapJob,
  getMindMapJobStatus,
  getMindMapNodeInfo,
  getInterviewDocuments,
  startMindMapJob,
  type InterviewKnowledgeDocument,
  type MindMapNodeInfoResponse,
  type MindMapNodeSource,
} from '../lib/api'

// ── Constants ──────────────────────────────────────────────────────────────────

type MapStatus = 'idle' | 'generating' | 'ready' | 'failed'

const NODE_WIDTH = 200
const NODE_HEIGHT = 40

// NotebookLM-style dark palette: bg, border, text, selectedBg
const GROUP_PALETTE = [
  { bg: '#2A3A2E', border: '#4A7C59', text: '#D4EDD9', selectedBg: '#3A5A42' }, // root – deep green
  { bg: '#1E2A38', border: '#3D6B9E', text: '#C8DCEF', selectedBg: '#2A3D55' }, // main – blue
  { bg: '#2E2218', border: '#8A5C2A', text: '#F0DEC4', selectedBg: '#44321E' }, // detail – amber
  { bg: '#2A1E38', border: '#6B4A9E', text: '#DEC8EF', selectedBg: '#3D2A55' }, // purple
  { bg: '#1E2E2E', border: '#2E8A7A', text: '#C4EDE9', selectedBg: '#1E4040' }, // teal
  { bg: '#2E1E24', border: '#9E3D5C', text: '#EFC8D4', selectedBg: '#441826' }, // rose
]

// ── Custom node ────────────────────────────────────────────────────────────────

interface MindMapNodeData {
  label: string
  palette: typeof GROUP_PALETTE[number]
  isSelected: boolean
  isRoot: boolean
  hasChildren: boolean
  isExpanded: boolean
  onToggle: (id: string) => void
}

function MindMapNode({ id, data }: NodeProps<MindMapNodeData>) {
  const { label, palette, isSelected, isRoot, hasChildren, isExpanded, onToggle } = data
  return (
    <div
      style={{
        borderRadius: 8,
        border: `1.5px solid ${isSelected ? palette.border : palette.border + 'BB'}`,
        background: isSelected ? palette.selectedBg : palette.bg,
        color: palette.text,
        fontSize: isRoot ? 13 : 12,
        fontWeight: isRoot ? 700 : isSelected ? 600 : 400,
        padding: '7px 10px 7px 14px',
        width: isRoot ? NODE_WIDTH + 20 : NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        boxShadow: isSelected
          ? `0 0 0 3px ${palette.border}55, 0 4px 12px rgba(0,0,0,0.4)`
          : '0 2px 6px rgba(0,0,0,0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 6,
        cursor: 'pointer',
        transition: 'all 0.15s ease',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: palette.border, width: 6, height: 6 }} />
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
      {hasChildren && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onToggle(id) }}
          style={{
            background: palette.border + '33',
            border: `1px solid ${palette.border}88`,
            borderRadius: 4,
            color: palette.text,
            width: 20,
            height: 20,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            cursor: 'pointer',
          }}
        >
          {isExpanded
            ? <ChevronLeft style={{ width: 11, height: 11 }} />
            : <ChevronRight style={{ width: 11, height: 11 }} />}
        </button>
      )}
      <Handle type="source" position={Position.Right} style={{ background: palette.border, width: 6, height: 6 }} />
    </div>
  )
}

const NODE_TYPES = { mindmap: MindMapNode }

// ── Dagre LR layout ────────────────────────────────────────────────────────────

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', ranksep: 100, nodesep: 32, marginx: 60, marginy: 60 })
  nodes.forEach((n) => {
    const w = (n.data as MindMapNodeData).isRoot ? NODE_WIDTH + 20 : NODE_WIDTH
    g.setNode(n.id, { width: w, height: NODE_HEIGHT })
  })
  edges.forEach((e) => g.setEdge(e.source, e.target))
  dagre.layout(g)
  return nodes.map((n) => {
    const pos = g.node(n.id)
    const w = (n.data as MindMapNodeData).isRoot ? NODE_WIDTH + 20 : NODE_WIDTH
    return { ...n, position: { x: pos.x - w / 2, y: pos.y - NODE_HEIGHT / 2 } }
  })
}

// ── Citation rendering ─────────────────────────────────────────────────────────

function CitationChip({ index, onClick }: { index: number; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-semibold bg-accent/20 text-accent hover:bg-accent/30 transition-colors align-middle"
    >
      {index}
    </button>
  )
}

/** Render text with [N] markers as clickable citation chips */
function CitedText({ text, onCitationClick }: { text: string; onCitationClick: (index: number) => void }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <span>
      {parts.map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part)
        if (match) {
          const idx = Number(match[1])
          return <CitationChip key={i} index={idx} onClick={() => onCitationClick(idx)} />
        }
        return <span key={i}>{part}</span>
      })}
    </span>
  )
}

// ── Sources accordion ──────────────────────────────────────────────────────────

function SourcesAccordion({
  sources,
  highlightIndex,
}: {
  sources: MindMapNodeSource[]
  highlightIndex: number | null
}) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null
  return (
    <div className="border border-border rounded-lg overflow-hidden text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-bg-secondary hover:bg-bg-tertiary transition-colors"
      >
        <span className="font-medium text-text-secondary">Sources ({sources.length})</span>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-text-muted" /> : <ChevronDown className="w-3.5 h-3.5 text-text-muted" />}
      </button>
      {open && (
        <div className="divide-y divide-border">
          {sources.map((src) => {
            const pdfUrl = src.doc_id
              ? `/api/interview-documents/${src.doc_id}/file${src.page_number ? `#page=${src.page_number}` : ''}`
              : null
            return (
              <div
                key={src.index}
                className={`px-3 py-2 space-y-1 transition-colors ${highlightIndex === src.index ? 'bg-accent/10' : ''}`}
              >
                <div className="flex items-center gap-1.5">
                  <span className="inline-flex items-center justify-center w-4 h-4 rounded text-[9px] font-bold bg-accent/20 text-accent shrink-0">
                    {src.index}
                  </span>
                  {pdfUrl ? (
                    <a
                      href={pdfUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-accent hover:underline truncate"
                    >
                      {src.filename}{src.page_number ? ` · p.${src.page_number}` : ''}
                    </a>
                  ) : (
                    <span className="font-medium text-text-primary truncate">{src.filename}</span>
                  )}
                </div>
                <p className="text-text-muted leading-relaxed line-clamp-3">{src.snippet}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Chat message ───────────────────────────────────────────────────────────────

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: MindMapNodeSource[]
}

function ChatBubble({ msg }: { msg: ChatMessage }) {
  const [citationHighlight, setCitationHighlight] = useState<number | null>(null)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-accent text-white rounded-2xl rounded-tr-sm px-3 py-2 text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="max-w-[95%] bg-bg-secondary rounded-2xl rounded-tl-sm px-3 py-2 text-sm leading-relaxed text-text-primary space-y-1.5">
        {msg.content.split('\n').map((line, i) => {
          const trimmed = line.trim()
          if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
            return (
              <div key={i} className="flex gap-2">
                <span className="text-accent shrink-0 mt-0.5">•</span>
                <CitedText text={trimmed.replace(/^[•\-*]\s*/, '')} onCitationClick={setCitationHighlight} />
              </div>
            )
          }
          if (!trimmed) return null
          return (
            <p key={i}>
              <CitedText text={line} onCitationClick={setCitationHighlight} />
            </p>
          )
        })}
      </div>
      {msg.sources && msg.sources.length > 0 && (
        <SourcesAccordion sources={msg.sources} highlightIndex={citationHighlight} />
      )}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function StudyMindMap() {
  const [documents, setDocuments] = useState<InterviewKnowledgeDocument[]>([])
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null)
  const [graph, setGraph] = useState<{ nodes: { id: string; label: string; group: string }[]; edges: { source: string; target: string; label: string }[] } | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [status, setStatus] = useState<MapStatus>('idle')
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [nodeInfoCache, setNodeInfoCache] = useState<Record<string, MindMapNodeInfoResponse>>({})

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const chatEndRef = useRef<HTMLDivElement>(null)

  const extractErrorMessage = (errorValue: unknown): string => {
    const typed = errorValue as { response?: { data?: { detail?: string } }; message?: string }
    return typed?.response?.data?.detail || typed?.message || 'Request failed'
  }

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // Polling during generation
  useEffect(() => {
    if (!taskId || status !== 'generating') return
    const interval = setInterval(async () => {
      try {
        const result = await getMindMapJobStatus(taskId)
        if (result.graph?.nodes?.length) setGraph(result.graph)
        if (result.status === 'done' || result.status === 'failed') {
          clearInterval(interval)
          setStatus(result.status === 'done' ? 'ready' : 'failed')
          if (result.error) setError(result.error)
        }
      } catch { /* swallow poll errors */ }
    }, 2500)
    return () => clearInterval(interval)
  }, [taskId, status])

  // Load documents on mount
  useEffect(() => {
    getInterviewDocuments()
      .then(setDocuments)
      .catch(() => setDocuments([]))
  }, [])

  // Auto-load latest cached map when doc selection changes
  useEffect(() => {
    setError('')
    setGraph(null)
    setSelectedNodeId(null)
    setTaskId(null)
    if (!selectedDocId) return
    getLatestMindMapJob({ doc_id: selectedDocId })
      .then((result) => {
        if (result.graph?.nodes?.length) {
          setGraph(result.graph)
          setTaskId(result.task_id)
          setStatus('ready')
        }
      })
      .catch(() => { /* no cached map yet — that's fine */ })
  }, [selectedDocId]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleGenerate = async () => {
    if (!selectedDocId) { toast.error('Select a document first.'); return }
    setIsGenerating(true)
    setError('')
    setGraph(null)
    setSelectedNodeId(null)
    setNodeInfoCache({})
    setChatMessages([])
    try {
      const { task_id } = await startMindMapJob({ doc_id: selectedDocId })
      setTaskId(task_id)
      setStatus('generating')
    } catch (err) {
      const message = extractErrorMessage(err)
      setError(message)
      toast.error(message)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleNodeClick = async (_: unknown, node: Node) => {
    setSelectedNodeId(node.id)
    if (!taskId) return
    setChatOpen(true)
    const nodeLabel = graph?.nodes.find((n) => n.id === node.id)?.label ?? node.id
    const question = `Tell me about "${nodeLabel}" in the context of this document.`
    setChatInput('')
    setChatMessages((prev) => [...prev, { role: 'user', content: question }])
    setIsChatLoading(true)
    try {
      if (!nodeInfoCache[node.id]) {
        getMindMapNodeInfo(taskId, node.id)
          .then((info) => setNodeInfoCache((prev) => ({ ...prev, [node.id]: info })))
          .catch(() => {})
      }
      const response = await chatMindMap(taskId, question)
      setChatMessages((prev) => [...prev, { role: 'assistant', content: response.answer, sources: response.sources }])
    } catch {
      setChatMessages((prev) => [...prev, { role: 'assistant', content: 'Could not load a response. Try again.', sources: [] }])
    } finally {
      setIsChatLoading(false)
    }
  }

  const handleChatSubmit = async () => {
    const message = chatInput.trim()
    if (!message || !taskId || isChatLoading) return
    setChatInput('')
    setChatMessages((prev) => [...prev, { role: 'user', content: message }])
    setIsChatLoading(true)
    try {
      const response = await chatMindMap(taskId, message)
      setChatMessages((prev) => [...prev, { role: 'assistant', content: response.answer, sources: response.sources }])
    } catch {
      setChatMessages((prev) => [...prev, { role: 'assistant', content: 'Could not load a response. Try again.', sources: [] }])
    } finally {
      setIsChatLoading(false)
    }
  }

  // ── Graph layout ─────────────────────────────────────────────────────────────

  // Build child map from edges
  const childMap = useMemo(() => {
    const map = new Map<string, string[]>()
    if (!graph) return map
    graph.nodes.forEach((n) => map.set(n.id, []))
    graph.edges.forEach((e) => map.get(e.source)?.push(e.target))
    return map
  }, [graph])

  // Reset expanded state when a new graph loads
  useEffect(() => {
    if (!graph) return
    setExpandedNodes(new Set())
  }, [graph])

  const handleToggle = (id: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // BFS: only include nodes reachable through expanded parents
  const visibleNodeIds = useMemo(() => {
    if (!graph) return new Set<string>()
    const rootId = graph.nodes.find((n) => n.id === 'root' || n.group === 'root')?.id ?? graph.nodes[0]?.id
    if (!rootId) return new Set<string>()
    const visible = new Set<string>([rootId])
    const queue = [rootId]
    while (queue.length) {
      const id = queue.shift()!
      if (!expandedNodes.has(id)) continue
      for (const child of childMap.get(id) ?? []) {
        if (!visible.has(child)) { visible.add(child); queue.push(child) }
      }
    }
    return visible
  }, [graph, expandedNodes, childMap])

  const groupIndexMap = useMemo(() => {
    if (!graph) return new Map<string, number>()
    const groups = Array.from(new Set(graph.nodes.map((n) => n.group || 'general')))
    return new Map(groups.map((g, i) => [g, i % GROUP_PALETTE.length]))
  }, [graph])

  const rawFlowNodes: Node[] = useMemo(() => {
    if (!graph) return []
    return graph.nodes
      .filter((node) => visibleNodeIds.has(node.id))
      .map((node) => {
        const group = node.group || 'general'
        const paletteIdx = group === 'root' ? 0 : group === 'main' ? 1 : (groupIndexMap.get(group) ?? 2)
        const palette = GROUP_PALETTE[paletteIdx % GROUP_PALETTE.length]
        const isRoot = node.id === 'root' || group === 'root'
        const hasChildren = (childMap.get(node.id)?.length ?? 0) > 0
        return {
          id: node.id,
          type: 'mindmap',
          data: {
            label: node.label,
            palette,
            isSelected: selectedNodeId === node.id,
            isRoot,
            hasChildren,
            isExpanded: expandedNodes.has(node.id),
            onToggle: handleToggle,
          } satisfies MindMapNodeData,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          position: { x: 0, y: 0 },
        }
      })
  }, [graph, visibleNodeIds, selectedNodeId, groupIndexMap, childMap, expandedNodes]) // eslint-disable-line react-hooks/exhaustive-deps

  const flowEdges: Edge[] = useMemo(() => {
    if (!graph) return []
    return graph.edges
      .filter((e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target))
      .map((edge, index) => ({
        id: `${edge.source}-${edge.target}-${index}`,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
        animated: false,
        style: { stroke: '#4A6858', strokeWidth: 1.5 },
      }))
  }, [graph, visibleNodeIds])

  const flowNodes = useMemo(
    () => (rawFlowNodes.length ? applyDagreLayout(rawFlowNodes, flowEdges) : rawFlowNodes),
    [rawFlowNodes, flowEdges],
  )

  const hasMindMap = Boolean(graph?.nodes.length)

  return (
    <div className="study-page">
      <section className="study-header">
        <h1>Concept Mind Map</h1>
        <p>Generate a visual knowledge graph, then chat with your documents.</p>
      </section>

      {/* Controls */}
      <section className="card p-4 space-y-3">
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label className="label" htmlFor="mindmap-doc-id">Document</label>
            <select
              id="mindmap-doc-id"
              className="input"
              value={selectedDocId ?? ''}
              onChange={(e) => setSelectedDocId(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">Select a document…</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>{doc.source_filename} (#{doc.id})</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <button
            type="button"
            className="btn-primary inline-flex items-center gap-2"
            onClick={() => { void handleGenerate() }}
            disabled={!selectedDocId || isGenerating || status === 'generating'}
          >
            {isGenerating || status === 'generating' ? <Loader2 className="spinner" /> : <Network className="w-4 h-4" />}
            {hasMindMap ? 'Regenerate' : 'Generate Mind Map'}
          </button>
          {hasMindMap && (
            <span className="text-xs text-text-muted self-center">
              {graph!.nodes.length} nodes · {graph!.edges.length} edges
            </span>
          )}
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      {status === 'generating' && (
        <section className="card p-4 flex items-center gap-2 text-sm text-text-secondary">
          <Loader2 className="spinner" />
          Extracting concepts… nodes will appear as batches complete.
        </section>
      )}

      {!hasMindMap && status !== 'generating' && (
        <section className="empty-state">
          <h2>{selectedDocId ? 'No cached map yet.' : 'Select a document to begin.'}</h2>
          <p>Pick a document and click Generate. Click any node to chat about it.</p>
        </section>
      )}

      {/* Map + chat layout */}
      {hasMindMap && (
        <section className="relative" style={{ height: 'calc(100vh - 240px)', minHeight: 600 }}>
          {/* Full-width mind map */}
          <article className="card p-0 overflow-hidden w-full h-full">
            <ReactFlow
              nodes={flowNodes}
              edges={flowEdges}
              nodeTypes={NODE_TYPES}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              onNodeClick={(event, node) => { void handleNodeClick(event, node) }}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#334" gap={24} />
              <Controls />
            </ReactFlow>
          </article>

          {/* Floating chat toggle button */}
          {taskId && !chatOpen && (
            <button
              type="button"
              onClick={() => setChatOpen(true)}
              className="absolute bottom-4 right-4 btn-primary flex items-center gap-2 shadow-lg z-10"
            >
              <Send className="w-4 h-4" />
              Chat
              {chatMessages.length > 0 && (
                <span className="bg-white/20 rounded-full text-xs px-1.5">{chatMessages.length}</span>
              )}
            </button>
          )}

          {/* Slide-in chat panel */}
          {chatOpen && (
            <div
              className="absolute top-0 right-0 h-full card rounded-l-xl rounded-r-none flex flex-col shadow-2xl z-10"
              style={{ width: 360 }}
            >
              <div className="px-4 py-3 border-b border-border shrink-0 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-text-primary">Chat</p>
                  <p className="text-xs text-text-muted">Ask anything about the document.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setChatOpen(false)}
                  className="text-text-muted hover:text-text-primary transition-colors text-lg leading-none"
                >
                  ×
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-0">
                {chatMessages.length === 0 && (
                  <p className="text-xs text-text-muted text-center pt-8">
                    Click a node in the graph or ask a question below.
                  </p>
                )}
                {chatMessages.map((msg, i) => <ChatBubble key={i} msg={msg} />)}
                {isChatLoading && (
                  <div className="flex items-center gap-2 text-sm text-text-secondary">
                    <Loader2 className="spinner" /> Generating…
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="px-4 py-3 border-t border-border shrink-0">
                <div className="flex gap-2">
                  <input
                    className="input flex-1 text-sm"
                    placeholder="Ask about the document…"
                    value={chatInput}
                    disabled={isChatLoading}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleChatSubmit() } }}
                  />
                  <button
                    type="button"
                    className="btn-primary shrink-0 px-3"
                    onClick={() => { void handleChatSubmit() }}
                    disabled={!chatInput.trim() || isChatLoading}
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  )
}
