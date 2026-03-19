import { useEffect, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { BookOpen, CheckSquare, FileText, Layers, Loader2, Pencil, Plus, Trash2, Upload, X } from 'lucide-react'
import toast from 'react-hot-toast'
import {
  deleteStudyCardSet,
  generateStudyFlashcardsAsync,
  getStudyFlashcardsJob,
  getInterviewDocuments,
  getStudyCardSet,
  listStudyCardSets,
  renameStudyCardSet,
  reviewStudyCard,
  uploadInterviewDocument,
  type FlashcardSetResponse,
  type InterviewKnowledgeDocument,
  type StudyCard,
  type StudyCardSetSummary,
} from '../lib/api'

const DEFAULT_NUM_CARDS = 10
const MAX_TOTAL_CARDS = 5000

export default function StudyDecks() {
  const [documents, setDocuments] = useState<InterviewKnowledgeDocument[]>([])
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<number[]>([])
  const [recentSets, setRecentSets] = useState<StudyCardSetSummary[]>([])
  const [loadingSets, setLoadingSets] = useState(true)
  const [cards, setCards] = useState<StudyCard[]>([])
  const [cardSetId, setCardSetId] = useState<number | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isFlipped, setIsFlipped] = useState(false)
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [uploadingDoc, setUploadingDoc] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatingJobId, setGeneratingJobId] = useState<string | null>(null)
  const [openingDeckId, setOpeningDeckId] = useState<number | null>(null)
  const [isReviewing, setIsReviewing] = useState(false)
  const [deckName, setDeckName] = useState('')
  const [topic, setTopic] = useState('')
  const [numCards, setNumCards] = useState(DEFAULT_NUM_CARDS)
  const [generatePerSection, setGeneratePerSection] = useState(false)
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [error, setError] = useState('')
  const [renameDeckId, setRenameDeckId] = useState<number | null>(null)
  const [renameDeckValue, setRenameDeckValue] = useState('')
  const [deckPendingDelete, setDeckPendingDelete] = useState<StudyCardSetSummary | null>(null)
  const [parentDeckModal, setParentDeckModal] = useState<StudyCardSetSummary | null>(null)
  const [loadingAllCards, setLoadingAllCards] = useState(false)

  const completedCount = Math.min(currentIndex, cards.length)
  const remainingCount = cards.length - completedCount
  const progressPercent = cards.length ? Math.round((completedCount / cards.length) * 100) : 0
  const currentCard = cards[currentIndex] ?? null

  const selectedDocsLabel = useMemo(() => {
    if (selectedDocumentIds.length === 0) return 'No documents selected'
    if (selectedDocumentIds.length === 1) return '1 document selected'
    return `${selectedDocumentIds.length} documents selected`
  }, [selectedDocumentIds.length])

  const deckIds = useMemo(() => new Set(recentSets.map((deck) => deck.id)), [recentSets])

  const childDecksByParentId = useMemo(() => {
    const grouped = new Map<number, StudyCardSetSummary[]>()
    for (const deck of recentSets) {
      if (!deck.parent_card_set_id || !deckIds.has(deck.parent_card_set_id)) continue
      const siblings = grouped.get(deck.parent_card_set_id) ?? []
      siblings.push(deck)
      grouped.set(deck.parent_card_set_id, siblings)
    }
    return grouped
  }, [recentSets, deckIds])

  const topLevelDecks = useMemo(
    () => recentSets.filter((deck) => !deck.parent_card_set_id || !deckIds.has(deck.parent_card_set_id)),
    [recentSets, deckIds],
  )

  const extractErrorMessage = (errorValue: unknown): string => {
    const typed = errorValue as { response?: { data?: { detail?: string } }; message?: string }
    return typed?.response?.data?.detail || typed?.message || 'Request failed'
  }

  const statusClass = (status: string): string => {
    const normalized = status.toLowerCase()
    if (normalized.includes('embed')) return 'bg-sage-100 text-sage-700 border border-sage-200'
    if (normalized.includes('processing')) return 'bg-amber-100 text-amber-700 border border-amber-200'
    if (normalized.includes('failed')) return 'bg-red-100 text-red-700 border border-red-200'
    return 'bg-gray-100 text-text-muted border border-border'
  }

  const loadDocuments = async () => {
    setLoadingDocs(true)
    try {
      const docs = await getInterviewDocuments()
      setDocuments(docs)
      if (docs.length > 0 && selectedDocumentIds.length === 0) {
        setSelectedDocumentIds([docs[0].id])
      }
    } catch (err: unknown) {
      console.error(err)
      setError('Could not load uploaded documents.')
    } finally {
      setLoadingDocs(false)
    }
  }

  const loadRecentSets = async () => {
    setLoadingSets(true)
    try {
      const sets = await listStudyCardSets(50)
      setRecentSets(sets)
    } catch (err: unknown) {
      console.error(err)
    } finally {
      setLoadingSets(false)
    }
  }

  useEffect(() => {
    void loadDocuments()
    void loadRecentSets()
  }, [])

  const handleUpload = async (file: File | null) => {
    if (!file) return
    setUploadingDoc(true)
    setError('')
    try {
      const uploaded = await uploadInterviewDocument(file)
      setDocuments((prev) => [uploaded, ...prev])
      setSelectedDocumentIds((prev) => (prev.includes(uploaded.id) ? prev : [uploaded.id, ...prev]))
      toast.success('Document uploaded to Study.')
    } catch (err: unknown) {
      console.error(err)
      setError(extractErrorMessage(err))
      toast.error(extractErrorMessage(err))
    } finally {
      setUploadingDoc(false)
    }
  }

  const { getRootProps: getUploadRootProps, getInputProps: getUploadInputProps, isDragActive: isUploadDragActive } = useDropzone({
    onDrop: (files) => void handleUpload(files[0] || null),
    multiple: false,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
    disabled: uploadingDoc,
  })

  const applyCreateResponse = (response: FlashcardSetResponse) => {
    setCards(response.cards)
    setCardSetId(response.card_set_id)
    setCurrentIndex(0)
    setIsFlipped(false)
    if (response.parent_card_set_id && response.card_sets.length > 1) {
      toast.success(`Generated ${response.card_sets.length} child decks under parent deck #${response.parent_card_set_id}.`)
    } else if (response.card_sets.length > 1) {
      toast.success(`Generated ${response.card_sets.length} section decks.`)
    } else if (response.cards.length > 0) {
      toast.success(`Generated ${response.cards.length} flashcards.`)
    } else {
      toast.error('No flashcards were generated.')
    }
    if (response.generation_diagnostics?.fallback_used) {
      const diagnostics = response.generation_diagnostics
      toast(
        `Generation used ${diagnostics.fallback_cards_used} fallback cards (LLM parsed ${diagnostics.llm_cards_parsed}/${diagnostics.requested_cards}).`,
        { icon: 'ℹ️' },
      )
    }
  }

  const createCards = async () => {
    if (!deckName.trim()) {
      toast.error('Please give your deck a name.')
      return
    }
    if (selectedDocumentIds.length === 0) {
      toast.error('Select at least one document first.')
      return
    }
    setIsGenerating(true)
    setError('')
    setIsFlipped(false)
    try {
      const started = await generateStudyFlashcardsAsync({
        name: deckName || undefined,
        document_ids: selectedDocumentIds,
        topic: topic || undefined,
        num_cards: numCards,
        generate_per_section: generatePerSection,
      })
      setGeneratingJobId(started.job_id)
      setIsCreateModalOpen(false)
      toast.success('Deck generation started. You can keep browsing while it runs.')
    } catch (err: unknown) {
      console.error(err)
      setError(extractErrorMessage(err))
      toast.error(extractErrorMessage(err))
    } finally {
      setIsGenerating(false)
    }
  }

  useEffect(() => {
    if (!generatingJobId) return
    let active = true
    let timer: ReturnType<typeof setTimeout> | null = null
    const pollJob = async () => {
      try {
        const job = await getStudyFlashcardsJob(generatingJobId)
        if (!active) return
        if (job.status === 'completed' && job.result) {
          applyCreateResponse(job.result)
          setGeneratingJobId(null)
          void loadRecentSets()
          return
        }
        if (job.status === 'failed') {
          const message = job.error || 'Flashcard generation failed.'
          setError(message)
          toast.error(message)
          setGeneratingJobId(null)
          return
        }
      } catch (err: unknown) {
        if (!active) return
        const message = extractErrorMessage(err)
        setError(message)
        toast.error(message)
        setGeneratingJobId(null)
        return
      }
      timer = setTimeout(() => {
        void pollJob()
      }, 2000)
    }
    void pollJob()
    return () => {
      active = false
      if (timer) clearTimeout(timer)
    }
  }, [generatingJobId])

  const handleReview = async (rating: 'easy' | 'hard') => {
    if (!currentCard || isReviewing || !isFlipped) return
    setIsReviewing(true)
    try {
      const updatedCard = await reviewStudyCard(currentCard.id, { rating })
      setCards((prev) => prev.map((card) => (card.id === currentCard.id ? updatedCard : card)))
      setCurrentIndex((prev) => prev + 1)
      setIsFlipped(false)
      toast.success('Review recorded.')
    } catch (err: unknown) {
      console.error(err)
      toast.error(extractErrorMessage(err))
    } finally {
      setIsReviewing(false)
    }
  }

  const openRecentSet = async (cardSetIdToOpen: number) => {
    setOpeningDeckId(cardSetIdToOpen)
    setError('')
    try {
      const response = await getStudyCardSet(cardSetIdToOpen)
      setCards(response.cards)
      setCardSetId(response.card_set_id)
      setCurrentIndex(0)
      setIsFlipped(false)
      setIsReviewing(false)
      toast.success(`Loaded ${response.cards.length} cards from "${response.name}".`)
    } catch (err: unknown) {
      console.error(err)
      setError(extractErrorMessage(err))
      toast.error(extractErrorMessage(err))
    } finally {
      setOpeningDeckId(null)
    }
  }

  const openAllChildCards = async (parentDeck: StudyCardSetSummary) => {
    const childDecks = childDecksByParentId.get(parentDeck.id) ?? []
    if (childDecks.length === 0) return
    setLoadingAllCards(true)
    setError('')
    try {
      const responses = await Promise.all(childDecks.map((child) => getStudyCardSet(child.id)))
      const allCards = responses.flatMap((r) => r.cards)
      // Fisher-Yates shuffle
      for (let i = allCards.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[allCards[i], allCards[j]] = [allCards[j], allCards[i]]
      }
      setCards(allCards)
      setCardSetId(parentDeck.id)
      setCurrentIndex(0)
      setIsFlipped(false)
      setIsReviewing(false)
      setParentDeckModal(null)
      toast.success(`Studying ${allCards.length} cards from "${parentDeck.name}" (shuffled).`)
    } catch (err: unknown) {
      console.error(err)
      toast.error(extractErrorMessage(err))
    } finally {
      setLoadingAllCards(false)
    }
  }

  const toggleDocumentSelection = (documentId: number) => {
    setSelectedDocumentIds((prev) =>
      prev.includes(documentId) ? prev.filter((id) => id !== documentId) : [...prev, documentId],
    )
  }

  const startRenameDeck = (deck: StudyCardSetSummary) => {
    setRenameDeckId(deck.id)
    setRenameDeckValue(deck.name || '')
  }

  const submitRenameDeck = async (deckId: number) => {
    const nextName = renameDeckValue.trim()
    if (!nextName) {
      toast.error('Deck name cannot be empty.')
      return
    }
    try {
      const updated = await renameStudyCardSet(deckId, { name: nextName })
      setRecentSets((prev) => prev.map((deck) => (deck.id === deckId ? updated : deck)))
      setRenameDeckId(null)
      setRenameDeckValue('')
      toast.success('Deck renamed.')
    } catch (err: unknown) {
      console.error(err)
      toast.error(extractErrorMessage(err))
    }
  }

  const removeDeck = async (deck: StudyCardSetSummary) => {
    try {
      await deleteStudyCardSet(deck.id)
      setRecentSets((prev) => prev.filter((item) => item.id !== deck.id))
      if (cardSetId === deck.id) {
        setCards([])
        setCardSetId(null)
        setCurrentIndex(0)
      }
      setDeckPendingDelete(null)
      toast.success('Deck deleted.')
    } catch (err: unknown) {
      console.error(err)
      toast.error(extractErrorMessage(err))
    }
  }

  if (loadingDocs) {
    return (
      <div className="study-page">
        <div className="loading-wrap">
          <Loader2 className="spinner" />
          Loading study documents...
        </div>
      </div>
    )
  }

  return (
    <div className="study-page">
      <section className="study-header">
        <h1>Flashcard Decks</h1>
        <p>Create named decks from one or more interview documents, then review with spaced repetition.</p>
      </section>

      <section className="card p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-text-primary">
            <Upload className="w-4 h-4" />
            Upload source documents
          </div>
          <button type="button" className="btn-primary h-9 inline-flex items-center gap-2" onClick={() => setIsCreateModalOpen(true)}>
            <Plus className="w-4 h-4" />
            Create Deck
          </button>
        </div>
        <div
          {...getUploadRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isUploadDragActive ? 'border-sage-400 bg-sage-50' : 'border-border hover:border-sage-300 hover:bg-gray-50'
          } ${uploadingDoc ? 'opacity-60 pointer-events-none' : ''}`}
        >
          <input {...getUploadInputProps()} />
          <FileText className="w-7 h-7 text-text-muted mx-auto mb-2" />
          <p className="text-sm font-medium text-text-secondary mb-1">
            {isUploadDragActive ? 'Drop interview docs here…' : 'Drop or upload PDF / DOCX / DOC / MD / TXT'}
          </p>
          <p className="text-xs text-text-muted">These are used as references when generating deck cards.</p>
          {uploadingDoc && <p className="text-xs text-sage-600 mt-2 font-medium">Uploading…</p>}
        </div>
      </section>

      {generatingJobId ? (
        <section className="card p-4 flex items-center gap-2 text-sm text-text-secondary">
          <Loader2 className="spinner" />
          Generating your deck in the background...
        </section>
      ) : null}

      <section className="card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="section-title">Deck Library</h2>
          {loadingSets ? <span className="text-xs text-text-muted">Loading…</span> : null}
        </div>
        {recentSets.length === 0 ? (
          <p className="text-xs text-text-muted">No saved decks yet.</p>
        ) : (
          <div className="space-y-2">
            {topLevelDecks.map((deck) => {
              const childDecks = childDecksByParentId.get(deck.id) ?? []
              const isParentContainer = deck.card_count === 0 && childDecks.length > 0
              const canOpenDeck = deck.card_count > 0
              return (
                <div
                  key={deck.id}
                  className={`border rounded p-3 ${
                    cardSetId === deck.id ? 'border-sage-300 bg-sage-50' : 'border-border'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <button
                      type="button"
                      className="text-left flex-1"
                      onClick={() => isParentContainer ? setParentDeckModal(deck) : (canOpenDeck ? void openRecentSet(deck.id) : undefined)}
                      disabled={openingDeckId === deck.id || (!canOpenDeck && !isParentContainer)}
                    >
                      <p className="text-sm font-semibold text-text-primary">{deck.name}</p>
                      <p className="text-xs text-text-muted mt-1">
                        {isParentContainer
                          ? `${childDecks.length} child decks · Deck group · ${deck.created_at ? new Date(deck.created_at).toLocaleString() : 'Unknown date'}`
                          : `${deck.document_count} docs · ${deck.card_count} cards · ${deck.topic || 'General study'} · ${
                              deck.created_at ? new Date(deck.created_at).toLocaleString() : 'Unknown date'
                            }`}
                      </p>
                      {openingDeckId === deck.id ? (
                        <span className="mt-2 inline-flex items-center gap-1 text-xs text-sage-700">
                          <Loader2 className="spinner" />
                          Loading deck...
                        </span>
                      ) : null}
                    </button>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="text-xs text-text-secondary hover:text-sage-700"
                        onClick={() => startRenameDeck(deck)}
                        aria-label="Rename deck"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                      <button
                        type="button"
                        className="text-xs text-red-600 hover:text-red-700"
                        onClick={() => setDeckPendingDelete(deck)}
                        aria-label="Delete deck"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  {renameDeckId === deck.id ? (
                    <div className="mt-2 flex items-center gap-2">
                      <input
                        value={renameDeckValue}
                        onChange={(event) => setRenameDeckValue(event.target.value)}
                        className="input h-9"
                        placeholder="Deck name"
                      />
                      <button type="button" className="btn-primary h-9 px-3" onClick={() => void submitRenameDeck(deck.id)}>
                        Save
                      </button>
                      <button type="button" className="btn-secondary h-9 px-3" onClick={() => setRenameDeckId(null)}>
                        Cancel
                      </button>
                    </div>
                  ) : null}
                  {childDecks.length > 0 ? (
                    <div className="mt-2 ml-4 space-y-2 border-l border-border pl-3">
                      {childDecks.map((childDeck) => (
                        <div
                          key={childDeck.id}
                          className={`border rounded p-2 ${
                            cardSetId === childDeck.id ? 'border-sage-300 bg-sage-50' : 'border-border'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <button
                              type="button"
                              className="text-left flex-1"
                              onClick={() => void openRecentSet(childDeck.id)}
                              disabled={openingDeckId === childDeck.id}
                            >
                              <p className="text-sm font-semibold text-text-primary">{childDeck.name}</p>
                              <p className="text-xs text-text-muted mt-1">
                                {childDeck.document_count} docs · {childDeck.card_count} cards · {childDeck.topic || 'General study'} ·{' '}
                                {childDeck.created_at ? new Date(childDeck.created_at).toLocaleString() : 'Unknown date'}
                              </p>
                              {openingDeckId === childDeck.id ? (
                                <span className="mt-2 inline-flex items-center gap-1 text-xs text-sage-700">
                                  <Loader2 className="spinner" />
                                  Loading deck...
                                </span>
                              ) : null}
                            </button>
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                className="text-xs text-text-secondary hover:text-sage-700"
                                onClick={() => startRenameDeck(childDeck)}
                                aria-label="Rename deck"
                              >
                                <Pencil className="w-4 h-4" />
                              </button>
                              <button
                                type="button"
                                className="text-xs text-red-600 hover:text-red-700"
                                onClick={() => setDeckPendingDelete(childDeck)}
                                aria-label="Delete deck"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                          {renameDeckId === childDeck.id ? (
                            <div className="mt-2 flex items-center gap-2">
                              <input
                                value={renameDeckValue}
                                onChange={(event) => setRenameDeckValue(event.target.value)}
                                className="input h-9"
                                placeholder="Deck name"
                              />
                              <button
                                type="button"
                                className="btn-primary h-9 px-3"
                                onClick={() => void submitRenameDeck(childDeck.id)}
                              >
                                Save
                              </button>
                              <button type="button" className="btn-secondary h-9 px-3" onClick={() => setRenameDeckId(null)}>
                                Cancel
                              </button>
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        )}
      </section>

      {cardSetId ? <p className="study-meta">Current deck #{cardSetId}</p> : null}

      <section className="study-progress">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
        </div>
        <p>{remainingCount} remaining</p>
      </section>

      {error ? <p className="error">{error}</p> : null}

      {cards.length === 0 ? (
        <section className="empty-state">
          <h2>Ready to create or open a deck?</h2>
          <p>Use Create Deck to pick multiple docs, set a topic, and generate cards.</p>
          <button onClick={() => setIsCreateModalOpen(true)} className="btn-primary">
            Open Create Deck
          </button>
        </section>
      ) : currentIndex >= cards.length ? (
        <section className="empty-state">
          <h2>Session complete</h2>
          <p>You reviewed all {cards.length} cards.</p>
          <button onClick={() => setIsCreateModalOpen(true)} className="btn-primary">
            Generate another deck
          </button>
        </section>
      ) : (
        <section className="study-card-wrap">
          <div className="study-session-header">
            <h3 className="study-session-title">Active Study Session</h3>
            <p className="study-session-help">Flip the card, then rate your confidence to schedule the next review.</p>
            <div className="study-session-meta">
              <span>
                Card {currentIndex + 1} of {cards.length}
              </span>
              <span>{remainingCount} remaining</span>
            </div>
          </div>
          <button
            type="button"
            className="card-flip w-full bg-transparent border-0 p-0 text-left"
            onClick={() => setIsFlipped((prev) => !prev)}
            aria-label="Flip study card"
          >
            <div className={`card-flip-inner ${isFlipped ? 'flipped' : ''}`}>
              <div className="card-face card-front">Q{currentIndex + 1}: {currentCard?.front}</div>
              <div className="card-face card-back">A: {currentCard?.back}</div>
            </div>
          </button>
          <div className="study-buttons">
            <button className="btn-hard" onClick={() => void handleReview('hard')} disabled={isReviewing || !isFlipped}>
              Hard
            </button>
            <button className="btn-easy" onClick={() => void handleReview('easy')} disabled={isReviewing || !isFlipped}>
              Easy
            </button>
          </div>
          <p className="study-tip">Tip: click the card to flip and reveal the answer.</p>
        </section>
      )}

      <footer className="study-footer">
        <BookOpen className="small-icon" />
        <p>
          Current card: {Math.min(currentIndex + 1, cards.length)}/{cards.length}
        </p>
      </footer>

      {isCreateModalOpen ? (
        <div className="study-modal-overlay" role="presentation" onClick={() => setIsCreateModalOpen(false)}>
          <div className="study-modal card" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h3 className="section-title">Create Deck</h3>
              <button type="button" className="text-text-muted hover:text-text-primary" onClick={() => setIsCreateModalOpen(false)}>
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3 mt-3">
              <label className="label" htmlFor="deckName">
                Deck name <span className="text-red-500">*</span>
              </label>
              <input
                id="deckName"
                value={deckName}
                onChange={(event) => setDeckName(event.target.value)}
                className="input"
                placeholder="e.g. System Design Q3"
                required
              />

              <label className="label">Documents</label>
              <p className="text-xs text-text-muted -mt-1">{selectedDocsLabel}</p>
              <div className="space-y-2 max-h-48 overflow-y-auto border border-border rounded p-2">
                {documents.length === 0 ? (
                  <p className="text-xs text-text-muted">Upload documents first.</p>
                ) : (
                  documents.map((doc) => (
                    <label key={doc.id} className="flex items-center justify-between gap-2 text-sm border border-border rounded p-2">
                      <span className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={selectedDocumentIds.includes(doc.id)}
                          onChange={() => toggleDocumentSelection(doc.id)}
                        />
                        <span className="text-text-primary">{doc.source_filename}</span>
                      </span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] ${statusClass(doc.status)}`}>
                        {doc.status}
                      </span>
                    </label>
                  ))
                )}
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <label className="label" htmlFor="topic">
                    Topic (optional)
                  </label>
                  <input
                    id="topic"
                    value={topic}
                    onChange={(event) => setTopic(event.target.value)}
                    className="input"
                    placeholder="Distributed systems"
                  />
                </div>
                {!generatePerSection ? (
                  <div>
                    <label className="label" htmlFor="numCards">
                      Card count
                    </label>
                    <input
                      id="numCards"
                      type="number"
                      min={1}
                      max={MAX_TOTAL_CARDS}
                      value={numCards}
                      onChange={(event) =>
                        setNumCards(Math.max(1, Math.min(MAX_TOTAL_CARDS, Number(event.target.value || DEFAULT_NUM_CARDS))))
                      }
                      className="input"
                    />
                    <p className="text-xs text-text-muted mt-1">Large targets are split into child decks of up to 50 cards each.</p>
                  </div>
                ) : (
                  <div>
                    <p className="label">Card count</p>
                    <p className="text-sm text-text-muted">Up to 50 cards per detected section — total depends on document structure.</p>
                  </div>
                )}
              </div>

              <label className="flex items-center gap-2 text-sm text-text-secondary">
                <input
                  type="checkbox"
                  checked={generatePerSection}
                  onChange={(event) => setGeneratePerSection(event.target.checked)}
                />
                <CheckSquare className="w-4 h-4" />
                Generate one mini-deck per detected section
              </label>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <button type="button" className="btn-secondary" onClick={() => setIsCreateModalOpen(false)}>
                Cancel
              </button>
              <button type="button" className="btn-primary inline-flex items-center gap-2" onClick={() => void createCards()} disabled={isGenerating}>
                {isGenerating ? <Loader2 className="spinner" /> : <Plus className="w-4 h-4" />}
                Generate Deck
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {deckPendingDelete ? (
        <div className="study-modal-overlay" role="presentation" onClick={() => setDeckPendingDelete(null)}>
          <div className="study-modal card" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <h3 className="section-title">Delete deck?</h3>
            <p className="text-sm text-text-muted mt-2">
              This will permanently delete <span className="font-medium text-text-primary">{deckPendingDelete.name}</span> and its
              cards.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button type="button" className="btn-secondary" onClick={() => setDeckPendingDelete(null)}>
                Cancel
              </button>
              <button type="button" className="btn-danger" onClick={() => void removeDeck(deckPendingDelete)}>
                Delete deck
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {parentDeckModal ? (
        <div className="study-modal-overlay" role="presentation" onClick={() => setParentDeckModal(null)}>
          <div className="study-modal card" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-sage-600" />
                <h3 className="section-title">{parentDeckModal.name}</h3>
              </div>
              <button type="button" className="text-text-muted hover:text-text-primary" onClick={() => setParentDeckModal(null)}>
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-text-muted mt-1">
              {(childDecksByParentId.get(parentDeckModal.id) ?? []).length} child decks ·{' '}
              {(childDecksByParentId.get(parentDeckModal.id) ?? []).reduce((sum, d) => sum + d.card_count, 0)} total cards
            </p>

            <button
              type="button"
              className="btn-primary w-full inline-flex items-center justify-center gap-2 mt-4"
              onClick={() => void openAllChildCards(parentDeckModal)}
              disabled={loadingAllCards}
            >
              {loadingAllCards ? <Loader2 className="spinner" /> : <BookOpen className="w-4 h-4" />}
              Study All (shuffled)
            </button>

            <div className="mt-4 space-y-2">
              <p className="text-xs font-semibold text-text-secondary">Or pick a child deck:</p>
              {(childDecksByParentId.get(parentDeckModal.id) ?? []).map((childDeck) => (
                <button
                  key={childDeck.id}
                  type="button"
                  className="w-full text-left border border-border rounded p-3 hover:border-sage-300 hover:bg-sage-50 transition-colors"
                  onClick={() => {
                    setParentDeckModal(null)
                    void openRecentSet(childDeck.id)
                  }}
                  disabled={openingDeckId === childDeck.id}
                >
                  <p className="text-sm font-semibold text-text-primary">{childDeck.name}</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    {childDeck.card_count} cards · {childDeck.topic || 'General study'}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
