import { TtsProviderRouter } from './ttsProviderRouter'

type SpeechRecognitionResultEventLike = Event & {
  results: ArrayLike<{ isFinal: boolean; 0?: { transcript?: string } }>
}

type SpeechRecognitionLike = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onstart: ((ev: Event) => unknown) | null
  onend: ((ev: Event) => unknown) | null
  onerror: ((ev: Event & { error?: string }) => unknown) | null
  onspeechstart: ((ev: Event) => unknown) | null
  onresult: ((ev: SpeechRecognitionResultEventLike) => unknown) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike

export type VoiceConversationState = 'IDLE' | 'LISTENING' | 'THINKING' | 'SPEAKING'

export interface VoicePreferences {
  preferredVoiceName?: string
  rate?: number
  pitch?: number
  ttsProvider?: 'native' | 'kokoro'
  kokoroVoice?: string
  kokoroSpeed?: number
}

type ManagerCallbacks = {
  onStateChange: (state: VoiceConversationState) => void
  onFinalTranscript: (text: string) => void
  onError: (message: string) => void
  onBargeIn: () => void
  getActiveAssistantText: () => string
  onTtsStatus?: (message: string, provider: 'native' | 'kokoro') => void
}

const normalize = (value: string): string =>
  value.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim()

const isLikelyEcho = (heard: string, assistant: string): boolean => {
  const a = normalize(assistant)
  const b = normalize(heard)
  if (!a || !b) return false
  if (a.includes(b) || b.includes(a)) return true
  const aTokens = new Set(a.split(' '))
  const bTokens = b.split(' ')
  const overlap = bTokens.filter((token) => aTokens.has(token)).length
  return overlap >= Math.max(3, Math.floor(bTokens.length * 0.7))
}

const SENTENCE_SPLIT_REGEX = /[^.!?]+[.!?]+(?:\s+|$)/g

export class AdvancedVoiceManager {
  private readonly callbacks: ManagerCallbacks
  private readonly ttsRouter: TtsProviderRouter
  private recognition: SpeechRecognitionLike | null = null
  private shouldListen = false
  private restarting = false
  private speaking = false
  private currentText = ''
  private preferences: VoicePreferences = {}
  private sentenceBuffer = ''
  private speechQueue: string[] = []
  private queueRunning = false
  private playbackAbortController: AbortController | null = null

  constructor(callbacks: ManagerCallbacks) {
    this.callbacks = callbacks
    this.ttsRouter = new TtsProviderRouter({
      onStatus: (event) => this.callbacks.onTtsStatus?.(event.message, event.activeProvider),
    })
    if (typeof window === 'undefined') return
    const ctor = (window as Window & { webkitSpeechRecognition?: SpeechRecognitionCtor; SpeechRecognition?: SpeechRecognitionCtor })
      .SpeechRecognition
      || (window as Window & { webkitSpeechRecognition?: SpeechRecognitionCtor; SpeechRecognition?: SpeechRecognitionCtor })
        .webkitSpeechRecognition
    if (!ctor) return

    const recognition = new ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.onstart = () => {
      this.callbacks.onStateChange(this.speaking ? 'SPEAKING' : 'LISTENING')
    }
    recognition.onspeechstart = () => {
      if (this.speaking) {
        this.callbacks.onBargeIn()
        this.cancelSpeech()
      }
    }
    recognition.onresult = (event) => {
      const results = Array.from(event.results || [])
      const finalChunk = results
        .filter((result) => result.isFinal)
        .map((result) => result[0]?.transcript || '')
        .join(' ')
        .trim()
      if (!finalChunk) return
      if (isLikelyEcho(finalChunk, this.callbacks.getActiveAssistantText())) return
      this.callbacks.onFinalTranscript(finalChunk)
    }
    recognition.onerror = (event) => {
      const msg = event.error || 'speech_recognition_error'
      this.callbacks.onError(msg)
      this.restartLater()
    }
    recognition.onend = () => {
      if (this.shouldListen && !this.restarting) {
        this.restartLater()
      } else if (!this.shouldListen) {
        this.callbacks.onStateChange(this.speaking ? 'SPEAKING' : 'IDLE')
      }
    }
    this.recognition = recognition
  }

  get isSupported(): boolean {
    return Boolean(this.recognition)
  }

  setPreferences(preferences: VoicePreferences): void {
    this.preferences = { ...this.preferences, ...preferences }
    this.ttsRouter.setPreferredProvider(this.preferences.ttsProvider || 'kokoro')
    void this.ttsRouter.warmup(this.toTtsOptions()).catch(() => {
      // Router emits status and fallback details.
    })
  }

  startListening(): void {
    if (!this.recognition) return
    this.shouldListen = true
    try {
      this.recognition.start()
      this.callbacks.onStateChange(this.speaking ? 'SPEAKING' : 'LISTENING')
    } catch {
      this.restartLater()
    }
  }

  stopListening(): void {
    this.shouldListen = false
    if (!this.recognition) return
    try {
      this.recognition.stop()
    } catch {
      // no-op
    }
  }

  speak(text: string): void {
    if (!text.trim()) return
    this.cancelSpeech()
    this.currentText = text
    this.enqueueText(text, true)
  }

  onAssistantToken(token: string): void {
    if (!token) return
    this.currentText += token
    this.enqueueText(token, false)
  }

  flushAssistantSpeechBuffer(): void {
    const chunk = this.sentenceBuffer.trim()
    this.sentenceBuffer = ''
    if (!chunk) return
    this.speechQueue.push(chunk)
    void this.runSpeechQueue()
  }

  getCurrentSpeechText(): string {
    return this.currentText
  }

  cancelSpeech(): void {
    if (this.playbackAbortController) {
      this.playbackAbortController.abort()
      this.playbackAbortController = null
    }
    this.ttsRouter.cancel()
    this.speechQueue = []
    this.sentenceBuffer = ''
    this.speaking = false
    this.queueRunning = false
    this.currentText = ''
    this.callbacks.onStateChange(this.shouldListen ? 'LISTENING' : 'IDLE')
  }

  dispose(): void {
    this.stopListening()
    this.cancelSpeech()
    this.ttsRouter.dispose()
  }

  private restartLater(): void {
    if (!this.shouldListen || !this.recognition) return
    this.restarting = true
    window.setTimeout(() => {
      this.restarting = false
      if (!this.shouldListen || !this.recognition) return
      try {
        this.recognition.start()
      } catch {
        this.restartLater()
      }
    }, 1200)
  }

  private enqueueText(input: string, flushTrailing: boolean): void {
    this.sentenceBuffer += input
    const segments = this.extractCompletedSentences(this.sentenceBuffer)
    if (segments.sentences.length > 0) {
      this.sentenceBuffer = segments.remainder
      this.speechQueue.push(...segments.sentences)
      void this.runSpeechQueue()
    }
    if (flushTrailing) {
      this.flushAssistantSpeechBuffer()
    }
  }

  private extractCompletedSentences(buffer: string): { sentences: string[]; remainder: string } {
    const sentences: string[] = []
    let lastIndex = 0
    SENTENCE_SPLIT_REGEX.lastIndex = 0
    let match = SENTENCE_SPLIT_REGEX.exec(buffer)
    while (match) {
      const chunk = match[0].trim()
      if (chunk) sentences.push(chunk)
      lastIndex = SENTENCE_SPLIT_REGEX.lastIndex
      match = SENTENCE_SPLIT_REGEX.exec(buffer)
    }
    return {
      sentences,
      remainder: buffer.slice(lastIndex),
    }
  }

  private async runSpeechQueue(): Promise<void> {
    if (this.queueRunning) return
    this.queueRunning = true
    while (this.speechQueue.length > 0) {
      const next = this.speechQueue.shift()
      if (!next) continue
      this.speaking = true
      this.callbacks.onStateChange('SPEAKING')
      this.playbackAbortController = new AbortController()
      try {
        await this.ttsRouter.speakChunk(next, {
          ...this.toTtsOptions(),
          signal: this.playbackAbortController.signal,
        })
      } catch (error) {
        const isAbort = error instanceof DOMException && error.name === 'AbortError'
        if (!isAbort) {
          this.callbacks.onError(error instanceof Error ? error.message : 'voice_playback_error')
        }
      } finally {
        this.playbackAbortController = null
      }
      if (!this.queueRunning) break
    }
    this.queueRunning = false
    this.speaking = false
    this.callbacks.onStateChange(this.shouldListen ? 'LISTENING' : 'IDLE')
  }

  private toTtsOptions() {
    return {
      preferredVoiceName: this.preferences.preferredVoiceName,
      rate: this.preferences.rate,
      pitch: this.preferences.pitch,
      kokoroVoice: this.preferences.kokoroVoice,
      kokoroSpeed: this.preferences.kokoroSpeed,
    }
  }
}
