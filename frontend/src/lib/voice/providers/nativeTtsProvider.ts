import type { TtsProvider, TtsSpeakOptions } from '../types'

const normalize = (value: string): string =>
  value.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim()

export class NativeTtsProvider implements TtsProvider {
  readonly id = 'native' as const
  private activeUtterance: SpeechSynthesisUtterance | null = null
  private abortHandler: (() => void) | null = null

  isSupported(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window
  }

  async init(): Promise<void> {
    if (!this.isSupported()) {
      throw new Error('native_tts_not_supported')
    }
  }

  speakChunk(text: string, options?: TtsSpeakOptions): Promise<void> {
    if (!this.isSupported()) return Promise.reject(new Error('native_tts_not_supported'))
    const content = text.trim()
    if (!content) return Promise.resolve()
    this.cancel()

    return new Promise<void>((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(content)
      this.activeUtterance = utterance

      const voices = window.speechSynthesis.getVoices()
      const preferred = normalize(options?.preferredVoiceName || '')
      if (preferred) {
        const match = voices.find((voice) => normalize(voice.name).includes(preferred))
        if (match) utterance.voice = match
      }
      utterance.rate = options?.rate ?? 1
      utterance.pitch = options?.pitch ?? 1

      utterance.onend = () => {
        this.cleanupAbortBinding(options?.signal)
        this.activeUtterance = null
        resolve()
      }
      utterance.onerror = () => {
        this.cleanupAbortBinding(options?.signal)
        this.activeUtterance = null
        reject(new Error('native_tts_playback_error'))
      }

      if (options?.signal) {
        const onAbort = () => {
          this.cancel()
          reject(new DOMException('Aborted', 'AbortError'))
        }
        options.signal.addEventListener('abort', onAbort, { once: true })
        this.abortHandler = () => options.signal?.removeEventListener('abort', onAbort)
      }

      window.speechSynthesis.speak(utterance)
    })
  }

  cancel(): void {
    if (!this.isSupported()) return
    this.cleanupAbortBinding()
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
      window.speechSynthesis.cancel()
    }
    this.activeUtterance = null
  }

  dispose(): void {
    this.cancel()
  }

  private cleanupAbortBinding(signal?: AbortSignal): void {
    if (this.abortHandler) this.abortHandler()
    this.abortHandler = null
    if (signal) {
      // no-op; listeners are removed through abortHandler
    }
  }
}
