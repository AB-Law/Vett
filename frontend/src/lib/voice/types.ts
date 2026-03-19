export type TtsProviderId = 'native' | 'kokoro'

export type TtsStatus = 'loading' | 'ready' | 'fallback_native' | 'error'

export type TtsStatusEvent = {
  status: TtsStatus
  message: string
  activeProvider: TtsProviderId
}

export interface TtsSpeakOptions {
  preferredVoiceName?: string
  rate?: number
  pitch?: number
  kokoroVoice?: string
  kokoroSpeed?: number
  signal?: AbortSignal
}

export interface TtsProvider {
  readonly id: TtsProviderId
  isSupported(): boolean
  init(options?: TtsSpeakOptions): Promise<void>
  speakChunk(text: string, options?: TtsSpeakOptions): Promise<void>
  cancel(): void
  dispose(): void
}
