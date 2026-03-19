import { KokoroTtsProvider } from './providers/kokoroTtsProvider'
import { NativeTtsProvider } from './providers/nativeTtsProvider'
import type { TtsProvider, TtsProviderId, TtsSpeakOptions, TtsStatusEvent } from './types'

type RouterOptions = {
  onStatus?: (event: TtsStatusEvent) => void
}

export class TtsProviderRouter {
  private readonly onStatus?: (event: TtsStatusEvent) => void
  private readonly providers: Record<TtsProviderId, TtsProvider> = {
    native: new NativeTtsProvider(),
    kokoro: new KokoroTtsProvider(),
  }
  private preferredProviderId: TtsProviderId = 'kokoro'
  private activeProviderId: TtsProviderId = 'native'
  private initialized = new Set<TtsProviderId>()

  constructor(options: RouterOptions = {}) {
    this.onStatus = options.onStatus
  }

  setPreferredProvider(id: TtsProviderId): void {
    this.preferredProviderId = id
  }

  get activeProviderIdValue(): TtsProviderId {
    return this.activeProviderId
  }

  async warmup(options?: TtsSpeakOptions): Promise<void> {
    await this.ensureActiveProvider(options)
  }

  async speakChunk(text: string, options?: TtsSpeakOptions): Promise<void> {
    const provider = await this.ensureActiveProvider(options)
    try {
      await provider.speakChunk(text, options)
    } catch (error) {
      if (this.activeProviderId === 'kokoro') {
        this.activeProviderId = 'native'
        this.emitStatus({
          status: 'fallback_native',
          message: 'Fallback to native voice',
          activeProvider: 'native',
        })
        const native = this.providers.native
        if (!this.initialized.has('native')) {
          await native.init(options)
          this.initialized.add('native')
        }
        await native.speakChunk(text, options)
        return
      }
      throw error
    }
  }

  cancel(): void {
    this.providers[this.activeProviderId].cancel()
    // Also cancel native queue in case fallback happened mid-flow.
    this.providers.native.cancel()
  }

  dispose(): void {
    this.providers.kokoro.dispose()
    this.providers.native.dispose()
    this.initialized.clear()
  }

  private async ensureActiveProvider(options?: TtsSpeakOptions): Promise<TtsProvider> {
    if (this.preferredProviderId === 'kokoro') {
      const kokoro = this.providers.kokoro
      if (kokoro.isSupported()) {
        try {
          this.emitStatus({
            status: 'loading',
            message: 'Loading voice model...',
            activeProvider: 'kokoro',
          })
          if (!this.initialized.has('kokoro')) {
            await kokoro.init(options)
            this.initialized.add('kokoro')
          }
          this.activeProviderId = 'kokoro'
          this.emitStatus({
            status: 'ready',
            message: 'Voice model ready',
            activeProvider: 'kokoro',
          })
          return kokoro
        } catch {
          this.activeProviderId = 'native'
          this.emitStatus({
            status: 'fallback_native',
            message: 'Fallback to native voice',
            activeProvider: 'native',
          })
        }
      } else {
        this.activeProviderId = 'native'
        this.emitStatus({
          status: 'fallback_native',
          message: 'Fallback to native voice',
          activeProvider: 'native',
        })
      }
    }

    const native = this.providers.native
    if (!this.initialized.has('native')) {
      await native.init(options)
      this.initialized.add('native')
    }
    if (this.activeProviderId === 'native') {
      this.emitStatus({
        status: 'ready',
        message: 'Voice model ready',
        activeProvider: 'native',
      })
    }
    return native
  }

  private emitStatus(event: TtsStatusEvent): void {
    this.onStatus?.(event)
  }
}
