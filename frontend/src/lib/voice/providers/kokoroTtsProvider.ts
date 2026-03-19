import type { TtsProvider, TtsSpeakOptions } from '../types'

type WorkerInitPayload = {
  preferredDeviceOrder?: Array<'webgpu' | 'wasm'>
  dtype?: 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16'
}

type WorkerRequest =
  | { type: 'init'; payload?: WorkerInitPayload }
  | { type: 'synthesize'; payload: { id: string; text: string; voice?: string; speed?: number } }
  | { type: 'cancel' }
  | { type: 'dispose' }

type WorkerResponse =
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'error'; error: string; phase: 'init' | 'synthesize' }
  | { type: 'audio'; id: string; sampleRate: number; data: ArrayBuffer }
  | { type: 'cancelled' }
  | { type: 'disposed' }

const makeRequestId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`

const isLikelyMobile = (): boolean => {
  if (typeof navigator === 'undefined') return false
  const ua = navigator.userAgent || ''
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua)
}

export class KokoroTtsProvider implements TtsProvider {
  readonly id = 'kokoro' as const
  private worker: Worker | null = null
  private initPromise: Promise<void> | null = null
  private activeSource: AudioBufferSourceNode | null = null
  private audioContext: AudioContext | null = null
  private waiting = new Map<
    string,
    {
      resolve: (value: void | PromiseLike<void>) => void
      reject: (reason?: unknown) => void
      signal?: AbortSignal
    }
  >()
  private currentRequestId: string | null = null
  private disposeRequested = false
  private webgpuReady = false

  isSupported(): boolean {
    if (typeof window === 'undefined') return false
    if (typeof Worker === 'undefined') return false
    if (typeof AudioContext === 'undefined' && typeof (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext === 'undefined') {
      return false
    }
    // Desktop-first: allow mobile fallback to native by reporting unsupported.
    if (isLikelyMobile()) return false
    return true
  }

  async init(options?: TtsSpeakOptions): Promise<void> {
    if (!this.isSupported()) throw new Error('kokoro_not_supported')
    if (this.initPromise) return this.initPromise
    this.initPromise = this.initializeWorker(options)
    return this.initPromise
  }

  async speakChunk(text: string, options?: TtsSpeakOptions): Promise<void> {
    const content = text.trim()
    if (!content) return
    await this.init(options)
    if (!this.worker) throw new Error('kokoro_worker_unavailable')
    const signal = options?.signal
    const requestId = makeRequestId()
    this.currentRequestId = requestId

    const responsePromise = new Promise<void>((resolve, reject) => {
      this.waiting.set(requestId, { resolve, reject, signal })
      this.worker?.postMessage({
        type: 'synthesize',
        payload: {
          id: requestId,
          text: content,
          voice: options?.kokoroVoice || 'af_heart',
          speed: options?.kokoroSpeed ?? 1,
        },
      } satisfies WorkerRequest)
    })

    if (signal) {
      if (signal.aborted) {
        this.cancel()
        throw new DOMException('Aborted', 'AbortError')
      }
      signal.addEventListener(
        'abort',
        () => {
          this.cancel()
        },
        { once: true },
      )
    }

    return responsePromise
  }

  cancel(): void {
    if (this.worker) {
      this.worker.postMessage({ type: 'cancel' } satisfies WorkerRequest)
    }
    if (this.activeSource) {
      try {
        this.activeSource.stop()
      } catch {
        // no-op
      }
      this.activeSource.disconnect()
      this.activeSource = null
    }
    if (this.currentRequestId) {
      const pending = this.waiting.get(this.currentRequestId)
      if (pending) {
        pending.reject(new DOMException('Aborted', 'AbortError'))
        this.waiting.delete(this.currentRequestId)
      }
    }
    this.currentRequestId = null
  }

  dispose(): void {
    this.disposeRequested = true
    this.cancel()
    if (this.worker) {
      this.worker.postMessage({ type: 'dispose' } satisfies WorkerRequest)
      this.worker.terminate()
      this.worker = null
    }
    this.waiting.forEach(({ reject }) => reject(new Error('kokoro_disposed')))
    this.waiting.clear()
    this.initPromise = null
    if (this.audioContext) {
      void this.audioContext.close()
      this.audioContext = null
    }
  }

  get isWebGpuReady(): boolean {
    return this.webgpuReady
  }

  private async initializeWorker(options?: TtsSpeakOptions): Promise<void> {
    this.worker = new Worker(new URL('../workers/kokoro.worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      void this.handleWorkerMessage(event.data)
    }
    this.worker.onerror = () => {
      this.rejectAll(new Error('kokoro_worker_error'))
    }
    this.worker.postMessage({
      type: 'init',
      payload: {
        preferredDeviceOrder: ['webgpu', 'wasm'],
        dtype: options?.kokoroSpeed && options.kokoroSpeed > 1.2 ? 'q8' : 'q8',
      },
    } satisfies WorkerRequest)

    await new Promise<void>((resolve, reject) => {
      const maxWaitMs = 90000
      const timeoutId = window.setTimeout(() => {
        reject(new Error('kokoro_init_timeout'))
      }, maxWaitMs)
      const onReady = (event: MessageEvent<WorkerResponse>) => {
        if (event.data.type === 'ready') {
          this.webgpuReady = event.data.device === 'webgpu'
          this.worker?.removeEventListener('message', onReady)
          clearTimeout(timeoutId)
          resolve()
        } else if (event.data.type === 'error' && event.data.phase === 'init') {
          this.worker?.removeEventListener('message', onReady)
          clearTimeout(timeoutId)
          reject(new Error(event.data.error))
        }
      }
      this.worker?.addEventListener('message', onReady)
    })
  }

  private async handleWorkerMessage(message: WorkerResponse): Promise<void> {
    if (this.disposeRequested) return
    if (message.type === 'audio') {
      const waiting = this.waiting.get(message.id)
      if (!waiting) return
      this.waiting.delete(message.id)
      if (waiting.signal?.aborted) {
        waiting.reject(new DOMException('Aborted', 'AbortError'))
        return
      }
      try {
        await this.playPcm(message.data, message.sampleRate)
        waiting.resolve()
      } catch (error) {
        waiting.reject(error)
      } finally {
        if (this.currentRequestId === message.id) {
          this.currentRequestId = null
        }
      }
      return
    }
    if (message.type === 'error' && message.phase === 'synthesize') {
      if (this.currentRequestId) {
        const waiting = this.waiting.get(this.currentRequestId)
        if (waiting) {
          this.waiting.delete(this.currentRequestId)
          waiting.reject(new Error(message.error))
        }
      }
      this.currentRequestId = null
    }
  }

  private async playPcm(buffer: ArrayBuffer, sampleRate: number): Promise<void> {
    const context = this.ensureAudioContext()
    await context.resume()
    const samples = new Float32Array(buffer)
    const audioBuffer = context.createBuffer(1, samples.length, sampleRate)
    audioBuffer.copyToChannel(samples, 0)
    const source = context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(context.destination)
    this.activeSource = source
    await new Promise<void>((resolve) => {
      source.onended = () => resolve()
      source.start()
    })
    source.disconnect()
    if (this.activeSource === source) {
      this.activeSource = null
    }
  }

  private ensureAudioContext(): AudioContext {
    if (this.audioContext) return this.audioContext
    const Ctor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctor) throw new Error('audio_context_not_supported')
    this.audioContext = new Ctor()
    return this.audioContext
  }

  private rejectAll(error: Error): void {
    this.waiting.forEach(({ reject }) => reject(error))
    this.waiting.clear()
    this.currentRequestId = null
  }
}
