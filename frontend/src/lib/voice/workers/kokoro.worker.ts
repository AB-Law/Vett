/// <reference lib="webworker" />

import { KokoroTTS } from 'kokoro-js'

type InitPayload = {
  preferredDeviceOrder?: Array<'webgpu' | 'wasm'>
  dtype?: 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16'
}

type KokoroVoice =
  | 'af_heart'
  | 'af_alloy'
  | 'af_aoede'
  | 'af_bella'
  | 'af_jessica'
  | 'af_kore'
  | 'af_nicole'
  | 'af_nova'
  | 'af_river'
  | 'af_sarah'
  | 'af_sky'
  | 'am_adam'
  | 'am_echo'
  | 'am_eric'
  | 'am_fenrir'
  | 'am_liam'
  | 'am_michael'
  | 'am_onyx'
  | 'am_puck'
  | 'am_santa'
  | 'bf_alice'
  | 'bf_emma'
  | 'bf_isabella'
  | 'bf_lily'
  | 'bm_daniel'
  | 'bm_fable'
  | 'bm_george'
  | 'bm_lewis'

type SynthesizePayload = {
  id: string
  text: string
  voice?: KokoroVoice
  speed?: number
}

type WorkerRequest =
  | { type: 'init'; payload?: InitPayload }
  | { type: 'synthesize'; payload: SynthesizePayload }
  | { type: 'cancel' }
  | { type: 'dispose' }

type WorkerResponse =
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'error'; error: string; phase: 'init' | 'synthesize' }
  | { type: 'audio'; id: string; sampleRate: number; data: ArrayBuffer }
  | { type: 'cancelled' }
  | { type: 'disposed' }

type KokoroAudioLike = {
  audio?: Float32Array | number[]
  data?: Float32Array | number[]
  sampling_rate?: number
  sampleRate?: number
}

let ttsInstance: Awaited<ReturnType<typeof KokoroTTS.from_pretrained>> | null = null
let activeDevice: 'webgpu' | 'wasm' = 'wasm'
let cancelledGeneration = false

const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX'

const supportsWebGPU = (): boolean => {
  // Worker global has navigator in modern browsers.
  const nav = self.navigator as Navigator & { gpu?: unknown }
  return Boolean(nav?.gpu)
}

const post = (message: WorkerResponse): void => {
  self.postMessage(message)
}

const toAudioArray = (result: unknown): Float32Array => {
  const candidate = result as KokoroAudioLike
  const raw = candidate?.audio ?? candidate?.data
  if (raw instanceof Float32Array) return raw
  if (Array.isArray(raw)) return Float32Array.from(raw)
  throw new Error('kokoro_invalid_audio')
}

const toSampleRate = (result: unknown): number => {
  const candidate = result as KokoroAudioLike
  return Number(candidate?.sampleRate ?? candidate?.sampling_rate ?? 24000)
}

const init = async (payload?: InitPayload): Promise<void> => {
  if (ttsInstance) return
  const requested = payload?.preferredDeviceOrder?.length ? payload.preferredDeviceOrder : ['webgpu', 'wasm']
  const fallbackOrder = Array.from(new Set(requested)) as Array<'webgpu' | 'wasm'>
  const dtype = payload?.dtype ?? 'q8'
  let lastError: Error | null = null

  for (const device of fallbackOrder) {
    if (device === 'webgpu' && !supportsWebGPU()) {
      continue
    }
    try {
      const instance = await KokoroTTS.from_pretrained(MODEL_ID, {
        device,
        dtype: device === 'webgpu' ? 'fp32' : dtype,
      })
      ttsInstance = instance
      activeDevice = device
      post({ type: 'ready', device })
      return
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))
    }
  }
  throw lastError || new Error('kokoro_init_failed')
}

const synthesize = async (payload: SynthesizePayload): Promise<void> => {
  if (!ttsInstance) throw new Error('kokoro_not_initialized')
  cancelledGeneration = false
  const output = await ttsInstance.generate(payload.text, {
    voice: payload.voice || 'af_heart',
    speed: payload.speed ?? 1.0,
  })
  if (cancelledGeneration) {
    post({ type: 'cancelled' })
    return
  }
  const audio = toAudioArray(output)
  const sampleRate = toSampleRate(output)
  const byteView = new Uint8Array(audio.buffer, audio.byteOffset, audio.byteLength)
  const transferable = byteView.slice().buffer
  post({ type: 'audio', id: payload.id, sampleRate, data: transferable })
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelledGeneration = true
    post({ type: 'cancelled' })
    return
  }
  if (request.type === 'dispose') {
    cancelledGeneration = true
    ttsInstance = null
    post({ type: 'disposed' })
    return
  }
  if (request.type === 'init') {
    void init(request.payload).catch((error) => {
      const message = error instanceof Error ? error.message : String(error)
      post({ type: 'error', phase: 'init', error: message })
    })
    return
  }
  if (request.type === 'synthesize') {
    void synthesize(request.payload).catch((error) => {
      const message = error instanceof Error ? error.message : String(error)
      post({ type: 'error', phase: 'synthesize', error: message })
    })
  }
}
