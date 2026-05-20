import * as ort from 'onnxruntime-web'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/'

export const EXPR_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] as const
export type ExprLabel = typeof EXPR_LABELS[number]

const FACE_INPUT = 640
const EMO_INPUT = 48

let faceSession: ort.InferenceSession | null = null
let emoSession: ort.InferenceSession | null = null

export async function loadModels(onProgress?: (msg: string) => void) {
  onProgress?.('Loading face detector (YOLOv11n-face, 10 MB)…')
  if (!faceSession) {
    faceSession = await ort.InferenceSession.create('/models/face.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  onProgress?.('Loading emotion model (ResNet18, 44 MB)…')
  if (!emoSession) {
    emoSession = await ort.InferenceSession.create('/models/expression.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  onProgress?.('Ready')
}

export interface FaceBox {
  x1: number; y1: number; x2: number; y2: number
  score: number
}

export interface FaceResult extends FaceBox {
  /** Index in original detect call. */
  idx: number
  /** Softmax probabilities over EXPR_LABELS. */
  scores: Float32Array
  /** Argmax index. */
  topIdx: number
  /** 48x48 grayscale crop, raw 0..255 luminance for display. */
  crop: Uint8ClampedArray
}

const _scratchCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null
const _emoCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null
if (_scratchCanvas) { _scratchCanvas.width = FACE_INPUT; _scratchCanvas.height = FACE_INPUT }
if (_emoCanvas) { _emoCanvas.width = EMO_INPUT; _emoCanvas.height = EMO_INPUT }

export async function detect(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  srcW: number,
  srcH: number,
  options: { minScore?: number; cropPad?: number } = {},
): Promise<FaceResult[]> {
  if (!faceSession || !emoSession) throw new Error('Models not loaded')
  const minScore = options.minScore ?? 0.4
  const cropPad = options.cropPad ?? 0.15

  // ---------- Face detection ----------
  const r = Math.min(FACE_INPUT / srcW, FACE_INPUT / srcH)
  const newW = Math.round(srcW * r)
  const newH = Math.round(srcH * r)
  const padX = (FACE_INPUT - newW) / 2
  const padY = (FACE_INPUT - newH) / 2

  const fctx = _scratchCanvas!.getContext('2d', { willReadFrequently: true })!
  fctx.fillStyle = '#727272'
  fctx.fillRect(0, 0, FACE_INPUT, FACE_INPUT)
  fctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH)
  const fpx = fctx.getImageData(0, 0, FACE_INPUT, FACE_INPUT).data

  const plane = FACE_INPUT * FACE_INPUT
  const fInp = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    fInp[i] = fpx[i * 4] / 255
    fInp[plane + i] = fpx[i * 4 + 1] / 255
    fInp[2 * plane + i] = fpx[i * 4 + 2] / 255
  }
  const fTensor = new ort.Tensor('float32', fInp, [1, 3, FACE_INPUT, FACE_INPUT])
  const fOut = await faceSession.run({ images: fTensor })
  const fKey = Object.keys(fOut)[0]
  const fData = fOut[fKey].data as Float32Array
  const fDims = fOut[fKey].dims as number[]
  const N = fDims[2]

  const candidates: FaceBox[] = []
  for (let i = 0; i < N; i++) {
    const s = fData[4 * N + i]
    if (s < minScore) continue
    const cx = fData[0 * N + i]
    const cy = fData[1 * N + i]
    const w = fData[2 * N + i]
    const h = fData[3 * N + i]
    let x1 = cx - w / 2, y1 = cy - h / 2
    let x2 = cx + w / 2, y2 = cy + h / 2
    x1 = (x1 - padX) / r; y1 = (y1 - padY) / r
    x2 = (x2 - padX) / r; y2 = (y2 - padY) / r
    candidates.push({ x1, y1, x2, y2, score: s })
  }
  const faces = nms(candidates, 0.45)
  if (faces.length === 0) return []

  // ---------- Batched emotion inference ----------
  const emoCtx = _emoCanvas!.getContext('2d', { willReadFrequently: true })!
  const batchInput = new Float32Array(faces.length * EMO_INPUT * EMO_INPUT)
  const crops: Uint8ClampedArray[] = []

  for (let f = 0; f < faces.length; f++) {
    const b = faces[f]
    // pad crop
    const cw = b.x2 - b.x1
    const ch = b.y2 - b.y1
    const pad = Math.max(cw, ch) * cropPad
    const cx1 = Math.max(0, b.x1 - pad)
    const cy1 = Math.max(0, b.y1 - pad)
    const cx2 = Math.min(srcW, b.x2 + pad)
    const cy2 = Math.min(srcH, b.y2 + pad)
    emoCtx.drawImage(source, cx1, cy1, cx2 - cx1, cy2 - cy1, 0, 0, EMO_INPUT, EMO_INPUT)
    const epx = emoCtx.getImageData(0, 0, EMO_INPUT, EMO_INPUT).data
    const cropGray = new Uint8ClampedArray(EMO_INPUT * EMO_INPUT)
    const offset = f * EMO_INPUT * EMO_INPUT
    for (let i = 0; i < EMO_INPUT * EMO_INPUT; i++) {
      const r2 = epx[i * 4], g2 = epx[i * 4 + 1], b2 = epx[i * 4 + 2]
      const gray = (0.299 * r2 + 0.587 * g2 + 0.114 * b2)
      cropGray[i] = gray
      batchInput[offset + i] = (gray / 255 - 0.5) / 0.5
    }
    crops.push(cropGray)
  }

  const emoTensor = new ort.Tensor('float32', batchInput, [faces.length, 1, EMO_INPUT, EMO_INPUT])
  const eOut = await emoSession.run({ input: emoTensor })
  const logits = eOut.logits.data as Float32Array
  const C = EXPR_LABELS.length

  const results: FaceResult[] = []
  for (let f = 0; f < faces.length; f++) {
    const slice = logits.subarray(f * C, (f + 1) * C)
    const probs = softmax(slice)
    let topIdx = 0
    for (let i = 1; i < C; i++) if (probs[i] > probs[topIdx]) topIdx = i
    results.push({
      ...faces[f],
      idx: f,
      scores: probs,
      topIdx,
      crop: crops[f],
    })
  }
  return results
}

function softmax(arr: Float32Array): Float32Array {
  let m = -Infinity
  for (const v of arr) if (v > m) m = v
  const out = new Float32Array(arr.length)
  let s = 0
  for (let i = 0; i < arr.length; i++) { out[i] = Math.exp(arr[i] - m); s += out[i] }
  for (let i = 0; i < arr.length; i++) out[i] /= s
  return out
}

function iou(a: FaceBox, b: FaceBox): number {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return ua > 0 ? inter / ua : 0
}

function nms<T extends FaceBox>(boxes: T[], thr: number): T[] {
  boxes.sort((a, b) => b.score - a.score)
  const keep: T[] = []
  const taken = new Array<boolean>(boxes.length).fill(false)
  for (let i = 0; i < boxes.length; i++) {
    if (taken[i]) continue
    keep.push(boxes[i])
    for (let j = i + 1; j < boxes.length; j++) {
      if (taken[j]) continue
      if (iou(boxes[i], boxes[j]) >= thr) taken[j] = true
    }
  }
  return keep
}

export { iou }
