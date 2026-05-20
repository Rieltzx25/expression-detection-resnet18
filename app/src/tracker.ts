import { iou, type FaceResult } from './inference'

const EMA_ALPHA = 0.4   // smoothing factor for emotion probs (higher = more responsive)
const TRACK_IOU_THRESHOLD = 0.3
const MAX_MISSING_FRAMES = 8 // ~250ms at 30fps before dropping a track

const TRACK_COLORS = [
  '#10b981', '#3b82f6', '#f59e0b', '#ec4899',
  '#8b5cf6', '#06b6d4', '#ef4444', '#84cc16',
]

export interface Track {
  id: number
  color: string
  /** Smoothed box (EMA). */
  box: { x1: number; y1: number; x2: number; y2: number }
  /** Latest raw detection. */
  raw: FaceResult
  /** Smoothed emotion probabilities. */
  smoothScores: Float32Array
  /** Argmax of smoothed scores. */
  topIdx: number
  /** Cropped 48x48 grayscale of last seen frame. */
  crop: Uint8ClampedArray
  /** Frames since last detection. */
  missing: number
  /** Total frames this track has been alive. */
  age: number
}

export class Tracker {
  private tracks: Track[] = []
  private nextId = 1
  private nextColor = 0

  step(detections: FaceResult[]): Track[] {
    // Greedy IoU match
    const used = new Set<number>()
    const newTracks: Track[] = []

    // For each existing track, find best matching detection
    for (const t of this.tracks) {
      let bestIdx = -1
      let bestIoU = TRACK_IOU_THRESHOLD
      const tBox = { ...t.box, score: 1 }
      for (let i = 0; i < detections.length; i++) {
        if (used.has(i)) continue
        const v = iou(tBox, detections[i])
        if (v > bestIoU) { bestIoU = v; bestIdx = i }
      }
      if (bestIdx >= 0) {
        const det = detections[bestIdx]
        used.add(bestIdx)
        // EMA smooth box
        const a = 0.5
        t.box = {
          x1: t.box.x1 * (1 - a) + det.x1 * a,
          y1: t.box.y1 * (1 - a) + det.y1 * a,
          x2: t.box.x2 * (1 - a) + det.x2 * a,
          y2: t.box.y2 * (1 - a) + det.y2 * a,
        }
        // EMA smooth scores
        for (let c = 0; c < t.smoothScores.length; c++) {
          t.smoothScores[c] = t.smoothScores[c] * (1 - EMA_ALPHA) + det.scores[c] * EMA_ALPHA
        }
        // Recompute argmax
        let top = 0
        for (let c = 1; c < t.smoothScores.length; c++) if (t.smoothScores[c] > t.smoothScores[top]) top = c
        t.topIdx = top
        t.raw = det
        t.crop = det.crop
        t.missing = 0
        t.age++
        newTracks.push(t)
      } else {
        t.missing++
        if (t.missing <= MAX_MISSING_FRAMES) newTracks.push(t)
      }
    }

    // Unmatched detections become new tracks
    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue
      const det = detections[i]
      newTracks.push({
        id: this.nextId++,
        color: TRACK_COLORS[(this.nextColor++) % TRACK_COLORS.length],
        box: { x1: det.x1, y1: det.y1, x2: det.x2, y2: det.y2 },
        raw: det,
        smoothScores: new Float32Array(det.scores),
        topIdx: det.topIdx,
        crop: det.crop,
        missing: 0,
        age: 1,
      })
    }

    this.tracks = newTracks
    return this.activeTracks()
  }

  activeTracks(): Track[] {
    return this.tracks.filter(t => t.missing === 0)
  }

  reset() {
    this.tracks = []
  }

  /** Returns the largest active track (by area). */
  dominant(): Track | null {
    const active = this.activeTracks()
    if (active.length === 0) return null
    let best = active[0]
    let bestArea = (best.box.x2 - best.box.x1) * (best.box.y2 - best.box.y1)
    for (let i = 1; i < active.length; i++) {
      const t = active[i]
      const a = (t.box.x2 - t.box.x1) * (t.box.y2 - t.box.y1)
      if (a > bestArea) { best = t; bestArea = a }
    }
    return best
  }
}
