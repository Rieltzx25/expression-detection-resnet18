import { useMemo } from 'react'
import { EXPR_LABELS } from './inference'

const COLORS: Record<string, string> = {
  angry: '#ef4444', disgust: '#84cc16', fear: '#a855f7',
  happy: '#10b981', neutral: '#71717a', sad: '#3b82f6', surprise: '#f59e0b',
}

export interface TimelinePoint {
  t: number          // ms timestamp
  topIdx: number     // 0..6
  topProb: number    // 0..1
}

interface Props {
  points: TimelinePoint[]
  windowMs: number
  width?: number
  height?: number
}

export function Timeline({ points, windowMs, width = 320, height = 80 }: Props) {
  const { paths, activeLabels } = useMemo(() => {
    if (points.length < 2) return { paths: [] as { d: string; color: string; idx: number }[], activeLabels: new Set<number>() }
    const tNow = points[points.length - 1].t
    const tStart = tNow - windowMs
    const filtered = points.filter(p => p.t >= tStart)
    if (filtered.length < 2) return { paths: [] as { d: string; color: string; idx: number }[], activeLabels: new Set<number>() }

    // Group by topIdx, build paths where we draw segments
    const usedIdx = new Set<number>()
    const segments: Record<number, { x: number; y: number }[][]> = {}
    let curIdx = -1
    let curSeg: { x: number; y: number }[] = []
    for (const p of filtered) {
      const x = ((p.t - tStart) / windowMs) * width
      const y = (1 - p.topProb) * (height - 8) + 4
      if (p.topIdx !== curIdx) {
        if (curSeg.length > 1) {
          (segments[curIdx] ||= []).push(curSeg)
        }
        curSeg = []
        curIdx = p.topIdx
      }
      curSeg.push({ x, y })
      usedIdx.add(p.topIdx)
    }
    if (curSeg.length > 1) {
      (segments[curIdx] ||= []).push(curSeg)
    }
    const paths: { d: string; color: string; idx: number }[] = []
    for (const idx of Object.keys(segments)) {
      const i = +idx
      for (const seg of segments[i]) {
        const d = seg.map((pt, j) => `${j === 0 ? 'M' : 'L'}${pt.x.toFixed(1)},${pt.y.toFixed(1)}`).join(' ')
        paths.push({ d, color: COLORS[EXPR_LABELS[i]] ?? '#71717a', idx: i })
      }
    }
    return { paths, activeLabels: usedIdx }
  }, [points, windowMs, width, height])

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
        {/* gridlines */}
        <line x1={0} x2={width} y1={4} y2={4} stroke="#e4e4e7" strokeDasharray="2 4" />
        <line x1={0} x2={width} y1={height - 4} y2={height - 4} stroke="#e4e4e7" strokeDasharray="2 4" />
        <line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke="#e4e4e7" strokeDasharray="2 4" />
        {paths.map((p, i) => (
          <path key={i} d={p.d} stroke={p.color} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        ))}
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
        {[...activeLabels].sort().map(i => (
          <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--ink-2)' }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: COLORS[EXPR_LABELS[i]], display: 'inline-block' }} />
            {EXPR_LABELS[i]}
          </span>
        ))}
      </div>
    </div>
  )
}
