import { EXPR_LABELS } from './inference'

const COLORS: Record<string, string> = {
  angry: '#ef4444', disgust: '#84cc16', fear: '#a855f7',
  happy: '#10b981', neutral: '#71717a', sad: '#3b82f6', surprise: '#f59e0b',
}

interface Props {
  /** Total ms per emotion index (length 7). */
  totals: number[]
  size?: number
  thickness?: number
}

export function Donut({ totals, size = 180, thickness = 22 }: Props) {
  const sum = totals.reduce((a, b) => a + b, 0)
  if (sum <= 0) return <div style={{ color: 'var(--muted)', fontSize: 13 }}>No data captured.</div>
  const cx = size / 2, cy = size / 2
  const r = (size - thickness) / 2
  const C = 2 * Math.PI * r

  let acc = 0
  const segs: { color: string; len: number; off: number; label: string; pct: number }[] = []
  for (let i = 0; i < totals.length; i++) {
    if (totals[i] <= 0) continue
    const frac = totals[i] / sum
    segs.push({
      color: COLORS[EXPR_LABELS[i]],
      len: frac * C,
      off: -acc,
      label: EXPR_LABELS[i],
      pct: frac,
    })
    acc += frac * C
  }

  const sorted = [...segs].sort((a, b) => b.pct - a.pct)
  const top = sorted[0]

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flexShrink: 0 }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--line)" strokeWidth={thickness} />
        {segs.map((s, i) => (
          <circle
            key={i}
            cx={cx} cy={cy} r={r} fill="none"
            stroke={s.color} strokeWidth={thickness}
            strokeDasharray={`${s.len} ${C - s.len}`}
            strokeDashoffset={s.off}
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: 'stroke-dasharray 0.4s' }}
          />
        ))}
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize="14" fontWeight="600" fill="var(--ink)" style={{ textTransform: 'capitalize' }}>{top.label}</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fontSize="11" fill="var(--muted)">{(top.pct * 100).toFixed(0)}% of session</text>
      </svg>
      <div style={{ flex: 1, minWidth: 160 }}>
        {sorted.map(s => (
          <div key={s.label} style={{ display: 'grid', gridTemplateColumns: '14px 1fr auto', gap: 8, alignItems: 'center', fontSize: 13, marginBottom: 6 }}>
            <span style={{ width: 10, height: 10, background: s.color, borderRadius: 2 }} />
            <span style={{ textTransform: 'capitalize', color: 'var(--ink-2)' }}>{s.label}</span>
            <span style={{ color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>{(s.pct * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
