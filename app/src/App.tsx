import { useEffect, useRef, useState } from 'react'
import { loadModels, detect, EXPR_LABELS } from './inference'
import { Tracker, type Track } from './tracker'
import { Timeline, type TimelinePoint } from './Timeline'
import { Donut } from './Donut'

type Status = 'idle' | 'loading' | 'live' | 'paused' | 'error'

const TIMELINE_WINDOW_MS = 30_000
const GAME_HOLD_MS = 600
const GAME_PROB_THRESHOLD = 0.5

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trackerRef = useRef(new Tracker())
  const lastImgRef = useRef<HTMLImageElement | null>(null)

  const rafRef = useRef<number | null>(null)
  const lastInfRef = useRef(0)
  const fpsRef = useRef({ t: performance.now(), n: 0 })
  const totalsRef = useRef<number[]>(new Array(7).fill(0))
  const lastFrameRef = useRef(performance.now())
  const sessionStartRef = useRef(0)
  const timelineRef = useRef<TimelinePoint[]>([])

  // game state refs (avoid re-render on each frame)
  const gameRef = useRef<{ targetIdx: number; startedAt: number; matchedSince: number | null; finishedMs: number | null } | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('Click "Start camera" to begin.')
  const [tracks, setTracks] = useState<Track[]>([])
  const [fps, setFps] = useState(0)
  const [latency, setLatency] = useState(0)
  const [, forceTick] = useState(0)

  // settings
  const [confidence, setConfidence] = useState(0.4)
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([])
  const [selectedCam, setSelectedCam] = useState<string>('')
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')

  // session summary
  const [summary, setSummary] = useState<{ totals: number[]; durationMs: number } | null>(null)

  // game
  const [gameOn, setGameOn] = useState(false)
  const [gameTarget, setGameTarget] = useState<number | null>(null)
  const [gameElapsed, setGameElapsed] = useState(0)
  const [gameResult, setGameResult] = useState<{ idx: number; ms: number } | null>(null)

  useEffect(() => () => stopAll(), [])

  useEffect(() => {
    // Enumerate cameras when any are available
    if (status !== 'live') return
    navigator.mediaDevices.enumerateDevices().then(devs => {
      setCameras(devs.filter(d => d.kind === 'videoinput'))
    }).catch(() => {})
  }, [status])

  function stopAll() {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    const v = videoRef.current
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      v.srcObject = null
    }
  }

  async function start(deviceId?: string) {
    try {
      setStatus('loading')
      setStatusMsg('Loading models…')
      await loadModels(m => setStatusMsg(m))

      setStatusMsg('Requesting camera…')
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, frameRate: { ideal: 30, max: 30 }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
        audio: false,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      const v = videoRef.current!
      v.muted = true
      v.playsInline = true
      v.srcObject = stream
      await new Promise<void>(res => {
        if (v.readyState >= 1) return res()
        const onmd = () => { v.removeEventListener('loadedmetadata', onmd); res() }
        v.addEventListener('loadedmetadata', onmd)
      })
      try { await v.play() } catch (err) { console.warn('video.play() rejected:', err) }

      const c = canvasRef.current!
      c.width = v.videoWidth
      c.height = v.videoHeight
      lastImgRef.current = null
      trackerRef.current.reset()
      totalsRef.current = new Array(7).fill(0)
      timelineRef.current = []
      sessionStartRef.current = performance.now()
      lastFrameRef.current = performance.now()
      setSummary(null)

      setStatus('live')
      setStatusMsg('Live')
      runLoop()
    } catch (e) {
      console.error('start() failed:', e)
      setStatus('error')
      const name = (e as { name?: string })?.name
      const msg = e instanceof Error ? e.message : String(e)
      let friendly = msg || 'Unknown error'
      if (name === 'NotAllowedError') friendly = 'Camera permission denied. Click the lock icon in the address bar → allow camera, then retry.'
      else if (name === 'NotFoundError') friendly = 'No camera detected on this device.'
      else if (name === 'NotReadableError') friendly = 'Camera is already in use by another app.'
      else if (name === 'OverconstrainedError') friendly = 'Camera does not support 1280x720@30fps.'
      setStatusMsg(friendly)
    }
  }

  function stop() {
    stopAll()
    setStatus('paused')
    setStatusMsg('Stopped')
    const c = canvasRef.current
    if (c) c.getContext('2d')?.clearRect(0, 0, c.width, c.height)
    const totals = totalsRef.current
    const total = totals.reduce((a, b) => a + b, 0)
    if (total > 1000) setSummary({ totals: [...totals], durationMs: total })
    setTracks([])
    setGameOn(false)
    setGameTarget(null)
    gameRef.current = null
  }

  function switchCamera(deviceId: string) {
    setSelectedCam(deviceId)
    if (status === 'live') {
      stopAll()
      start(deviceId)
    }
  }

  function flipCamera() {
    const next: 'user' | 'environment' = facingMode === 'user' ? 'environment' : 'user'
    setFacingMode(next)
    if (status === 'live') {
      stopAll()
      // start() reads facingMode from state, but we just set it — need to use the new value
      // Use a small timeout so React state updates first, OR pass the new value through.
      ;(async () => {
        try {
          setStatus('loading')
          setStatusMsg('Switching camera…')
          const constraints: MediaStreamConstraints = {
            video: { facingMode: { ideal: next }, frameRate: { ideal: 30, max: 30 }, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
          }
          const stream = await navigator.mediaDevices.getUserMedia(constraints)
          const v = videoRef.current!
          v.muted = true
          v.playsInline = true
          v.srcObject = stream
          await new Promise<void>(res => {
            if (v.readyState >= 1) return res()
            const onmd = () => { v.removeEventListener('loadedmetadata', onmd); res() }
            v.addEventListener('loadedmetadata', onmd)
          })
          try { await v.play() } catch (err) { console.warn('video.play() rejected:', err) }
          const c = canvasRef.current!
          c.width = v.videoWidth
          c.height = v.videoHeight
          trackerRef.current.reset()
          setStatus('live')
          setStatusMsg('Live')
          runLoop()
        } catch (e) {
          console.error('flip failed', e)
          setStatus('error')
          setStatusMsg(`Could not switch to ${next === 'user' ? 'front' : 'back'} camera`)
        }
      })()
    }
  }

  function runLoop() {
    let busy = false
    let busyStartedAt = 0
    // Watchdog: if a single inference pass takes longer than this, assume the
    // backend has stalled and clear `busy` so subsequent frames can retry. Without
    // this the page would freeze until manual refresh — exactly the laptop bug.
    const STALL_LIMIT_MS = 9000
    const tick = async () => {
      const v = videoRef.current
      if (!v || v.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }
      const now = performance.now()
      const f = fpsRef.current
      f.n++
      if (now - f.t >= 1000) {
        setFps(Math.round((f.n * 1000) / (now - f.t)))
        f.n = 0; f.t = now
      }
      if (busy && now - busyStartedAt > STALL_LIMIT_MS) {
        console.warn(`inference stalled >${STALL_LIMIT_MS}ms; releasing watchdog`)
        busy = false
      }
      if (!busy && now - lastInfRef.current >= 33) {
        busy = true
        busyStartedAt = now
        const t0 = performance.now()
        try {
          const dets = await detect(v, v.videoWidth, v.videoHeight, { minScore: confidence })
          const tr = trackerRef.current.step(dets)
          // Update timeline & totals using dominant face
          const dominant = trackerRef.current.dominant()
          const dt = now - lastFrameRef.current
          lastFrameRef.current = now
          if (dominant) {
            totalsRef.current[dominant.topIdx] += dt
            timelineRef.current.push({ t: now, topIdx: dominant.topIdx, topProb: dominant.smoothScores[dominant.topIdx] })
            const cutoff = now - TIMELINE_WINDOW_MS - 1000
            while (timelineRef.current.length > 0 && timelineRef.current[0].t < cutoff) timelineRef.current.shift()
          }
          // Game logic
          if (gameRef.current && dominant) {
            const g = gameRef.current
            const matched = dominant.topIdx === g.targetIdx && dominant.smoothScores[g.targetIdx] >= GAME_PROB_THRESHOLD
            if (matched) {
              if (g.matchedSince == null) g.matchedSince = now
              if (now - g.matchedSince >= GAME_HOLD_MS && g.finishedMs == null) {
                g.finishedMs = now - g.startedAt
                setGameResult({ idx: g.targetIdx, ms: Math.round(g.finishedMs) })
                setGameOn(false)
                setGameTarget(null)
                gameRef.current = null
              }
            } else {
              g.matchedSince = null
            }
            setGameElapsed(Math.round(now - g.startedAt))
          }
          setTracks(tr)
          drawOverlay(tr, v.videoWidth, v.videoHeight)
          setLatency(Math.round(performance.now() - t0))
          forceTick(x => (x + 1) % 1024) // re-render for timeline
        } catch (e) {
          console.error(e)
        } finally {
          lastInfRef.current = performance.now()
          busy = false
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  function drawOverlay(tracks: Track[], w: number, h: number, redrawImg = false) {
    const c = canvasRef.current
    if (!c) return
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    if (redrawImg && lastImgRef.current) ctx.drawImage(lastImgRef.current, 0, 0)
    const lw = Math.max(2, c.width / 400)
    const fontSize = Math.max(13, Math.round(c.width / 55))
    ctx.lineWidth = lw
    ctx.font = `500 ${fontSize}px 'Inter', sans-serif`
    ctx.textBaseline = 'bottom'
    for (const t of tracks) {
      const { x1, y1, x2, y2 } = t.box
      ctx.strokeStyle = t.color
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const label = `#${t.id} ${EXPR_LABELS[t.topIdx]} ${(t.smoothScores[t.topIdx] * 100).toFixed(0)}%`
      const padX = 7, padY = 4
      const tw = ctx.measureText(label).width + padX * 2
      const th = fontSize + padY * 2
      ctx.fillStyle = t.color
      ctx.fillRect(x1, Math.max(0, y1 - th), tw, th)
      ctx.fillStyle = '#0a0a0a'
      ctx.fillText(label, x1 + padX, Math.max(th, y1) - padY)
    }
  }

  async function processBlob(f: Blob) {
    try {
      setStatus('loading')
      setStatusMsg('Loading models…')
      await loadModels(m => setStatusMsg(m))
      stopAll()
      const url = URL.createObjectURL(f)
      const img = new Image()
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url })
      lastImgRef.current = img
      const c = canvasRef.current!
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)

      trackerRef.current.reset()
      setStatusMsg('Detecting…')
      const t0 = performance.now()
      const dets = await detect(img, img.naturalWidth, img.naturalHeight, { minScore: confidence })
      const tr = trackerRef.current.step(dets)
      drawOverlay(tr, img.naturalWidth, img.naturalHeight, true)
      setTracks(tr)
      setLatency(Math.round(performance.now() - t0))
      setFps(0)
      setStatus('paused')
      setStatusMsg(tr.length ? `Found ${tr.length} face${tr.length > 1 ? 's' : ''}` : 'No face detected')
    } catch (err) {
      console.error(err)
      setStatus('error')
      setStatusMsg(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    try { await processBlob(f) } finally { e.target.value = '' }
  }

  // Ctrl/Cmd+V paste an image from clipboard. Skips when user is typing in an input.
  useEffect(() => {
    const onPaste = (ev: ClipboardEvent) => {
      const t = ev.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      const items = ev.clipboardData?.items
      if (!items) return
      for (const it of items) {
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          const blob = it.getAsFile()
          if (blob) {
            ev.preventDefault()
            processBlob(blob)
            return
          }
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confidence])

  function startGame() {
    // pick a target excluding neutral (boring) and disgust (model rarely fires it accurately)
    const choices = [0, 2, 3, 5, 6] // angry, fear, happy, sad, surprise
    const targetIdx = choices[Math.floor(Math.random() * choices.length)]
    setGameTarget(targetIdx)
    setGameElapsed(0)
    setGameResult(null)
    setGameOn(true)
    gameRef.current = { targetIdx, startedAt: performance.now(), matchedSince: null, finishedMs: null }
  }
  function cancelGame() {
    setGameOn(false)
    setGameTarget(null)
    gameRef.current = null
  }

  const dominant = tracks.length ? trackerRef.current.dominant() : null
  const top = dominant
    ? { idx: dominant.topIdx, scores: dominant.smoothScores }
    : null

  return (
    <div className="shell">
      <div className="topbar">
        <div className="brand"><span className="dot" />expression.local</div>
        <nav>
          <a href="https://github.com/Rieltzx25/expression-detection-resnet18" target="_blank" rel="noreferrer">model</a>
          <a href="#about">about</a>
        </nav>
      </div>

      <h1>Real-time facial expression recognition</h1>
      <p className="lede">
        Multi-face detection (YOLOv11n-face + WIDERFACE) cropped and classified by a ResNet18
        fine-tuned on FER2013. Everything runs locally in your browser via ONNX Runtime Web —
        no frames ever leave your device.
      </p>

      <div className="grid">
        <div className="card">
          <div className="stage">
            <video ref={videoRef} autoPlay playsInline muted style={{ display: status === 'live' ? 'block' : 'none' }} />
            <canvas ref={canvasRef} />
            {status !== 'live' && status !== 'paused' && (
              <div className="empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="3" y="6" width="14" height="12" rx="2" />
                  <path d="M17 10l4-2v8l-4-2" />
                </svg>
                <span>{status === 'loading' ? statusMsg : 'Camera off'}</span>
              </div>
            )}
            {gameOn && gameTarget != null && (
              <div className="game-banner">
                <div className="game-instruction">Show me <strong>{EXPR_LABELS[gameTarget]}</strong> 😶</div>
                <div className="game-timer">{(gameElapsed / 1000).toFixed(1)}s</div>
                <button className="btn-mini" onClick={cancelGame}>cancel</button>
              </div>
            )}
            {gameResult && !gameOn && (
              <div className="game-result">
                ✓ Got <strong>{EXPR_LABELS[gameResult.idx]}</strong> in <strong>{(gameResult.ms / 1000).toFixed(2)}s</strong>
                <button className="btn-mini" onClick={() => setGameResult(null)}>×</button>
              </div>
            )}
          </div>
          <div className="toolbar">
            {status !== 'live' ? (
              <button className="btn primary" onClick={() => start(selectedCam || undefined)} disabled={status === 'loading'}>
                {status === 'loading' ? 'Loading…' : 'Start camera'}
              </button>
            ) : (
              <button className="btn" onClick={stop}>Stop</button>
            )}
            <label className="btn" style={{ cursor: 'pointer' }}>
              Upload image
              <input type="file" accept="image/*" onChange={handleFile} />
            </label>
            {status === 'live' && (
              <button className="btn icon-btn" onClick={flipCamera} title={`Switch to ${facingMode === 'user' ? 'back' : 'front'} camera`} aria-label="Flip camera">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 7h3l2-3h8l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z" />
                  <path d="M9 13a3 3 0 1 0 6 0 3 3 0 1 0-6 0" />
                  <path d="M15 10l1.5-1.5M9 15l-1.5 1.5" />
                </svg>
                <span className="hide-on-mobile">Flip</span>
              </button>
            )}
            {status === 'live' && !gameOn && (
              <button className="btn" onClick={startGame}>🎯 Mirror this</button>
            )}
            <div style={{ flex: 1 }} />
            <span style={{ alignSelf: 'center', fontSize: 12, color: 'var(--muted)' }}>
              <span className={`status-dot ${status === 'live' ? 'live' : status === 'loading' ? 'loading' : status === 'error' ? 'error' : ''}`} />
              {statusMsg}
            </span>
          </div>
        </div>

        <aside className="side">
          <div className="card panel">
            <h3>Top prediction {dominant && <span style={{ color: 'var(--muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>· face #{dominant.id}</span>}</h3>
            {top && dominant ? (
              <>
                <div className="dominant">
                  <span className="label">{EXPR_LABELS[top.idx]}</span>
                  <span className="conf">{(top.scores[top.idx] * 100).toFixed(1)}%</span>
                </div>
                <div className="bars">
                  {Array.from(top.scores).map((s, i) => (
                    <div className={`bar ${i === top.idx ? 'top' : ''}`} key={i}>
                      <span className="name">{EXPR_LABELS[i]}</span>
                      <div className="track"><div className="fill" style={{ width: `${(s * 100).toFixed(2)}%` }} /></div>
                      <span className="pct">{(s * 100).toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-side">
                {status === 'live' ? 'Looking for a face…' : 'Start the camera or upload an image to see predictions.'}
              </div>
            )}
          </div>

          <div className="card panel">
            <h3>Timeline · last 30 s</h3>
            {timelineRef.current.length > 1 ? (
              <Timeline points={timelineRef.current} windowMs={TIMELINE_WINDOW_MS} height={70} />
            ) : (
              <div className="empty-side" style={{ padding: '8px 0' }}>Awaiting data…</div>
            )}
          </div>

          <div className="card panel">
            <h3>Performance</h3>
            <div className="meta">
              <div className="k">Display</div><div className="v">{fps} fps</div>
              <div className="k">Inference</div><div className="v">{latency} ms</div>
              <div className="k">Faces</div><div className="v">{tracks.length}</div>
              <div className="k">Backend</div><div className="v">WASM</div>
            </div>
          </div>
        </aside>
      </div>

      {tracks.length > 0 && (
        <section className="card panel" style={{ marginTop: 16 }}>
          <h3>Detected faces · {tracks.length}</h3>
          <div className="faces-grid">
            {tracks.map(t => (
              <FaceCard key={t.id} track={t} isDominant={dominant?.id === t.id} />
            ))}
          </div>
        </section>
      )}

      <section className="settings-row">
        <div className="card panel">
          <h3>Settings</h3>
          <div className="setting">
            <label htmlFor="conf">Confidence threshold <span className="mono">{confidence.toFixed(2)}</span></label>
            <input id="conf" type="range" min="0.2" max="0.9" step="0.05" value={confidence} onChange={e => setConfidence(parseFloat(e.target.value))} />
          </div>
          {cameras.length > 1 && (
            <div className="setting">
              <label htmlFor="cam">Camera</label>
              <select id="cam" value={selectedCam} onChange={e => switchCamera(e.target.value)}>
                <option value="">Default</option>
                {cameras.map(c => <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${c.deviceId.slice(0, 6)}`}</option>)}
              </select>
            </div>
          )}
        </div>

        {summary && (
          <div className="card panel">
            <h3>Session summary · {(summary.durationMs / 1000).toFixed(1)} s of dominant face</h3>
            <Donut totals={summary.totals} />
          </div>
        )}
      </section>

      <p className="privacy" id="about">
        Inference happens locally with ONNX Runtime Web. Faces are detected with YOLOv11n-face
        (WIDERFACE), tracked across frames with IoU matching, and emotion is classified by a
        ResNet18 trained on FER2013 — outputs are smoothed with an exponential moving average
        for stability. Nothing is uploaded.
      </p>

      <footer>
        <span>YOLOv11n-face · ResNet18 · FER2013 · ONNX FP32 · 55 MB total</span>
        <span>
          source: <a href="https://github.com/Rieltzx25/expression-detection-resnet18" target="_blank" rel="noreferrer">github</a>
        </span>
      </footer>
    </div>
  )
}

function FaceCard({ track, isDominant }: { track: Track; isDominant: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    c.width = 48; c.height = 48
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(48, 48)
    for (let i = 0; i < 48 * 48; i++) {
      const g = track.crop[i]
      img.data[i * 4] = g
      img.data[i * 4 + 1] = g
      img.data[i * 4 + 2] = g
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [track.crop])

  return (
    <div className="face-card" style={{ borderColor: track.color, ...(isDominant ? { boxShadow: `0 0 0 2px ${track.color}33` } : {}) }}>
      <canvas ref={canvasRef} className="face-crop" />
      <div className="face-meta">
        <div className="face-id">
          <span className="dot-color" style={{ background: track.color }} />
          face #{track.id}
          {isDominant && <span className="badge-dom">dominant</span>}
        </div>
        <div className="face-emotion">{EXPR_LABELS[track.topIdx]}</div>
        <div className="face-prob mono">{(track.smoothScores[track.topIdx] * 100).toFixed(0)}%</div>
      </div>
    </div>
  )
}
