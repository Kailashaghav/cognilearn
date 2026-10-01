import { useEffect, useRef, useState } from "react";
import { createLandmarker, extractFeatures } from "./face.js";
import { analyzeFace, analyzeText } from "./api.js";

const SESSION_ID = "session-" + Math.random().toString(36).slice(2, 8);
const SEND_EVERY_MS = 200; // 5 Hz
const MAX_POINTS = 150;

function Timeline({ points }) {
  const w = 600, h = 90;
  if (points.length < 2) return <div className="empty">Start the camera to see your focus timeline.</div>;
  const step = w / (MAX_POINTS - 1);
  const d = points.map((p, i) => `${i ? "L" : "M"}${(i * step).toFixed(1)},${(h - p.attention * h).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="timeline" role="img" aria-label="Attention over time">
      <line x1="0" x2={w} y1={h * 0.3} y2={h * 0.3} className="threshold" />
      <path d={d} className="line" />
    </svg>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(0);
  const lastSentRef = useRef(0);
  const [status, setStatus] = useState("idle"); // idle | loading | running | error
  const [error, setError] = useState("");
  const [reading, setReading] = useState(null);
  const [points, setPoints] = useState([]);
  const [text, setText] = useState("");
  const [textResult, setTextResult] = useState(null);

  async function start() {
    setStatus("loading");
    setError("");
    try {
      if (!landmarkerRef.current) landmarkerRef.current = await createLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("running");
      loop();
    } catch (e) {
      setError(e.message || "Could not start the camera.");
      setStatus("error");
    }
  }

  function stop() {
    cancelAnimationFrame(rafRef.current);
    videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
    setStatus("idle");
  }

  function loop() {
    const v = videoRef.current;
    const now = performance.now();
    if (v && v.readyState >= 2 && now - lastSentRef.current >= SEND_EVERY_MS) {
      lastSentRef.current = now;
      const features = extractFeatures(landmarkerRef.current.detectForVideo(v, now));
      analyzeFace({ session_id: SESSION_ID, ...features })
        .then((res) => {
          setReading(res);
          setPoints((p) => [...p.slice(-(MAX_POINTS - 1)), res]);
        })
        .catch((e) => setError(e.message));
    }
    rafRef.current = requestAnimationFrame(loop);
  }

  useEffect(() => () => stop(), []);

  async function sendText(e) {
    e.preventDefault();
    if (!text.trim()) return;
    try {
      setTextResult(await analyzeText(text));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main>
      <header>
        <h1>CogniLearn</h1>
        <p>Your study companion notices when you lose focus or get stuck. Video stays in this browser; only face numbers are sent.</p>
      </header>

      <section className="grid">
        <div className="panel camera">
          <video ref={videoRef} playsInline muted className={status === "running" ? "on" : ""} />
          {status !== "running" && <div className="placeholder">Camera is off</div>}
          <div className="actions">
            {status === "running" ? (
              <button onClick={stop} className="secondary">Stop camera</button>
            ) : (
              <button onClick={start} disabled={status === "loading"}>
                {status === "loading" ? "Loading face model…" : "Start camera"}
              </button>
            )}
          </div>
          {error && <p className="error" role="alert">{error} Check that the backend is running on port 8000.</p>}
        </div>

        <div className="panel readout">
          <div className="attention">
            <span className="big">{reading ? Math.round(reading.attention * 100) : "--"}</span>
            <span className="unit">% attention</span>
          </div>
          <div className={`emotion ${reading?.emotion ?? ""}`}>{reading ? reading.emotion : "waiting for camera"}</div>
          {reading && (
            <ul className="reasons">
              {reading.reasons.map((r) => <li key={r}>{r}</li>)}
            </ul>
          )}
          <Timeline points={points} />
        </div>
      </section>

      <section className="panel">
        <h2>Tell it how the lesson is going</h2>
        <form onSubmit={sendText} className="textform">
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. I'm lost on gradient descent" aria-label="How is the lesson going?" />
          <button type="submit">Check tone</button>
        </form>
        {textResult && (
          <p className="tone">
            Tone: <strong>{textResult.label}</strong> ({textResult.sentiment}) via {textResult.engine}
          </p>
        )}
      </section>
    </main>
  );
}
