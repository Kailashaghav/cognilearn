const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeFace(payload) {
  const r = await fetch(`${BASE}/api/perception/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`Backend returned ${r.status}`);
  return r.json();
}

export async function analyzeText(text) {
  const r = await fetch(`${BASE}/api/perception/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`Backend returned ${r.status}`);
  return r.json();
}
