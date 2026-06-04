// api.js — the only place the frontend talks to the backend.
// In dev, BASE is "" and Vite proxies /api to localhost:8000.
// In prod, set VITE_API_BASE to the deployed backend URL (e.g. the Render URL).
const BASE = import.meta.env.VITE_API_BASE || "";

async function getJSON(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export const api = {
  latest: () => getJSON("/api/readings/latest"),
  series: (n = 120) => getJSON(`/api/readings?limit=${n}`),
  alerts: () => getJSON("/api/alerts?limit=10"),
  summary: () => getJSON("/api/analytics/summary"),
  whatif: (voltage, current, time = 1800) =>
    getJSON(`/api/whatif?voltage=${voltage}&current=${current}&time_in_cycle=${time}`),
};
