# Frontend — React + Vite Dashboard

The monitoring dashboard. It reads **only** from the backend API (never ThingSpeak
directly), polls every 3 seconds, and shows live battery intelligence.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173  (proxies /api to backend on :8000)
```
Start the backend first (see `../backend/README.md`).

## What it shows
- **Metric cards** — Voltage, Current, Power, State of Charge
- **Temperature gauge** — the ML soft-sensor's predicted battery temperature, colour-coded
  green→amber→red against the over-temp limit
- **Safety alerts** — over-temp / anomaly events from the backend
- **Status chips** — battery health (from the classifier), anomaly status, active alerts
- **Live charts** — predicted temperature, power, SOC, voltage over time
- **"Last updated X ago"** — single freshness indicator (live vs replay is decided
  automatically by the backend and is invisible here, by design)

## Files
- `src/api.js` — the only place that calls the backend.
- `src/App.jsx` — polls the API every 3s and lays out the dashboard.
- `src/components/` — `MetricCard`, `TempGauge`, `AlertsPanel`, `TimeChart`.
- `vite.config.js` — dev proxy to the backend.

## Production
Set `VITE_API_BASE` (in `.env`) to the deployed backend URL, then `npm run build`.
Deploy the `dist/` folder to Vercel.
