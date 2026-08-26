# EV WPT Monitor frontend

The frontend is a Vite/React dashboard for the EV wireless power transfer monitor. It reads only from the FastAPI backend, polls every three seconds, and presents the active telemetry and model outputs.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
```

Start the backend first from `../backend/`. Vite proxies `/api` requests to `http://localhost:8000` during development.

## What it shows

| Feature | Description |
|---|---|
| Metric cards | Voltage, current, power, and state of charge |
| Temperature gauge | Predicted battery temperature from the ML soft-sensor, color-coded against the safety limit |
| Anomaly status | Current operating-point classification from the anomaly detector |
| Safety alerts | Over-temperature and anomaly events from the backend |
| Live charts | Predicted temperature, power, state of charge, and voltage over time |
| Freshness status | Whether the latest reading is live, replayed, or unavailable |
| Model page | Explanations of the temperature and anomaly models |
| Architecture page | Overview of the backend, database, ML, and frontend data flow |

## Files

| Path | Purpose |
|---|---|
| `src/api.js` | The frontend's backend API adapter |
| `src/App.jsx` | Hash routes for Dashboard, Models, and Architecture |
| `src/components/` | Reusable dashboard cards, charts, alerts, and temperature gauge |
| `src/pages/` | Dashboard, model, and architecture screens |
| `vite.config.js` | Local development proxy to the backend |

## Production

Set `VITE_API_BASE` to the deployed backend URL before building:

```text
VITE_API_BASE=https://your-render-service.onrender.com
```

Then run `npm run build` and deploy the generated `dist/` directory to Vercel.
