const STACK = [
  ["Frontend", "React + Vite + Tailwind", "Dashboard, charts (Recharts), what-if panel"],
  ["Backend", "FastAPI (Python)", "REST API + ingestion worker + auto /docs"],
  ["Database", "PostgreSQL (Neon)", "Source of truth: readings, predictions, alerts"],
  ["ML", "scikit-learn", "4 models trained on real NASA battery data"],
  ["Data source", "ThingSpeak / MATLAB", "Live telemetry (optional) + recorded replay"],
  ["Hosting", "Vercel · Render · Neon", "Frontend · backend · database"],
];

export default function Architecture() {
  return (
    <div className="space-y-8 pb-16 max-w-4xl">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Architecture</h1>
        <p className="text-sm text-muted-foreground">How data flows from the battery to the dashboard.</p>
      </header>

      <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6">
        <h3 className="font-semibold mb-4 text-sm">Data flow</h3>
        <pre className="text-xs leading-relaxed text-muted-foreground overflow-x-auto">
{`ThingSpeak (optional) ──┐
                        ▼
        ingestion worker (every 3s)
        auto-picks: LIVE → REPLAY → last-known
                        │  runs 4 ML models on each reading
                        ▼
   PostgreSQL  ◄── readings + predictions + alerts
        ▲
        │  REST
   FastAPI  ──►  React dashboard (reads only the backend)`}
        </pre>
      </div>

      <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6">
        <h3 className="font-semibold mb-4 text-sm">Design decisions</h3>
        <ul className="space-y-2 text-sm text-muted-foreground list-disc list-inside">
          <li><b className="text-foreground">Backend as source of truth:</b> the frontend reads only the database, never ThingSpeak directly — so an upstream outage degrades gracefully to last-known-value.</li>
          <li><b className="text-foreground">Auto live↔replay:</b> if no fresh live data, the worker streams recorded cycles, so the demo always animates.</li>
          <li><b className="text-foreground">Right-sized worker:</b> a simple scheduled in-process task, not a message queue — correct for one low-rate source.</li>
          <li><b className="text-foreground">Honest ML:</b> every metric is held-out / cross-validated; no data leakage.</li>
        </ul>
      </div>

      <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6">
        <h3 className="font-semibold mb-4 text-sm">Tech stack</h3>
        <div className="space-y-2">
          {STACK.map(([layer, tech, role]) => (
            <div key={layer} className="grid grid-cols-3 gap-3 text-sm py-2 border-b border-border last:border-0">
              <div className="font-medium">{layer}</div>
              <div className="text-foreground">{tech}</div>
              <div className="text-muted-foreground text-xs sm:text-sm">{role}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
