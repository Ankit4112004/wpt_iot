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
    <div className="space-y-10 pb-16 max-w-4xl">
      <header>
        <h1 className="text-3xl font-bold tracking-tight text-foreground bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-teal-400">Architecture</h1>
        <p className="mt-2 text-muted-foreground text-lg">How data flows from the battery to the dashboard.</p>
      </header>

      <div className="rounded-2xl border border-white/5 bg-card text-card-foreground shadow-lg overflow-hidden">
        <div className="p-6 border-b border-border/50 bg-secondary/20">
          <h3 className="font-semibold text-base flex items-center gap-2">
             <span className="w-2 h-2 rounded-full bg-blue-500"></span>
             Data Flow
          </h3>
        </div>
        <div className="p-6 bg-[#0a0a0a]">
          <pre className="text-[13px] leading-relaxed text-blue-300/80 overflow-x-auto font-mono">
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
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-white/5 bg-card text-card-foreground shadow-lg p-6 hover:shadow-xl transition-shadow">
          <h3 className="font-semibold mb-5 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-violet-500"></span>
            Design Decisions
          </h3>
          <ul className="space-y-4 text-sm text-muted-foreground">
            <li className="flex gap-3 items-start"><span className="text-violet-500 mt-1">●</span><div><b className="text-foreground">Backend as source of truth:</b> the frontend reads only the database, never ThingSpeak directly — so an upstream outage degrades gracefully to last-known-value.</div></li>
            <li className="flex gap-3 items-start"><span className="text-violet-500 mt-1">●</span><div><b className="text-foreground">Auto live↔replay:</b> if no fresh live data, the worker streams recorded cycles, so the demo always animates.</div></li>
            <li className="flex gap-3 items-start"><span className="text-violet-500 mt-1">●</span><div><b className="text-foreground">Right-sized worker:</b> a simple scheduled in-process task, not a message queue — correct for one low-rate source.</div></li>
            <li className="flex gap-3 items-start"><span className="text-violet-500 mt-1">●</span><div><b className="text-foreground">Honest ML:</b> every metric is held-out / cross-validated; no data leakage.</div></li>
          </ul>
        </div>

        <div className="rounded-2xl border border-white/5 bg-card text-card-foreground shadow-lg p-6 hover:shadow-xl transition-shadow">
          <h3 className="font-semibold mb-5 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            Tech Stack
          </h3>
          <div className="space-y-0 text-sm border-l-2 border-emerald-500/20 ml-2 pl-4">
            {STACK.map(([layer, tech, role]) => (
              <div key={layer} className="py-3 border-b border-border/50 last:border-0 relative">
                <div className="absolute -left-[21px] top-4 w-2 h-2 rounded-full bg-emerald-500/50"></div>
                <div className="font-medium text-foreground mb-0.5">{layer} <span className="text-muted-foreground font-normal mx-1">—</span> <span className="text-emerald-400/90">{tech}</span></div>
                <div className="text-muted-foreground text-xs leading-snug">{role}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
