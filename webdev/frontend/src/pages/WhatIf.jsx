import WhatIfPanel from "../components/WhatIfPanel.jsx";

export default function WhatIf() {
  return (
    <div className="space-y-10 pb-16 max-w-4xl">
      <header>
        <h1 className="text-3xl font-bold tracking-tight text-foreground bg-clip-text text-transparent bg-gradient-to-r from-violet-500 to-fuchsia-500">What-if Predictor</h1>
        <p className="mt-2 text-lg text-muted-foreground">
          Feed your own inputs to the live ML models and see the predictions instantly.
        </p>
      </header>

      <div className="relative">
        <div className="absolute -inset-1 bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 rounded-[1.5rem] blur-xl opacity-50 pointer-events-none"></div>
        <div className="relative">
          <WhatIfPanel />
        </div>
      </div>

      <div className="rounded-2xl border border-white/5 bg-card text-card-foreground shadow-lg p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-fuchsia-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
          <span className="w-1.5 h-6 rounded-full bg-fuchsia-500 block"></span>
          How this works
        </h3>
        <p className="text-muted-foreground leading-relaxed">
          The sliders set per-cell electrical inputs (the range the models were trained on).
          Each change calls the backend's <code className="bg-secondary/50 text-foreground px-1.5 py-0.5 rounded text-[13px] border border-border/50">/api/whatif</code> endpoint,
          which runs all the models on your inputs and returns the predictions in real time.
        </p>
        <ul className="mt-6 space-y-3 text-sm text-muted-foreground">
          <li className="flex gap-3 items-start"><span className="text-fuchsia-500 mt-1">●</span><div><b className="text-foreground">Predicted temperature</b> comes from the soft-sensor — drag the discharge current up and watch it rise (more current → more heating).</div></li>
          <li className="flex gap-3 items-start"><span className="text-fuchsia-500 mt-1">●</span><div><b className="text-foreground">± confidence</b> is the spread of predictions across the Random Forest's individual trees: a tight band means the trees agree.</div></li>
          <li className="flex gap-3 items-start"><span className="text-fuchsia-500 mt-1">●</span><div><b className="text-foreground">Health</b> and <b className="text-foreground">RUL</b> come from the per-cycle models.</div></li>
        </ul>
      </div>
    </div>
  );
}
