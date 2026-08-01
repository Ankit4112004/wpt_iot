import WhatIfPanel from "../components/WhatIfPanel.jsx";

export default function WhatIf() {
  return (
    <div className="space-y-8 pb-16 mx-auto max-w-5xl">
      <header className="border-b pb-6">
        <h1 className="text-2xl font-bold tracking-tight text-foreground">What-if Predictor</h1>
        <p className="mt-2 text-muted-foreground">
          Feed your own inputs to the live ML models and see the predictions instantly.
        </p>
      </header>

      <div>
        <WhatIfPanel />
      </div>

      <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
        <div className="p-4 border-b bg-muted/40">
          <h3 className="font-semibold text-sm">How it works</h3>
        </div>
        <div className="p-5">
          <p className="text-sm text-muted-foreground leading-relaxed">
            The sliders set per-cell electrical inputs (the range the models were trained on).
            Each change calls the backend's <code className="bg-secondary text-foreground px-1 py-0.5 rounded text-[13px] border">/api/whatif</code> endpoint,
            which runs all the models on your inputs and returns the predictions in real time.
          </p>
          <ul className="mt-4 space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-3"><span className="text-muted-foreground mt-0.5">•</span><div><b className="text-foreground font-medium">Predicted temperature</b> comes from the soft-sensor — drag the discharge current up and watch it rise (more current → more heating).</div></li>
            <li className="flex gap-3"><span className="text-muted-foreground mt-0.5">•</span><div><b className="text-foreground font-medium">± confidence</b> is the spread of predictions across the Random Forest's individual trees: a tight band means the trees agree.</div></li>
            {/* --- COMMENTED OUT — Health & RUL bullet (downgraded) --- */}
            {/* <li className="flex gap-3"><span className="text-muted-foreground mt-0.5">•</span><div><b className="text-foreground font-medium">Health & RUL</b> come from the per-cycle models based on the characteristics of the simulated cycle.</div></li> */}
          </ul>
        </div>
      </div>
    </div>
  );
}
