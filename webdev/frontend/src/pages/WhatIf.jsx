import WhatIfPanel from "../components/WhatIfPanel.jsx";

export default function WhatIf() {
  return (
    <div className="space-y-8 pb-16 max-w-4xl">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">What-if Predictor</h1>
        <p className="text-sm text-muted-foreground">
          Feed your own inputs to the live ML models and see the predictions instantly.
        </p>
      </header>

      <WhatIfPanel />

      <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6 text-sm leading-relaxed">
        <h3 className="font-semibold mb-2">How this works</h3>
        <p className="text-muted-foreground">
          The sliders set per-cell electrical inputs (the range the models were trained on).
          Each change calls the backend's <code className="text-foreground">/api/whatif</code> endpoint,
          which runs all the models on your inputs and returns the predictions in real time.
        </p>
        <ul className="mt-3 space-y-1 text-muted-foreground list-disc list-inside">
          <li><b className="text-foreground">Predicted temperature</b> comes from the soft-sensor — drag the discharge current up and watch it rise (more current → more heating).</li>
          <li><b className="text-foreground">± confidence</b> is the spread of predictions across the Random Forest's individual trees: a tight band means the trees agree.</li>
          <li><b className="text-foreground">Health</b> and <b className="text-foreground">RUL</b> come from the per-cycle models.</li>
        </ul>
      </div>
    </div>
  );
}
