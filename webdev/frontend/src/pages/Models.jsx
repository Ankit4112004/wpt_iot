// Static descriptions of the 4 ML models. Metrics match ml/reports/metrics.json
// (held-out / cross-validated — honest numbers).
const MODELS = [
  {
    name: "Temperature Soft-Sensor",
    type: "Supervised regression · Random Forest",
    accent: "#ef4444",
    metric: "R² 0.98 · ± 0.41 °C",
    what: "Estimates battery temperature from electrical signals (voltage, current, power, time).",
    why: "The real charging circuit has no temperature sensor. Instead of faking it, the model infers temperature from the data we do have — a 'virtual sensor'.",
    use: "Drives the temperature gauge and the over-temp safety alert on the dashboard.",
  },
  {
    name: "Battery-Health Classifier",
    type: "Supervised classification · Random Forest",
    accent: "#f59e0b",
    metric: "89% accuracy · 0.88 F1",
    what: "Classifies the battery as Healthy or Degraded from the shape of a discharge cycle.",
    why: "Capacity fades as a cell ages; the discharge curve reveals it. Validated with GroupKFold by cell, so it's scored on batteries it never trained on.",
    use: "Shows the health status chip and feeds the alerts/analytics views.",
  },
  {
    name: "Remaining Useful Life (RUL)",
    type: "Supervised regression · Random Forest",
    accent: "#10b981",
    metric: "R² 0.86 · ± 9.9 cycles",
    what: "Predicts how many charge cycles remain before end-of-life (capacity → 1.4 Ah).",
    why: "A classic battery-prognostics task. Reuses the same per-cycle features as the health model, so it needs no extra sensors.",
    use: "Shown as the RUL chip on the dashboard and in the What-if predictor.",
  },
  {
    name: "Anomaly Detector",
    type: "Unsupervised · Isolation Forest",
    accent: "#8b5cf6",
    metric: "flags ~2% outliers",
    what: "Flags abnormal operating points (unusual voltage/current/power/temperature combos).",
    why: "Unsupervised — it learns what 'normal' charging looks like and flags deviations, needing no labels. Useful for catching faults or foreign objects.",
    use: "Raises anomaly alerts and the anomaly status chip.",
  },
];

function ModelCard({ m }) {
  return (
    <div className="flex flex-col rounded-lg border bg-card text-card-foreground shadow-sm">
      <div className="p-5 border-b flex items-start justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: m.accent }} />
          <div>
            <h3 className="font-semibold text-base">{m.name}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{m.type}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium px-2.5 py-1 rounded bg-secondary text-secondary-foreground border">
            {m.metric}
          </span>
        </div>
      </div>
      <div className="p-5 flex-1">
        <dl className="space-y-4 text-sm">
          <div>
            <dt className="text-xs font-medium text-muted-foreground mb-1">Function</dt>
            <dd className="text-foreground leading-relaxed">{m.what}</dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-muted-foreground mb-1">Rationale</dt>
            <dd className="text-muted-foreground leading-relaxed">{m.why}</dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-muted-foreground mb-1">Application</dt>
            <dd className="text-muted-foreground leading-relaxed">{m.use}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}

export default function Models() {
  return (
    <div className="space-y-8 pb-16">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">ML Models</h1>
        <p className="text-sm text-muted-foreground">
          Four models across the three main ML types, trained on the real NASA Li-ion battery dataset.
        </p>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {MODELS.map((m) => <ModelCard key={m.name} m={m} />)}
      </div>
    </div>
  );
}
