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
    <div className="group relative rounded-2xl border border-white/5 bg-card text-card-foreground shadow-lg overflow-hidden transition-all duration-500 hover:shadow-2xl hover:-translate-y-1.5" style={{ borderTop: `4px solid ${m.accent}` }}>
      <div 
        className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500 pointer-events-none"
        style={{ background: `radial-gradient(circle at top right, ${m.accent}, transparent 70%)` }}
      />
      <div className="p-6">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h3 className="font-semibold tracking-tight text-lg group-hover:text-foreground transition-colors">{m.name}</h3>
            <p className="text-xs text-muted-foreground mt-1 tracking-wide uppercase">{m.type}</p>
          </div>
          <span 
            className="text-xs font-semibold px-3 py-1.5 rounded-full border shadow-sm" 
            style={{ color: m.accent, borderColor: `${m.accent}40`, backgroundColor: `${m.accent}10` }}
          >
            {m.metric}
          </span>
        </div>
        <dl className="mt-6 space-y-4 text-sm leading-relaxed relative z-10">
          <div className="p-3 rounded-lg bg-secondary/30 border border-border/50">
            <dt className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1.5">What it does</dt>
            <dd className="text-foreground/90">{m.what}</dd>
          </div>
          <div className="px-1">
            <dt className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">Why & How</dt>
            <dd className="text-muted-foreground">{m.why}</dd>
          </div>
          <div className="px-1">
            <dt className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">Used for</dt>
            <dd className="text-muted-foreground">{m.use}</dd>
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
