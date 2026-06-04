// Interactive "what-if" panel: drag the sliders and the ML models predict live.
// Great for demos — "increase the discharge current and watch predicted temperature rise."
import { useEffect, useState } from "react";
import { api } from "../api";

function Slider({ label, value, min, max, step, onChange, suffix }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium text-foreground">{value}{suffix}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[#da7756] cursor-pointer"
      />
    </div>
  );
}

export default function WhatIfPanel() {
  // Per-cell inputs (the model's native range). Current is a discharge magnitude.
  const [voltage, setVoltage] = useState(3.7);
  const [currentMag, setCurrentMag] = useState(2.0);
  const [time, setTime] = useState(1800);
  const [res, setRes] = useState(null);

  // Debounced fetch so dragging doesn't spam the backend.
  useEffect(() => {
    const id = setTimeout(async () => {
      try {
        setRes(await api.whatif(voltage, -currentMag, time)); // discharge -> negative
      } catch {
        setRes(null);
      }
    }, 250);
    return () => clearTimeout(id);
  }, [voltage, currentMag, time]);

  const t = res?.predicted_temp;
  const tempColor = t == null ? "#a3a3a3" : t >= 38 ? "#ef4444" : t >= 33 ? "#f59e0b" : "#10b981";
  const healthy = res?.health_label !== "Degraded";

  return (
    <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6">
      <div className="flex items-center gap-2 mb-4">
        <h3 className="font-semibold leading-none tracking-tight text-sm">What-if Predictor</h3>
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider bg-zinc-800 text-zinc-300 border border-zinc-700">
          interactive ML
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* sliders */}
        <div className="space-y-4">
          <Slider label="Cell voltage" value={voltage} min={2.5} max={4.2} step={0.05}
                  onChange={setVoltage} suffix=" V" />
          <Slider label="Discharge current" value={currentMag} min={1} max={25} step={0.5}
                  onChange={setCurrentMag} suffix=" A" />
          <Slider label="Time on load" value={time} min={0} max={3600} step={60}
                  onChange={setTime} suffix=" s" />
        </div>

        {/* live result */}
        <div className="flex flex-col justify-center">
          <div className="text-xs text-muted-foreground mb-1">Predicted temperature</div>
          <div className="text-4xl font-semibold tracking-tight" style={{ color: tempColor }}>
            {t != null ? t.toFixed(1) : "—"}
            <span className="text-base ml-1 opacity-70">°C</span>
            {res?.temp_confidence != null && (
              <span className="text-sm text-muted-foreground ml-2">± {res.temp_confidence}</span>
            )}
          </div>
          <div className="flex flex-wrap gap-2 mt-3">
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${healthy ? "text-emerald-500 border-emerald-500/20" : "text-amber-500 border-amber-500/20"}`}>
              {res?.health_label ?? "—"}
            </span>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${res?.is_anomaly ? "text-amber-500 border-amber-500/20" : "text-emerald-500 border-emerald-500/20"}`}>
              {res?.is_anomaly ? "Anomaly" : "Normal"}
            </span>
          </div>
          <div className="text-sm mt-3">
            <span className="text-muted-foreground">Remaining useful life: </span>
            <b className="text-foreground">{res?.predicted_rul ?? "—"} cycles</b>
          </div>
          <div className="text-xs text-muted-foreground mt-2">
            ± is the spread across the forest's trees (model confidence)
          </div>
        </div>
      </div>
    </div>
  );
}
