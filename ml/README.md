# ML — Battery Intelligence Models

Three machine-learning models for the EV WPT monitor, trained on the **real NASA Li-ion
Battery Aging dataset** (measured discharge cycles for cells B0005/B0006/B0007/B0018).

One model of each main ML type, so the project demonstrates breadth without complexity:

| Model | Type | Predicts | Held-out score |
|-------|------|----------|----------------|
| Temperature soft-sensor | Supervised **regression** | Battery temperature (°C) from voltage/current/power | **R² 0.98, MAE 0.41 °C** |
| Battery-health classifier | Supervised **classification** | Healthy vs Degraded (per discharge cycle) | **89% acc, 0.88 F1** |
| Anomaly detector | **Unsupervised** | Abnormal operating points | flags ~2% outliers |

Scores are honest: the regressor is tested on a held-out split, and the classifier uses
**GroupKFold by battery cell** — it's scored on cells it never trained on, so the number
reflects real generalisation, not memorisation.

## Run

```bash
pip install -r requirements.txt
python get_data.py     # downloads the raw dataset (~21 MB)
python train.py        # trains 3 models -> models/*.pkl
python evaluate.py     # writes reports/metrics.json + plots
```

## Files
- `get_data.py` — downloads the raw NASA dataset.
- `features.py` — data loading + feature engineering (one source of truth).
- `train.py` — trains and saves the 3 models.
- `evaluate.py` — honest metrics + confusion-matrix / feature-importance plots.
- `predict.py` — loads the models and exposes simple `predict_*` functions for the backend.

## The USP (why this design)
The real charging circuit has **no temperature sensor** — it only measures voltage, current,
power and SOC. So instead of faking temperature, the **soft-sensor infers it from the
electrical signals**. That turns a hardware limitation into the project's headline feature,
and feeds the safety alerts (over-temp) and the health/anomaly views on the dashboard.
