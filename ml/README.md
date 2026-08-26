# Production machine-learning models

This directory contains the two machine-learning models used by the EV WPT monitor. Both models are trained from the NASA Li-ion Battery Aging dataset and consume instantaneous electrical measurements.

| Model | Type | Predicts | Runtime use |
|---|---|---|---|
| Temperature soft-sensor | Supervised regression | Battery temperature from voltage, current, power, and time | Feeds the temperature gauge and over-temperature alerts |
| Anomaly detector | Unsupervised Isolation Forest | Whether an operating point is abnormal | Feeds anomaly status and alerts |

The temperature model is a virtual sensor for installations where the charging circuit does not expose a temperature measurement. The anomaly model receives the predicted temperature together with voltage, current, and power, allowing the full live inference chain to operate from electrical signals.

## Run locally

```bash
pip install -r requirements.txt
python get_data.py     # downloads the raw dataset
python train.py        # trains the two models
```

The raw dataset is downloaded into `ml/data/raw/` and is ignored by Git. Trained model files are written to `ml/models/`.

## Files

| File | Purpose |
|---|---|
| `get_data.py` | Downloads the NASA battery dataset required for training |
| `features.py` | Loads raw measurements and builds instantaneous model features |
| `train.py` | Trains and saves the temperature and anomaly models |
| `predict.py` | Loads the models and exposes runtime prediction functions |
| `models/metadata.json` | Records the active model feature sets and dataset name |
