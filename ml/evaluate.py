"""
evaluate.py — measure how good the models really are, on data they did NOT train on.

Run:  python ml/evaluate.py   (run train.py first)

Writes ml/reports/metrics.json plus PNG plots (confusion matrix, feature importance).
These are the numbers you put on the resume / talk about in an interview — and crucially
they are computed honestly (held-out data / cross-validation), not on the training set.

How to explain this file in an interview:
  - "Regression is scored with R2 and MAE on a held-out test split."
  - "The health classifier is scored with GroupKFold split BY battery cell — the model is
     tested on a cell it never saw, so the score reflects real generalisation, not memorising."
  - "I report a confusion matrix and precision/recall, not just accuracy."
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; we just save PNGs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict, GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

import features as F

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS, exist_ok=True)


def evaluate_temperature(df):
    """Hold out 20% of readings, score the temperature soft-sensor on them."""
    X, y = F.build_instant_table(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(
        n_estimators=40, max_depth=10, min_samples_leaf=30, random_state=42, n_jobs=-1
    ).fit(Xtr, ytr)
    pred = m.predict(Xte)
    res = {"r2": round(float(r2_score(yte, pred)), 4),
           "mae_celsius": round(float(mean_absolute_error(yte, pred)), 4)}

    # feature importance plot
    _bar_plot(F.INSTANT_FEATURES, m.feature_importances_,
              "Temperature model - feature importance",
              os.path.join(REPORTS, "temp_feature_importance.png"))
    return res


def evaluate_health(df):
    """
    Score the health classifier with GroupKFold by battery: each fold tests on a cell
    that was held out of training, so the score is honest cross-cell generalisation.
    """
    cyc = F.build_cycle_table(df)
    X, y, groups = cyc[F.CYCLE_FEATURES], cyc["degraded"], cyc["Battery"]
    gkf = GroupKFold(n_splits=4)
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=gkf, groups=groups)

    res = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
    }
    _confusion_plot(confusion_matrix(y, pred), ["Healthy", "Degraded"],
                    os.path.join(REPORTS, "health_confusion_matrix.png"))
    return res


def evaluate_anomaly(df):
    """Anomaly detection is unsupervised, so we report how much it flags (the rate)."""
    from sklearn.ensemble import IsolationForest
    feats = ["Voltage_measured", "Current_measured", "Power", "Temperature_measured"]
    m = IsolationForest(n_estimators=120, contamination=0.02, random_state=42, n_jobs=-1)
    flags = m.fit_predict(df[feats])  # -1 = anomaly, 1 = normal
    return {"flagged_rate": round(float((flags == -1).mean()), 4),
            "contamination_setting": 0.02}


def _bar_plot(names, values, title, path):
    plt.figure(figsize=(6, 3.5))
    order = np.argsort(values)
    plt.barh(np.array(names)[order], np.array(values)[order], color="#3b82f6")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def _confusion_plot(cm, labels, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], labels); plt.yticks([0, 1], labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Health confusion matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    df = F.load_raw()
    metrics = {
        "temperature_soft_sensor": evaluate_temperature(df),
        "battery_health_classifier": evaluate_health(df),
        "anomaly_detector": evaluate_anomaly(df),
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nReports + plots saved to ml/reports/")


if __name__ == "__main__":
    main()
