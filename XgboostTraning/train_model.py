"""
Indoor Navigation — XGBoost Training Pipeline
=============================================
Usage:
    python train_model.py                          # uses fingerprints.json
    python train_model.py --data my_data.json      # custom file
    python train_model.py --data my_data.json --test-size 0.25

Output:
    models/node_classifier.pkl
    models/x_model.pkl
    models/y_model.pkl
    models/label_encoder.pkl
    models/beacons.pkl
    models/feature_columns.pkl
    models/training_report.json
"""

import json
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

from xgboost import XGBClassifier, XGBRegressor


# ══════════════════════════════════════════════
# CLI ARGS
# ══════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Train indoor navigation models")
parser.add_argument("--data",      default="fingerprints.json", help="Path to fingerprints JSON")
parser.add_argument("--test-size", type=float, default=0.2,    help="Test split ratio (default: 0.2)")
parser.add_argument("--seed",      type=int,   default=42,     help="Random seed")
parser.add_argument("--no-cv",     action="store_true",        help="Skip cross-validation (faster)")
args = parser.parse_args()


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def section(title):
    print(f"\n{'═'*50}")
    print(f"  {title}")
    print(f"{'═'*50}")

def subsection(title):
    print(f"\n── {title}")


# ══════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════

section("LOADING DATA")

if not os.path.exists(args.data):
    print(f"❌  File not found: {args.data}")
    print("    Create fingerprints.json or pass --data <path>")
    exit(1)

with open(args.data, "r") as f:
    fingerprints = json.load(f)

print(f"✓  Loaded {len(fingerprints)} fingerprints from {args.data}")

# Validate minimum dataset size
unique_nodes = set(fp["nodeId"] for fp in fingerprints)
print(f"✓  Unique nodes: {len(unique_nodes)}")

if len(fingerprints) < 50:
    print(f"⚠️   WARNING: Only {len(fingerprints)} fingerprints. Recommend 200+ for reliable results.")
if len(unique_nodes) < 10:
    print(f"⚠️   WARNING: Only {len(unique_nodes)} nodes. Recommend 50+ nodes.")


# ══════════════════════════════════════════════
# 2. DISCOVER ALL BEACONS
# ══════════════════════════════════════════════

section("BEACON DISCOVERY")

all_beacons = set()
for fp in fingerprints:
    for beacon_id in fp["signals"].keys():
        all_beacons.add(beacon_id)

all_beacons = sorted(list(all_beacons))
print(f"✓  Found {len(all_beacons)} beacons: {all_beacons}")

if len(all_beacons) < 3:
    print("⚠️   WARNING: Fewer than 3 beacons. Accuracy will be poor. Aim for 4–8 beacons.")


# ══════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════

section("BUILDING FEATURES")

MISSING_RSSI = -100   # value used when a beacon is not visible

rows = []
for fp in fingerprints:
    row = {}

    # ── RSSI features (one column per beacon) ──────────────────
    for beacon in all_beacons:
        if beacon in fp["signals"]:
            sig = fp["signals"][beacon]
            # Support both { "median": -70 } and raw -70
            row[f"b_{beacon}"] = sig["median"] if isinstance(sig, dict) else sig
        else:
            row[f"b_{beacon}"] = MISSING_RSSI

    # ── Heading features ───────────────────────────────────────
    # sin/cos encoding avoids the 0°/360° discontinuity
    row["headingSin"] = fp.get("headingSin", 0.0)
    row["headingCos"] = fp.get("headingCos", 1.0)

    # ── Targets ────────────────────────────────────────────────
    row["nodeId"] = fp["nodeId"]
    row["x"]      = fp["x"]
    row["y"]      = fp["y"]

    rows.append(row)

df = pd.DataFrame(rows)

feature_columns = [c for c in df.columns if c not in ["nodeId", "x", "y"]]
X = df[feature_columns]

print(f"✓  Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"   Features: {feature_columns}")
subsection("Sample data")
print(df.head(3).to_string())


# ══════════════════════════════════════════════
# 4. NODE CLASSIFIER
# ══════════════════════════════════════════════

section("TRAINING NODE CLASSIFIER")

label_encoder = LabelEncoder()
y_node = label_encoder.fit_transform(df["nodeId"])

print(f"   Classes: {list(label_encoder.classes_)}")

# Stratify only when test set will have at least one sample per class
n_classes = len(np.unique(y_node))
min_test  = int(len(y_node) * args.test_size)
use_stratify = y_node if min_test >= n_classes else None
if use_stratify is None:
    print(f"  ⚠️   Stratify disabled (test set too small for {n_classes} classes). Collect more data.")

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_node,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=use_stratify,
)

node_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=args.seed,
    n_jobs=-1,
)

node_model.fit(
    X_train_cls, y_train_cls,
    eval_set=[(X_test_cls, y_test_cls)],
    verbose=False,
)

preds_cls = node_model.predict(X_test_cls)
accuracy  = accuracy_score(y_test_cls, preds_cls)
print(f"\n  Holdout Accuracy : {accuracy*100:.2f}%")

# Cross-validation — auto-pick safe number of folds
# n_splits must be <= min samples per class
cv_accuracy = None
if not args.no_cv:
    min_class_count = int(np.bincount(y_node).min())
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        subsection(f"{n_splits}-Fold Cross-Validation")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(node_model, X, y_node, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_accuracy = cv_scores.mean()
        print(f"  CV Accuracy: {cv_accuracy*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    else:
        print("  WARNING: CV skipped — need at least 2 fingerprints per node.")

subsection("Per-class report (top 10 nodes)")
present_classes = sorted(set(y_test_cls) | set(preds_cls))
present_names   = label_encoder.inverse_transform(present_classes)
report = classification_report(
    y_test_cls, preds_cls,
    labels=present_classes,
    target_names=present_names,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).T
print(report_df[["precision","recall","f1-score"]].head(10).round(2).to_string())


# ══════════════════════════════════════════════
# 5. X REGRESSOR
# ══════════════════════════════════════════════

section("TRAINING X REGRESSOR")

X_train_reg, X_test_reg, yx_train, yx_test = train_test_split(
    X, df["x"],
    test_size=args.test_size,
    random_state=args.seed,
)

x_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=args.seed,
    n_jobs=-1,
)

x_model.fit(X_train_reg, yx_train, eval_set=[(X_test_reg, yx_test)], verbose=False)

pred_x  = x_model.predict(X_test_reg)
mae_x   = mean_absolute_error(yx_test, pred_x)
print(f"  Holdout MAE (X) : {mae_x:.3f} meters")


# ══════════════════════════════════════════════
# 6. Y REGRESSOR
# ══════════════════════════════════════════════

section("TRAINING Y REGRESSOR")

X_train_reg, X_test_reg, yy_train, yy_test = train_test_split(
    X, df["y"],
    test_size=args.test_size,
    random_state=args.seed,
)

y_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=args.seed,
    n_jobs=-1,
)

y_model.fit(X_train_reg, yy_train, eval_set=[(X_test_reg, yy_test)], verbose=False)

pred_y  = y_model.predict(X_test_reg)
mae_y   = mean_absolute_error(yy_test, pred_y)
print(f"  Holdout MAE (Y) : {mae_y:.3f} meters")


# ══════════════════════════════════════════════
# 7. SAVE MODELS
# ══════════════════════════════════════════════

section("SAVING MODELS")

os.makedirs("models", exist_ok=True)

joblib.dump(node_model,      "models/node_classifier.pkl")
joblib.dump(x_model,         "models/x_model.pkl")
joblib.dump(y_model,         "models/y_model.pkl")
joblib.dump(label_encoder,   "models/label_encoder.pkl")
joblib.dump(all_beacons,     "models/beacons.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

# Save a training report for the API to serve
report_data = {
    "trained_at":       datetime.utcnow().isoformat() + "Z",
    "fingerprint_file": args.data,
    "n_fingerprints":   len(fingerprints),
    "n_nodes":          len(unique_nodes),
    "n_beacons":        len(all_beacons),
    "beacons":          all_beacons,
    "nodes":            sorted(list(unique_nodes)),
    "metrics": {
        "node_accuracy_holdout": round(accuracy, 4),
        "node_accuracy_cv":      round(cv_accuracy, 4) if cv_accuracy else None,
        "x_mae_meters":          round(float(mae_x), 4),
        "y_mae_meters":          round(float(mae_y), 4),
    }
}

with open("models/training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("  ✓  models/node_classifier.pkl")
print("  ✓  models/x_model.pkl")
print("  ✓  models/y_model.pkl")
print("  ✓  models/label_encoder.pkl")
print("  ✓  models/beacons.pkl")
print("  ✓  models/feature_columns.pkl")
print("  ✓  models/training_report.json")


# ══════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════

section("TRAINING COMPLETE")
print(f"""
  Fingerprints  : {len(fingerprints)}
  Nodes         : {len(unique_nodes)}
  Beacons       : {len(all_beacons)}

  Node Accuracy : {accuracy*100:.2f}%  (holdout)
  X Error       : {mae_x:.3f} m
  Y Error       : {mae_y:.3f} m

  → Start the API:   uvicorn api:app --reload
  → Test it:         open http://localhost:8000
""")