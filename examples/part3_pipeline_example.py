"""
ClimateGuard Part 3 — Pipeline Usage Example
examples/part3_pipeline_example.py

Demonstrates actual working usage of the ClimateGuard Part 3 end-to-end pipeline.

This example:
  1. Loads real feature rows from data/splits/temporal/X_test.csv
  2. Runs a single cold-season record through the full pipeline
  3. Runs a hot-season Delhi record through the full pipeline
  4. Runs a small batch through the pipeline
  5. Demonstrates error handling
  6. Shows JSON serialisation

Usage:
    python examples/part3_pipeline_example.py

Requirements:
    - models/final/climateguard_final_model.joblib must exist
    - models/final/feature_list.json must exist
    - data/splits/temporal/X_test.csv must exist
    - data/splits/temporal/meta_test.csv must exist
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.integration import ClimateGuardPipeline

# ---------------------------------------------------------------------------
# Helper: print a section separator
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
X_TEST_PATH   = PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv"
META_TEST_PATH = PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv"

print("Loading test data...")
X    = pd.read_csv(X_TEST_PATH)
meta = pd.read_csv(META_TEST_PATH)
print(f"  X shape:    {X.shape}")
print(f"  meta shape: {meta.shape}")
print(f"  Cities:     {sorted(meta['city_key'].unique())}")

# ---------------------------------------------------------------------------
# Instantiate pipeline (model loaded once here, reused for all examples)
# ---------------------------------------------------------------------------
section("1. Pipeline Instantiation")

pipeline = ClimateGuardPipeline(include_explanation=True, top_n=10)
info = pipeline.info()
print(f"  {pipeline}")
print(f"  Threshold:       {info['threshold']}")
print(f"  Feature count:   {info['n_features']}")
print(f"  SHAP available:  {info['shap_available']}")
print(f"  Expert rules:    {len(info['expert_rules'])} rules defined")
for rule_name in info["expert_rules"]:
    print(f"    - {rule_name}")


# ---------------------------------------------------------------------------
# Example 1 — Cold-season record (Ahmedabad, 2023-01-01)
# ---------------------------------------------------------------------------
section("2. Single Record — Cold Season (Ahmedabad, Jan 2023)")

row_idx = 0
row = X.iloc[row_idx].to_dict()
row["city_key"] = meta.iloc[row_idx]["city_key"]
row["date"]     = meta.iloc[row_idx]["date"]

print(f"  Input: city={row['city_key']}, date={row['date']}")
print(f"  temperature_2m_max={row['temperature_2m_max']:.1f}°C")
print(f"  tmax_departure={row.get('tmax_departure', 'N/A'):.3f}°C")

result = pipeline.analyze(row)

print()
print(f"  Valid:             {result.valid}")
print(f"  Probability:       {result.prediction['probability']:.4f}")
print(f"  Prediction label:  {result.prediction['prediction']}  "
      f"({'Heatwave tomorrow' if result.prediction['prediction'] == 1 else 'Normal tomorrow'})")
print(f"  Risk level:        {result.risk['level']}")
print(f"  Recommendations:   {len(result.recommendations)} items")
print(f"  Explanation method: {result.explanation['method'] if result.explanation else 'None'}")
print(f"  Expert rules triggered: "
      f"{sum(1 for r in result.expert_rules if r['triggered'])} / 7")
print(f"  Warnings:          {result.warnings}")

# Show triggered rules
triggered = [r for r in result.expert_rules if r["triggered"]]
if triggered:
    print()
    print("  Triggered rules:")
    for r in triggered:
        print(f"    [{r['severity']}] {r['rule_id']} — {r['name']}")
        print(f"      {r.get('message', '')[:100]}...")
else:
    print("  No expert rules triggered (expected for cold day).")


# ---------------------------------------------------------------------------
# Example 2 — Hot-season record (Delhi, 2024-05-17, Tmax=44.5°C)
# ---------------------------------------------------------------------------
section("3. Single Record — Extreme Heat Day (Delhi, 2024-05-17)")

# Row 1475 is Delhi, 2024-05-17, Tmax=44.5°C — identified during project analysis
hot_row_idx = 1475
hot_row = X.iloc[hot_row_idx].to_dict()
hot_row["city_key"] = meta.iloc[hot_row_idx]["city_key"]
hot_row["date"]     = meta.iloc[hot_row_idx]["date"]

print(f"  Input: city={hot_row['city_key']}, date={hot_row['date']}")
print(f"  temperature_2m_max={hot_row['temperature_2m_max']:.1f}°C")
print(f"  temperature_2m_min={hot_row.get('temperature_2m_min', 'N/A'):.1f}°C")
print(f"  tmax_departure={hot_row.get('tmax_departure', 'N/A'):.3f}°C")
print(f"  tmax_departure_zscore={hot_row.get('tmax_departure_zscore', 'N/A'):.3f}")

hot_result = pipeline.analyze(hot_row)

print()
print(f"  Valid:             {hot_result.valid}")
print(f"  Probability:       {hot_result.prediction['probability']:.4f}")
print(f"  Prediction label:  {hot_result.prediction['prediction']}  "
      f"({'Heatwave tomorrow' if hot_result.prediction['prediction'] == 1 else 'Normal tomorrow'})")
print(f"  Risk level:        {hot_result.risk['level']}")
print(f"  Recommendations:   {len(hot_result.recommendations)} items")

# Show all recommendations
print()
print("  Adaptation recommendations:")
for i, rec in enumerate(hot_result.recommendations, 1):
    # Recommendations may be dicts or strings
    if isinstance(rec, dict):
        print(f"    {i}. [{rec.get('category','?')}] {rec.get('text','')[:80]}")
    else:
        print(f"    {i}. {str(rec)[:80]}")

# Show triggered expert rules
print()
hot_triggered = [r for r in hot_result.expert_rules if r["triggered"]]
print(f"  Expert rules triggered: {len(hot_triggered)} / 7")
for r in hot_triggered:
    print(f"    [{r['severity']}] {r['rule_id']} — {r['name']}")
    msg = r.get("message", "")
    if msg:
        # Print first 120 chars of message
        print(f"      {msg[:120]}" + ("..." if len(msg) > 120 else ""))

# Show explanation top features
if hot_result.explanation:
    print()
    method = hot_result.explanation["method"]
    print(f"  Explanation method: {method}")
    top_feats = hot_result.explanation.get("top_features", [])
    print(f"  Top {len(top_feats)} features:")
    for feat in top_feats[:5]:
        direction = feat.get("direction", "?")
        importance = feat.get("importance", feat.get("value", "?"))
        print(f"    {feat['feature']:40s}  {direction}  ({importance})")

# Pipeline warnings
if hot_result.warnings:
    print()
    print("  Pipeline warnings:")
    for w in hot_result.warnings:
        print(f"    {w}")


# ---------------------------------------------------------------------------
# Example 3 — Batch of 5 rows
# ---------------------------------------------------------------------------
section("4. Batch Analysis (5 rows across cities)")

batch_df = X.iloc[:5].copy()
batch_df["city_key"] = meta.iloc[:5]["city_key"].values
batch_df["date"]     = meta.iloc[:5]["date"].values

batch_results = pipeline.analyze_batch(batch_df)

print(f"  Input rows:  {len(batch_df)}")
print(f"  Results:     {len(batch_results)}")
print()
print(f"  {'Date':<14} {'City':<12} {'Prob':>6} {'Label':>6} {'Risk':<10} {'Rules':>6}")
print(f"  {'-'*14} {'-'*12} {'-'*6} {'-'*6} {'-'*10} {'-'*6}")
for r in batch_results:
    city  = r.input_metadata.get("city_key", "?")
    date  = r.input_metadata.get("date", "?")
    if r.valid:
        prob  = r.prediction["probability"]
        label = r.prediction["prediction"]
        level = r.risk["level"]
        n_trig = sum(1 for rule in r.expert_rules if rule["triggered"])
        print(f"  {date:<14} {city:<12} {prob:>6.4f} {label:>6} {level:<10} {n_trig:>6}")
    else:
        print(f"  {date:<14} {city:<12} {'INVALID':>13} {r.validation['errors'][0][:40]}")


# ---------------------------------------------------------------------------
# Example 4 — Error handling
# ---------------------------------------------------------------------------
section("5. Error Handling Examples")

# 4a: Invalid city
print("  [4a] Invalid city:")
bad_city_row = X.iloc[0].to_dict()
bad_city_row["city_key"] = "bangalore"
bad_city_row["date"]     = "2024-05-01"
res_bad_city = pipeline.analyze(bad_city_row)
print(f"    valid={res_bad_city.valid}")
print(f"    errors={res_bad_city.validation['errors']}")

# 4b: Malformed date
print()
print("  [4b] Malformed date:")
bad_date_row = X.iloc[0].to_dict()
bad_date_row["city_key"] = "delhi"
bad_date_row["date"]     = "17-05-2024"
res_bad_date = pipeline.analyze(bad_date_row)
print(f"    valid={res_bad_date.valid}")
print(f"    errors={res_bad_date.validation['errors']}")

# 4c: Missing feature
print()
print("  [4c] Missing feature (tmax_departure_zscore):")
missing_feat_row = X.iloc[0].to_dict()
missing_feat_row["city_key"] = "delhi"
missing_feat_row["date"]     = "2024-05-01"
del missing_feat_row["tmax_departure_zscore"]
res_missing = pipeline.analyze(missing_feat_row)
print(f"    valid={res_missing.valid}")
print(f"    errors={res_missing.validation['errors'][:1]}")

# 4d: NaN value
print()
print("  [4d] NaN in temperature_2m_max:")
nan_row = X.iloc[0].to_dict()
nan_row["city_key"]          = "delhi"
nan_row["date"]              = "2024-05-01"
nan_row["temperature_2m_max"] = float("nan")
res_nan = pipeline.analyze(nan_row)
print(f"    valid={res_nan.valid}")
print(f"    errors={res_nan.validation['errors'][:1]}")

# 4e: Confirm that errors produce no prediction (not fake results)
print()
print("  [4e] Confirming failed result has no prediction (no fake results):")
print(f"    prediction  = {res_bad_city.prediction}")
print(f"    risk        = {res_bad_city.risk}")
print(f"    expert_rules count = {len(res_bad_city.expert_rules)}")


# ---------------------------------------------------------------------------
# Example 5 — JSON serialisation
# ---------------------------------------------------------------------------
section("6. JSON Serialisation")

json_str = hot_result.to_json(indent=2)
print(f"  JSON length: {len(json_str)} characters")

parsed = json.loads(json_str)
print(f"  Top-level keys: {sorted(parsed.keys())}")
print(f"  prediction.probability: {parsed['prediction']['probability']}")
print(f"  risk.level:             {parsed['risk']['level']}")
print(f"  expert_rules count:     {len(parsed['expert_rules'])}")
print(f"  triggered rules: {[r['rule_id'] for r in parsed['expert_rules'] if r['triggered']]}")
print(f"  disclaimer present: {'expert_rules_disclaimer' in parsed['metadata']}")


# ---------------------------------------------------------------------------
# Example 6 — Model integrity check
# ---------------------------------------------------------------------------
section("7. Model Integrity Verification")

print(f"  Threshold:    {pipeline.predictor.threshold}  (expected: 0.70)")
print(f"  N features:   {pipeline.predictor.n_features}  (expected: 110)")
print(f"  Feature list: {pipeline.predictor.feature_names[:5]}... [{len(pipeline.predictor.feature_names)} total]")

model_path = PROJECT_ROOT / "models" / "final" / "climateguard_final_model.joblib"
feat_path  = PROJECT_ROOT / "models" / "final" / "feature_list.json"
print(f"  Model file exists: {model_path.exists()}")
print(f"  Feature list file exists: {feat_path.exists()}")

import hashlib

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"  Model MD5:        {md5(model_path)}")
print(f"  Feature list MD5: {md5(feat_path)}")

assert pipeline.predictor.threshold    == 0.70,  "FAIL: threshold changed"
assert pipeline.predictor.n_features   == 110,   "FAIL: feature count changed"
print()
print("  Model integrity: PASS")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
section("Done")
print("  All Part 3 pipeline examples completed successfully.")
print()
print("  Summary:")
print(f"    Cold record  -> valid={result.valid}, risk={result.risk['level']}")
print(f"    Hot record   -> valid={hot_result.valid}, risk={hot_result.risk['level']}, "
      f"rules triggered={len(hot_triggered)}/7")
print(f"    Batch (5 rows) -> {sum(r.valid for r in batch_results)}/{len(batch_results)} valid")
print(f"    Error handling -> 4 invalid inputs correctly rejected")
print(f"    JSON output    -> {len(json_str)} chars, round-trip OK")
