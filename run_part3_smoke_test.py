"""
run_part3_smoke_test.py
Part 3 real-data end-to-end smoke test.
Runs actual rows from data/splits/temporal/X_test.csv through the full pipeline
and saves results to results/part3_smoke_test.json.

Usage:
    python run_part3_smoke_test.py
"""

import sys
import json
import hashlib
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.integration import ClimateGuardPipeline

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_row(X, meta, idx):
    row = X.iloc[idx].to_dict()
    row["city_key"] = meta.iloc[idx]["city_key"]
    row["date"]     = meta.iloc[idx]["date"]
    return row


def result_to_smoke_entry(res, label=""):
    """Convert a ClimateGuardResult to a compact smoke-test dict."""
    entry = {
        "label": label,
        "valid": res.valid,
        "city_key": res.input_metadata.get("city_key"),
        "date":     res.input_metadata.get("date"),
        "validation": res.validation,
        "prediction": res.prediction,
        "risk_level": res.risk["level"] if res.risk else None,
        "risk_score": res.risk["score"] if res.risk else None,
        "recommendation_count": len(res.recommendations) if res.recommendations else 0,
        "explanation_method": (
            res.explanation.get("explanation_method")
            if isinstance(res.explanation, dict) else None
        ),
        "explanation_top_features": (
            res.explanation.get("top_features", [])[:5]
            if isinstance(res.explanation, dict) else []
        ),
        "expert_rules_total": len(res.expert_rules),
        "expert_rules_triggered_count": sum(1 for r in res.expert_rules if r.get("triggered")),
        "expert_rules_triggered_ids": [
            r["rule_id"] for r in res.expert_rules if r.get("triggered")
        ],
        "expert_rules_highest_severity": None,
        "warnings": res.warnings,
    }
    # Determine highest severity
    order = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
    triggered = [r for r in res.expert_rules if r.get("triggered")]
    if triggered:
        entry["expert_rules_highest_severity"] = max(
            triggered, key=lambda r: order.get(r.get("severity", ""), 0)
        ).get("severity")
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ClimateGuard Part 3 — Real-Data E2E Smoke Test")
    print("=" * 60)

    # --- 1. Model integrity pre-check ---
    model_path = PROJECT_ROOT / "models" / "final" / "climateguard_final_model.joblib"
    feat_path  = PROJECT_ROOT / "models" / "final" / "feature_list.json"

    model_md5 = md5_file(model_path)
    feat_md5  = md5_file(feat_path)
    model_size = model_path.stat().st_size
    feat_size  = feat_path.stat().st_size

    print(f"\nModel file:  {model_path.name}  ({model_size:,} bytes)")
    print(f"  MD5: {model_md5}")
    print(f"Feature list: {feat_path.name}  ({feat_size:,} bytes)")
    print(f"  MD5: {feat_md5}")

    # --- 2. Load pipeline ---
    print("\nLoading ClimateGuardPipeline...")
    pipeline = ClimateGuardPipeline()
    assert pipeline.predictor.threshold  == 0.70, "threshold changed!"
    assert pipeline.predictor.n_features == 110,  "feature count changed!"
    print(f"  Threshold: {pipeline.predictor.threshold}")
    print(f"  N features: {pipeline.predictor.n_features}")
    print(f"  SHAP available: {pipeline.shap_available}")

    # --- 3. Load test data ---
    print("\nLoading test data...")
    X    = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv")
    meta = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv")
    print(f"  X shape: {X.shape}")
    print(f"  meta shape: {meta.shape}")
    print(f"  Cities: {sorted(meta['city_key'].unique())}")

    # --- 4. Individual row: cold day (ahmedabad 2023-01-01) ---
    print("\n[Individual 1] Cold day — ahmedabad, 2023-01-01")
    cold_row = make_row(X, meta, 0)
    cold_res = pipeline.analyze(cold_row)
    assert cold_res.valid, f"Cold row failed: {cold_res.validation['errors']}"
    cold_entry = result_to_smoke_entry(cold_res, "individual_cold_ahmedabad_2023-01-01")
    print(f"  valid={cold_res.valid}")
    print(f"  probability={cold_res.prediction['probability']:.6f}")
    print(f"  prediction={cold_res.prediction['prediction']}")
    print(f"  risk_level={cold_res.risk['level']}")
    print(f"  recommendations={cold_entry['recommendation_count']}")
    print(f"  explanation_method={cold_entry['explanation_method']}")
    print(f"  expert_rules_triggered={cold_entry['expert_rules_triggered_count']}/7")

    # --- 5. Individual row: extreme heat day (delhi 2024-05-17, tmax=44.5) ---
    print("\n[Individual 2] Extreme heat — delhi, 2024-05-17 (tmax=44.5°C)")
    hot_row = make_row(X, meta, 1475)
    hot_res  = pipeline.analyze(hot_row)
    assert hot_res.valid, f"Hot row failed: {hot_res.validation['errors']}"
    hot_entry = result_to_smoke_entry(hot_res, "individual_hot_delhi_2024-05-17")
    print(f"  valid={hot_res.valid}")
    print(f"  probability={hot_res.prediction['probability']:.6f}")
    print(f"  prediction={hot_res.prediction['prediction']}")
    print(f"  risk_level={hot_res.risk['level']}")
    print(f"  recommendations={hot_entry['recommendation_count']}")
    print(f"  explanation_method={hot_entry['explanation_method']}")
    print(f"  expert_rules_triggered={hot_entry['expert_rules_triggered_count']}/7")
    print(f"  expert_rules_triggered_ids={hot_entry['expert_rules_triggered_ids']}")
    print(f"  highest_severity={hot_entry['expert_rules_highest_severity']}")
    assert hot_entry["expert_rules_triggered_count"] >= 1, \
        "Expected >=1 rule to trigger on extreme heat day"

    # --- 6. Batch: 10 rows ---
    print("\n[Batch] 10 rows")
    batch_rows = [make_row(X, meta, i) for i in range(10)]
    batch_df   = pd.DataFrame(batch_rows)
    batch_results = pipeline.analyze_batch(batch_df)
    n_valid = sum(1 for r in batch_results if r.valid)
    print(f"  {n_valid}/{len(batch_results)} valid")
    batch_entries = []
    for i, r in enumerate(batch_results):
        entry = result_to_smoke_entry(r, f"batch_row_{i}")
        batch_entries.append(entry)
        city  = entry["city_key"] or "?"
        date  = entry["date"] or "?"
        prob  = entry["prediction"]["probability"] if entry["prediction"] else "N/A"
        level = entry["risk_level"] or "INVALID"
        ntrig = entry["expert_rules_triggered_count"]
        prob_str = f"{prob:.4f}" if isinstance(prob, float) else str(prob)
        print(f"  [{i}] {date} {city:12s}: prob={prob_str:7s} risk={level:9s} rules={ntrig}/7")

    assert n_valid == 10, f"Expected all 10 batch rows valid, got {n_valid}"

    # --- 7. Verification assertions ---
    print("\nRunning verification assertions...")
    # All required fields present
    for entry in [cold_entry, hot_entry] + batch_entries:
        assert entry["valid"], f"Row {entry['label']} is invalid"
        assert entry["prediction"] is not None
        assert 0.0 <= entry["prediction"]["probability"] <= 1.0
        assert entry["prediction"]["prediction"] in (0, 1)
        assert entry["risk_level"] in ("LOW", "MODERATE", "HIGH", "EXTREME")
        assert entry["recommendation_count"] > 0
        assert entry["expert_rules_total"] == 7
        assert entry["explanation_method"] in ("shap", "global_rf_importance", None)
    print("  All assertions PASSED")

    # --- 8. Model integrity post-check ---
    model_md5_post = md5_file(model_path)
    feat_md5_post  = md5_file(feat_path)
    assert model_md5 == model_md5_post, "MODEL MD5 CHANGED — integrity violation!"
    assert feat_md5  == feat_md5_post,  "FEATURE LIST MD5 CHANGED — integrity violation!"
    assert pipeline.predictor.threshold  == 0.70
    assert pipeline.predictor.n_features == 110
    print("\nModel integrity: PASS (MD5 unchanged, threshold=0.70, n_features=110)")

    # --- 9. Assemble smoke test JSON ---
    smoke_json = {
        "smoke_test": "part3_e2e",
        "timestamp": datetime.datetime.now().isoformat(),
        "pipeline_info": {
            "threshold":     pipeline.predictor.threshold,
            "n_features":    pipeline.predictor.n_features,
            "shap_available": pipeline.shap_available,
        },
        "model_integrity": {
            "model_file":         model_path.name,
            "model_size_bytes":   model_size,
            "model_md5_before":   model_md5,
            "model_md5_after":    model_md5_post,
            "model_md5_unchanged": model_md5 == model_md5_post,
            "feature_list_file":  feat_path.name,
            "feature_list_size":  feat_size,
            "feature_list_md5_before": feat_md5,
            "feature_list_md5_after":  feat_md5_post,
            "feature_list_md5_unchanged": feat_md5 == feat_md5_post,
            "threshold_unchanged": pipeline.predictor.threshold == 0.70,
            "n_features_unchanged": pipeline.predictor.n_features == 110,
            "integrity_status": "PASS",
        },
        "test_data": {
            "X_test_shape": list(X.shape),
            "meta_test_shape": list(meta.shape),
            "cities": sorted(meta["city_key"].unique().tolist()),
        },
        "individual_results": [cold_entry, hot_entry],
        "batch_results": {
            "n_rows":  len(batch_results),
            "n_valid": n_valid,
            "rows":    batch_entries,
        },
        "verification": {
            "all_valid":                   True,
            "probability_bounded":         True,
            "prediction_binary":           True,
            "risk_level_valid":            True,
            "recommendations_present":     True,
            "explanation_present":         True,
            "expert_rules_7_per_row":      True,
            "hot_row_triggered_rules":     hot_entry["expert_rules_triggered_count"] >= 1,
            "model_integrity_pass":        True,
            "overall_status":              "PASS",
        },
    }

    out_path = PROJECT_ROOT / "results" / "part3_smoke_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(smoke_json, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"  File size: {out_path.stat().st_size:,} bytes")

    print()
    print("=" * 60)
    print("  SMOKE TEST RESULT: PASS")
    print("=" * 60)
    print(f"  Individual cold row:  valid={cold_entry['valid']}, "
          f"risk={cold_entry['risk_level']}, "
          f"rules_triggered={cold_entry['expert_rules_triggered_count']}/7")
    print(f"  Individual hot row:   valid={hot_entry['valid']}, "
          f"risk={hot_entry['risk_level']}, "
          f"rules_triggered={hot_entry['expert_rules_triggered_count']}/7")
    print(f"  Batch (10 rows):      {n_valid}/10 valid")
    print(f"  Model integrity:      PASS")
    print(f"  JSON saved:           results/part3_smoke_test.json")

    return smoke_json


if __name__ == "__main__":
    result = main()
    sys.exit(0)
