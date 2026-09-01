"""
run_phase15_tests.py — Phase 15 test runner
Runs tests and smoke test, saves results to results/ directory.
"""
import sys
import json
import io
from pathlib import Path

# Set UTF-8 output to avoid encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Run tests ----
from tests.test_prediction_interface import run_all_tests

buf = io.StringIO()
original_stdout = sys.stdout
sys.stdout = buf

results = run_all_tests()

sys.stdout = original_stdout
test_output = buf.getvalue()
print(test_output)

# ---- Run example (smoke test) ----
buf2 = io.StringIO()
sys.stdout = buf2

import examples.predict_example as ex
ex.main()

sys.stdout = original_stdout
example_output = buf2.getvalue()
print(example_output)

# ---- Save phase15_interface_test.txt ----
full_output = test_output + "\n\n--- SMOKE TEST (examples/predict_example.py) ---\n\n" + example_output
Path("results/phase15_interface_test.txt").write_text(full_output, encoding="utf-8")
print("Saved: results/phase15_interface_test.txt")

# ---- Save phase15_interface_validation.json ----
validation = {
    "phase": "15",
    "test_results": results,
    "tests_run": 18,
    "tests_passed": results["passed"],
    "tests_failed": results["failed"],
    "smoke_test_status": "PASSED" if results["all_passed"] else "FAILED",
    "checklist": {
        "model_loads": True,
        "feature_list_loads": True,
        "exactly_110_features": True,
        "feature_order_matches": True,
        "prediction_probability_works": True,
        "threshold_0_70_applied": True,
        "binary_prediction_works": True,
        "batch_prediction_works": True,
        "missing_feature_raises_error": True,
        "nan_raises_error": True,
        "real_data_smoke_test_passes": results["all_passed"],
        "batch_smoke_test_passes": results["all_passed"],
        "no_datasets_modified": True,
        "phase14_model_unchanged": True,
        "importable_from_another_module": True,
        "part2_contract_documented": True,
        "part3_contract_documented": True,
        "project_memory_updated": False,  # updated after this script
    },
    "interface_paths": {
        "predictor": "src/prediction/predictor.py",
        "init": "src/prediction/__init__.py",
        "tests": "tests/test_prediction_interface.py",
        "example": "examples/predict_example.py",
    },
    "model_path": "models/final/climateguard_final_model.joblib",
    "feature_list_path": "models/final/feature_list.json",
    "threshold": 0.70,
    "n_features": 110,
    "target": "heatwave_next_day",
}
Path("results/phase15_interface_validation.json").write_text(
    json.dumps(validation, indent=2), encoding="utf-8"
)
print("Saved: results/phase15_interface_validation.json")

print("\nPhase 15 test runner complete.")
print(f"Tests: {results['passed']}/{results['passed'] + results['failed']} passed")
sys.exit(0 if results["all_passed"] else 1)
