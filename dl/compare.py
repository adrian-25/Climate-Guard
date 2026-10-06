"""dl/compare.py — Generate comparison table, CSV, JSON, and PNGs.

Usage
-----
  python -m dl.compare

Outputs in dl/results/:
  dl_vs_rf.csv
  dl_vs_rf.json
  dl_vs_rf.md
  pr_curves.png
  per_city_f1.png
  calibration.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import DL_RESULTS
from .evaluate import evaluate_all

# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _make_row(
    name: str, train_window: str, features: str, threshold_rule: str, metrics: dict, note: str = ""
) -> dict:
    def _r(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return round(v, 4)
        return v

    return {
        "model": name,
        "training_window": train_window,
        "features": features,
        "threshold_rule": threshold_rule,
        "F1": _r(metrics.get("f1")),
        "Precision": _r(metrics.get("precision")),
        "Recall": _r(metrics.get("recall")),
        "PR-AUC": _r(metrics.get("pr_auc")),
        "ROC-AUC": _r(metrics.get("roc_auc")),
        "Brier": _r(metrics.get("brier")),
        "ECE": _r(metrics.get("ece")),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_pr_curves(
    model_probs: dict[str, np.ndarray], labels: np.ndarray, out_path: Path, pr_points: dict
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "RF production": "#C8421B",
        "RF fair": "#8C8580",
        "GRU raw-seq ens": "#2563EB",
        "GRU raw-seq focal ens": "#D97706",
        "LSTM raw-seq ens": "#7C3AED",
        "GRU feat110 ens": "#059669",
    }
    for name, probs in model_probs.items():
        if probs is None or np.isnan(probs).any():
            continue
        pr, re, _ = precision_recall_curve(labels.astype(int), probs)
        auc = average_precision_score(labels.astype(int), probs)
        color = colors.get(name, "#666")
        ax.plot(re, pr, label=f"{name} (AUC={auc:.3f})", color=color, lw=1.6)
        pr_points[name] = {"precision": pr.tolist(), "recall": re.tolist(), "auc": round(auc, 4)}

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves (India test set 2023–2025)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Annotate sparsity
    ax.text(
        0.01,
        0.01,
        f"38 positives / 4 865 test rows (0.78% base rate)",
        transform=ax.transAxes,
        fontsize=7,
        color="grey",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Saved {out_path}")


def _plot_per_city(rows: list[dict], out_path: Path) -> None:
    """Bar chart: per-city F1 (n/a shown as missing bar, not zero)."""
    # Collect city data for a few key models
    cities = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
    city_labels = ["Delhi", "Lucknow", "Nagpur", "Ahmedabad\n(n/a)", "Mumbai\n(n/a)"]
    x = np.arange(len(cities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    model_names = [
        r["model"] for r in rows if "DL" not in r["model"] or "ens" in r["model"].lower()
    ]
    palette = {"RF production": "#C8421B", "RF fair": "#8C8580"}
    dl_colors = ["#2563EB", "#D97706", "#7C3AED", "#059669"]
    dl_i = 0
    offset = -width
    for row in rows:
        name = row["model"]
        per_city = row.get("per_city", {})
        if not per_city:
            continue
        f1s = []
        for city in cities:
            m = per_city.get(city)
            f1s.append(m["f1"] if m and m.get("f1") is not None else np.nan)
        color = palette.get(name, dl_colors[dl_i % len(dl_colors)])
        if name not in palette:
            dl_i += 1
        bars = ax.bar(x + offset, f1s, width, label=name, color=color, alpha=0.85)
        offset += width

    ax.set_xticks(x)
    ax.set_xticklabels(city_labels, fontsize=8)
    ax.set_ylabel("F1 (val-optimal threshold)")
    ax.set_title("Per-city F1 on India test set 2023–2025\n(grey/missing = no test positives)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Saved {out_path}")


def _plot_calibration(
    model_probs: dict[str, np.ndarray], labels: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    colors = {
        "RF production": "#C8421B",
        "RF fair": "#8C8580",
        "GRU raw-seq ens": "#2563EB",
        "GRU raw-seq focal ens": "#D97706",
        "LSTM raw-seq ens": "#7C3AED",
        "GRU feat110 ens": "#059669",
    }
    for name, probs in model_probs.items():
        if probs is None or np.isnan(probs).any():
            continue
        try:
            obs, pred = calibration_curve(labels.astype(int), probs, n_bins=8, strategy="quantile")
            ax.plot(pred, obs, "o-", label=name, color=colors.get(name, "#666"), lw=1.4, ms=4)
        except Exception:
            pass
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability (calibration) plot")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    DL_RESULTS.mkdir(parents=True, exist_ok=True)
    print("[compare] Running full evaluation...")
    results = evaluate_all()

    labels = results["labels_test"]
    rf_prod_p = results["rf_production"]["probs"]
    rf_fair_p = results["rf_fair"]["probs"]

    rows = []
    pr_pts = {}
    model_probs: dict[str, np.ndarray | None] = {
        "RF production": rf_prod_p,
        "RF fair": rf_fair_p,
    }
    per_city_per_model = {}

    # RF production row
    m_rf = results["rf_production"]["metrics_070"]
    rows.append(
        _make_row(
            "RF production",
            "1990–2022 (train+val)",
            "110 features (incl. qualifying_day)",
            "Fixed 0.70",
            m_rf,
            note="Production model; trained on train+val combined.",
        )
    )
    per_city_per_model["RF production"] = results["rf_production"]["per_city"]

    # RF fair row
    m_rf_fair = results["rf_fair"]["metrics_val_opt"]
    rf_fair_t = results["rf_fair"]["threshold"]
    rows.append(
        _make_row(
            "RF fair",
            f"1990–2019 (train only)",
            f"104 features (no qualifying_day, no heatwave_lag1)",
            f"Val-optimal ({rf_fair_t:.2f})",
            m_rf_fair,
            note="Same RF hyperparameters; train-only window to match DL.",
        )
    )
    per_city_per_model["RF fair"] = results["rf_fair"]["per_city"]

    # DL rows
    dl_name_map = {
        "gru_raw_seq": "GRU raw-seq ens",
        "lstm_raw_seq": "LSTM raw-seq ens",
        "gru_feat110_seq": "GRU feat110 ens",
        "gru_raw_seq_focal": "GRU raw-seq focal ens",
    }
    for key, dl_data in results["dl"].items():
        if "error" in dl_data:
            continue
        display = dl_name_map.get(key, key)
        ens_m = dl_data["ensemble"]["metrics_val_opt"]
        ens_thr = dl_data["ensemble"]["threshold_val_opt"]
        mean_f1 = dl_data.get("mean_f1")
        std_f1 = dl_data.get("std_f1")
        mean_pa = dl_data.get("mean_prauc")
        std_pa = dl_data.get("std_prauc")
        cfg = dl_data["input_config"]
        loss_name = dl_data.get("loss", "weighted-bce")
        n_seeds = dl_data["n_seeds_completed"]
        ci = dl_data.get("bootstrap_ci_vs_rf_prod", {})
        ci_fair = dl_data.get("bootstrap_ci_vs_rf_fair", {})
        ci_note = ""
        if ci:
            iz_f1 = ci.get("includes_zero_f1")
            iz_pr = ci.get("includes_zero_prauc")
            verdict_f1 = (
                "competitive, no statistically significant difference"
                if iz_f1
                else "significantly different"
            )
            verdict_pr = (
                "competitive, no statistically significant difference"
                if iz_pr
                else "significantly different"
            )
            ci_note = (
                f"vs RF-prod ΔF1 95% CI [{ci.get('f1_ci_lower')}, {ci.get('f1_ci_upper')}] "
                f"includes_zero={iz_f1} ({verdict_f1}); "
                f"ΔPR-AUC CI [{ci.get('prauc_ci_lower')}, {ci.get('prauc_ci_upper')}] "
                f"includes_zero={iz_pr} ({verdict_pr})"
            )
        if ci_fair:
            iz2_f1 = ci_fair.get("includes_zero_f1")
            iz2_pr = ci_fair.get("includes_zero_prauc")
            verdict2_f1 = (
                "competitive, no statistically significant difference"
                if iz2_f1
                else "significantly different"
            )
            verdict2_pr = (
                "competitive, no statistically significant difference"
                if iz2_pr
                else "significantly different"
            )
            ci_note += (
                f"; vs RF-fair ΔF1 [{ci_fair.get('f1_ci_lower')}, {ci_fair.get('f1_ci_upper')}] "
                f"includes_zero={iz2_f1} ({verdict2_f1}); "
                f"ΔPR-AUC CI [{ci_fair.get('prauc_ci_lower')}, {ci_fair.get('prauc_ci_upper')}] "
                f"includes_zero={iz2_pr} ({verdict2_pr})"
            )
        rows.append(
            _make_row(
                display,
                "1990–2019 (train only)",
                f"{'21 raw+cal' if cfg=='raw-seq' else str(len([e['name'] for e in json.loads((Path(__file__).parents[1]/'models/final/feature_list.json').read_text(encoding='utf-8')) if e['name'] not in {'qualifying_day','heatwave_lag1','city_encoded','latitude','longitude','is_coastal'}]))+' feat'} (no qualifying_day)",
                f"Val-optimal (≈{ens_thr:.2f})",
                ens_m,
                note=f"Ensemble of {n_seeds} seeds; "
                f"loss={loss_name}; "
                f"mean F1={mean_f1}±{std_f1}; "
                f"PR-AUC={mean_pa}±{std_pa}. {ci_note}",
            )
        )
        per_city_per_model[display] = dl_data.get("per_city", {})
        # Collect probs for ensemble (stored in-memory only)
        # Re-extract from per_seed probs if needed; for now use rf as placeholder
        model_probs[display] = dl_data.get("probs")

    # ── Also add per_seed headlines for DL GRU raw-seq ─────────────────────
    for key, dl_data in results["dl"].items():
        if "error" in dl_data or key != "gru_raw_seq":
            continue
        for ps in dl_data.get("per_seed", []):
            m = ps["metrics_val_opt"]
            rows.append(
                _make_row(
                    f"GRU raw-seq seed{ps['seed']}",
                    "1990–2019 (train only)",
                    "21 raw+cal (no qualifying_day)",
                    f"Val-optimal ({ps['threshold_val']:.2f})",
                    m,
                    note=f"Individual seed for variance reporting.",
                )
            )

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_cols = [
        "model",
        "training_window",
        "features",
        "threshold_rule",
        "F1",
        "Precision",
        "Recall",
        "PR-AUC",
        "ROC-AUC",
        "Brier",
        "ECE",
        "note",
    ]
    df = pd.DataFrame(rows, columns=csv_cols)
    df.to_csv(DL_RESULTS / "dl_vs_rf.csv", index=False)
    print(f"[compare] Saved dl_vs_rf.csv")

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_lines = [
        "# Deep Learning vs Random Forest — India Test Set (2023–2025)",
        "",
        "**38 positive heatwave days out of 4 865 test rows (0.78%).**",
        "Ahmedabad and Mumbai: 0 test positives — per-city metrics undefined (n/a).",
        "RF-production was trained on 1990–2022 (train+val combined). "
        "All other models use train only (≤ 2019-12-31) for fairness.",
        "",
        df.drop(columns=["note"]).to_markdown(index=False),
        "",
        "## Notes",
        "- **qualifying_day** excluded from DL and RF-fair: same threshold conditions as the heatwave label (structural correlation).",
        "- **heatwave_lag1** excluded: derived from the label column; would leak label-correlated information across the sequence boundary.",
        "- DL probabilities are not calibrated the same way as the RF. Threshold comparisons are approximate.",
        "- With only 38 test positives, confidence intervals are wide; see bootstrap CIs in dl_vs_rf.json.",
        "- The RF remains the production model for all predictions.",
        "",
    ]
    (DL_RESULTS / "dl_vs_rf.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[compare] Saved dl_vs_rf.md")

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_out = {
        "summary": "India test set 2023-2025, 38 positives / 4865 rows",
        "rows": rows,
        "pr_curves": {},  # filled below
        "bootstrap_cis": {
            k: {
                "vs_rf_prod": v.get("bootstrap_ci_vs_rf_prod"),
                "vs_rf_fair": v.get("bootstrap_ci_vs_rf_fair"),
            }
            for k, v in results["dl"].items()
            if "probs" in v
        },
        "ensemble_diagnostics": {
            key: {
                "loss": value.get("loss", "weighted-bce"),
                "uncertainty": value.get("ensemble", {}).get("uncertainty"),
                "test_ece": value.get("ensemble", {}).get("metrics_val_opt", {}).get("ece"),
            }
            for key, value in results["dl"].items()
            if "probs" in value
        },
        "per_city": {
            name: {city: (m if m else None) for city, m in pc.items()}
            for name, pc in per_city_per_model.items()
        },
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    probs_for_plots = {k: v for k, v in model_probs.items() if v is not None}
    _plot_pr_curves(probs_for_plots, labels, DL_RESULTS / "pr_curves.png", json_out["pr_curves"])
    _plot_per_city(
        [
            {**r, "per_city": per_city_per_model.get(r["model"], {})}
            for r in rows
            if r["model"] in per_city_per_model
        ],
        DL_RESULTS / "per_city_f1.png",
    )
    _plot_calibration(probs_for_plots, labels, DL_RESULTS / "calibration.png")

    (DL_RESULTS / "dl_vs_rf.json").write_text(
        json.dumps(json_out, indent=2, default=str), encoding="utf-8"
    )
    print("[compare] Saved dl_vs_rf.json")
    print("\n[compare] Done. Results in", DL_RESULTS)


if __name__ == "__main__":
    main()
