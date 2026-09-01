"""
eda_climateguard.py
ClimateGuard Phase 4 — Exploratory Data Analysis
================================================
Dataset : data/raw/all_cities_era5_raw.csv
Output  : results/plots/EDA/   (all plots)
          results/eda_summary.txt  (text report)

READ-ONLY: The master raw dataset is never modified by this script.

Run:
    python eda_climateguard.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on headless systems
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
RAW_CSV     = ROOT / "data" / "raw" / "all_cities_era5_raw.csv"
PLOT_DIR    = ROOT / "results" / "plots" / "EDA"
RESULTS_DIR = ROOT / "results"
DOCS_DIR    = ROOT / "docs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── PALETTE ────────────────────────────────────────────────────────────────────
CITY_ORDER  = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
CITY_LABELS = {
    "delhi":     "New Delhi",
    "lucknow":   "Lucknow",
    "nagpur":    "Nagpur",
    "ahmedabad": "Ahmedabad",
    "mumbai":    "Mumbai",
}
CITY_COLORS = {
    "delhi":     "#E63946",
    "lucknow":   "#457B9D",
    "nagpur":    "#2A9D8F",
    "ahmedabad": "#E9C46A",
    "mumbai":    "#9B5DE5",
}
REGION_COLORS = {"plains": "#E76F51", "coastal": "#48CAE4"}

# Heatwave preliminary thresholds
HW_THRESH = {
    "delhi": 40.0, "lucknow": 40.0, "nagpur": 40.0,
    "ahmedabad": 40.0, "mumbai": 37.0,
}

SEASONS = {
    "Winter":   [12, 1, 2],
    "Pre-Monsoon": [3, 4, 5],
    "Monsoon":  [6, 7, 8, 9],
    "Post-Monsoon": [10, 11],
}

WEATHER_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# ── PLOTTING DEFAULTS ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.5,
})

REPORT_LINES = []

def rpt(line=""):
    """Append to in-memory report."""
    REPORT_LINES.append(line)
    print(line)

def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    rpt(f"  [PLOT] {path.relative_to(ROOT)}")

def section(title):
    bar = "=" * 70
    rpt(f"\n{bar}")
    rpt(f"  {title}")
    rpt(bar)

def subsection(title):
    rpt(f"\n--- {title} ---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND INSPECT
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — LOAD AND INSPECT")

df = pd.read_csv(RAW_CSV, parse_dates=["date"])
rpt(f"Loaded: {RAW_CSV.relative_to(ROOT)}")
rpt(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

# Confirm columns and dtypes
subsection("Columns and data types")
for col, dtype in df.dtypes.items():
    rpt(f"  {col:<40} {dtype}")

# City coverage
subsection("City coverage")
city_counts = df.groupby("city_key").size()
for ck in CITY_ORDER:
    n = city_counts.get(ck, 0)
    label = CITY_LABELS.get(ck, ck)
    region = df.loc[df["city_key"] == ck, "region_type"].iloc[0] if n > 0 else "?"
    rpt(f"  {ck:<12} {label:<14} {region:<8} rows={n:,}")

# Date range
subsection("Date range per city")
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]
    rpt(f"  {ck:<12} {sub['date'].min().date()} → {sub['date'].max().date()}  ({len(sub):,} rows)")

# Missing values
subsection("Missing values")
total_miss = df[WEATHER_VARS].isnull().sum().sum()
rpt(f"  Total missing weather values : {total_miss}")
for col in WEATHER_VARS:
    n = df[col].isnull().sum()
    if n > 0:
        rpt(f"  {col:<40} {n:,}  ({n/len(df)*100:.4f}%)")
if total_miss == 0:
    rpt("  All 16 weather variables: 0 missing values. ✓")

# Duplicates
subsection("Duplicate check")
full_dups = df.duplicated().sum()
key_dups  = df.duplicated(subset=["city_key", "date"]).sum()
rpt(f"  Full duplicate rows    : {full_dups}")
rpt(f"  city_key+date dups     : {key_dups}")

# Metadata consistency
subsection("Metadata consistency")
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]
    for col in ["city", "latitude", "longitude", "region_type", "state"]:
        uniq = sub[col].nunique()
        val  = sub[col].iloc[0]
        flag = "✓" if uniq == 1 else "⚠ INCONSISTENT"
        rpt(f"  {ck:<12} {col:<14} = {val}  {flag}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — DATA DICTIONARY")

VAR_META = {
    "temperature_2m_max": {
        "meaning": "Daily maximum air temperature at 2 m above surface",
        "unit": "°C",
        "hw_relevant": True,
    },
    "temperature_2m_min": {
        "meaning": "Daily minimum air temperature at 2 m above surface",
        "unit": "°C",
        "hw_relevant": True,
    },
    "temperature_2m_mean": {
        "meaning": "Daily mean air temperature at 2 m above surface",
        "unit": "°C",
        "hw_relevant": True,
    },
    "apparent_temperature_max": {
        "meaning": "Daily maximum apparent (feels-like) temperature",
        "unit": "°C",
        "hw_relevant": True,
    },
    "apparent_temperature_min": {
        "meaning": "Daily minimum apparent (feels-like) temperature",
        "unit": "°C",
        "hw_relevant": True,
    },
    "apparent_temperature_mean": {
        "meaning": "Daily mean apparent (feels-like) temperature",
        "unit": "°C",
        "hw_relevant": True,
    },
    "precipitation_sum": {
        "meaning": "Total daily precipitation (rain + snow water equivalent)",
        "unit": "mm",
        "hw_relevant": True,
    },
    "rain_sum": {
        "meaning": "Daily liquid rain total",
        "unit": "mm",
        "hw_relevant": False,
    },
    "wind_speed_10m_max": {
        "meaning": "Maximum daily wind speed at 10 m above surface",
        "unit": "km/h",
        "hw_relevant": True,
    },
    "wind_gusts_10m_max": {
        "meaning": "Maximum daily wind gusts at 10 m above surface",
        "unit": "km/h",
        "hw_relevant": False,
    },
    "relative_humidity_2m_max": {
        "meaning": "Daily maximum relative humidity at 2 m",
        "unit": "%",
        "hw_relevant": True,
    },
    "relative_humidity_2m_min": {
        "meaning": "Daily minimum relative humidity at 2 m",
        "unit": "%",
        "hw_relevant": True,
    },
    "relative_humidity_2m_mean": {
        "meaning": "Daily mean relative humidity at 2 m",
        "unit": "%",
        "hw_relevant": True,
    },
    "surface_pressure_mean": {
        "meaning": "Daily mean surface (station) pressure",
        "unit": "hPa",
        "hw_relevant": False,
    },
    "shortwave_radiation_sum": {
        "meaning": "Daily sum of downward shortwave solar radiation at surface",
        "unit": "MJ/m²",
        "hw_relevant": True,
    },
    "et0_fao_evapotranspiration": {
        "meaning": "Daily FAO reference evapotranspiration (water stress proxy)",
        "unit": "mm",
        "hw_relevant": True,
    },
}

dd_rows = []
rpt(f"\n{'Variable':<40} {'Unit':<8} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8} {'Std':>8} {'Missing%':>9}  HW?")
rpt("-" * 110)
for col in WEATHER_VARS:
    s    = df[col].dropna()
    meta = VAR_META.get(col, {})
    row  = {
        "variable":        col,
        "meaning":         meta.get("meaning", ""),
        "unit":            meta.get("unit",    ""),
        "dtype":           str(df[col].dtype),
        "min":             round(float(s.min()), 2),
        "max":             round(float(s.max()), 2),
        "mean":            round(float(s.mean()), 4),
        "median":          round(float(s.median()), 2),
        "std":             round(float(s.std()), 4),
        "missing_pct":     round(df[col].isnull().mean() * 100, 4),
        "hw_relevant":     meta.get("hw_relevant", False),
    }
    dd_rows.append(row)
    hw_flag = "YES" if row["hw_relevant"] else "no"
    rpt(
        f"{col:<40} {row['unit']:<8} {row['min']:>8.2f} {row['max']:>8.2f} "
        f"{row['mean']:>8.2f} {row['median']:>8.2f} {row['std']:>8.2f} "
        f"{row['missing_pct']:>8.2f}%  {hw_flag}"
    )

# Save as JSON for docs
dd_path = RESULTS_DIR / "data_dictionary.json"
with open(dd_path, "w") as f:
    json.dump(dd_rows, f, indent=2)
rpt(f"\n  Saved: {dd_path.relative_to(ROOT)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CITY COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — CITY COMPARISON TABLE")

city_summary = {}
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]
    city_summary[ck] = {
        "city":             CITY_LABELS[ck],
        "region":           sub["region_type"].iloc[0],
        "state":            sub["state"].iloc[0],
        "lat":              sub["latitude"].iloc[0],
        "lon":              sub["longitude"].iloc[0],
        "mean_tmax":        round(sub["temperature_2m_max"].mean(), 2),
        "max_tmax":         round(sub["temperature_2m_max"].max(), 2),
        "mean_tmin":        round(sub["temperature_2m_min"].mean(), 2),
        "min_tmin":         round(sub["temperature_2m_min"].min(), 2),
        "mean_humidity":    round(sub["relative_humidity_2m_mean"].mean(), 2),
        "max_humidity":     round(sub["relative_humidity_2m_max"].max(), 2),
        "mean_precip":      round(sub["precipitation_sum"].mean(), 4),
        "total_precip_mm":  round(sub["precipitation_sum"].sum(), 1),
        "mean_wind":        round(sub["wind_speed_10m_max"].mean(), 2),
        "mean_pressure":    round(sub["surface_pressure_mean"].mean(), 2),
        "mean_app_tmax":    round(sub["apparent_temperature_max"].mean(), 2),
        "mean_radiation":   round(sub["shortwave_radiation_sum"].mean(), 2),
    }

header = (f"{'Metric':<28}" +
          "".join(f"{CITY_LABELS[ck]:>14}" for ck in CITY_ORDER))
rpt(header)
rpt("-" * (28 + 14 * 5))

metrics = [
    ("Mean Tmax (°C)",         "mean_tmax"),
    ("Max Tmax (°C)",          "max_tmax"),
    ("Mean Tmin (°C)",         "mean_tmin"),
    ("Min Tmin (°C)",          "min_tmin"),
    ("Mean Humidity (%)",      "mean_humidity"),
    ("Max Humidity (%)",       "max_humidity"),
    ("Mean Precip (mm/day)",   "mean_precip"),
    ("Total Precip (mm)",      "total_precip_mm"),
    ("Mean Wind (km/h)",       "mean_wind"),
    ("Mean Pressure (hPa)",    "mean_pressure"),
    ("Mean App Tmax (°C)",     "mean_app_tmax"),
    ("Mean Radiation (MJ/m²)", "mean_radiation"),
]
for label, key in metrics:
    row_str = f"{label:<28}"
    for ck in CITY_ORDER:
        row_str += f"{city_summary[ck][key]:>14.2f}"
    rpt(row_str)

# ── City comparison bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("City-Level Weather Comparison — ERA5 1990–2025", fontsize=14, fontweight="bold")

keys_titles_units = [
    ("mean_tmax",      "Mean Daily Tmax",        "°C"),
    ("mean_tmin",      "Mean Daily Tmin",         "°C"),
    ("mean_humidity",  "Mean Relative Humidity",  "%"),
    ("mean_precip",    "Mean Daily Precipitation","mm/day"),
    ("mean_wind",      "Mean Wind Speed (max)",   "km/h"),
    ("mean_app_tmax",  "Mean Apparent Tmax",      "°C"),
]
for ax, (key, title, unit) in zip(axes.flat, keys_titles_units):
    vals   = [city_summary[ck][key] for ck in CITY_ORDER]
    colors = [CITY_COLORS[ck] for ck in CITY_ORDER]
    bars   = ax.bar([CITY_LABELS[ck] for ck in CITY_ORDER], vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
save(fig, "01_city_comparison_bars.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TEMPERATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — TEMPERATURE ANALYSIS")

# ── 4a: Tmax distribution by city ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]["temperature_2m_max"]
    ax.hist(sub, bins=60, alpha=0.55, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], edgecolor="none", density=True)
    # KDE overlay
    kde_x = np.linspace(sub.min(), sub.max(), 400)
    kde   = stats.gaussian_kde(sub)
    ax.plot(kde_x, kde(kde_x), color=CITY_COLORS[ck], linewidth=2)
ax.set_title("Distribution of Daily Maximum Temperature (Tmax) by City — 1990–2025")
ax.set_xlabel("Tmax (°C)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "02_tmax_distribution_by_city.png")

# ── 4b: Tmin distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]["temperature_2m_min"]
    ax.hist(sub, bins=60, alpha=0.55, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], edgecolor="none", density=True)
    kde_x = np.linspace(sub.min(), sub.max(), 400)
    kde   = stats.gaussian_kde(sub)
    ax.plot(kde_x, kde(kde_x), color=CITY_COLORS[ck], linewidth=2)
ax.set_title("Distribution of Daily Minimum Temperature (Tmin) by City — 1990–2025")
ax.set_xlabel("Tmin (°C)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "03_tmin_distribution_by_city.png")

# ── 4c: Mean temperature distribution ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]["temperature_2m_mean"]
    ax.hist(sub, bins=60, alpha=0.55, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], edgecolor="none", density=True)
    kde_x = np.linspace(sub.min(), sub.max(), 400)
    kde   = stats.gaussian_kde(sub)
    ax.plot(kde_x, kde(kde_x), color=CITY_COLORS[ck], linewidth=2)
ax.set_title("Distribution of Daily Mean Temperature by City — 1990–2025")
ax.set_xlabel("T_mean (°C)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "04_tmean_distribution_by_city.png")

# ── 4d: Apparent Tmax distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]["apparent_temperature_max"]
    ax.hist(sub, bins=60, alpha=0.55, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], edgecolor="none", density=True)
    kde_x = np.linspace(sub.min(), sub.max(), 400)
    kde   = stats.gaussian_kde(sub)
    ax.plot(kde_x, kde(kde_x), color=CITY_COLORS[ck], linewidth=2)
ax.set_title("Distribution of Daily Max Apparent (Feels-Like) Temperature by City — 1990–2025")
ax.set_xlabel("Apparent Tmax (°C)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "05_app_tmax_distribution_by_city.png")

# ── 4e: Tmax time series (annual rolling mean per city) ───────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].set_index("date")["temperature_2m_max"]
    # 365-day rolling mean
    rolling = sub.rolling(365, min_periods=300, center=True).mean()
    ax.plot(sub.index, rolling, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], linewidth=1.8)
ax.set_title("Daily Maximum Temperature — 365-Day Rolling Mean by City (1990–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Tmax rolling mean (°C)")
ax.legend(loc="upper left", ncol=2)
plt.tight_layout()
save(fig, "06_tmax_timeseries_rolling.png")

# ── 4f: Tmin time series (annual rolling mean per city) ───────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].set_index("date")["temperature_2m_min"]
    rolling = sub.rolling(365, min_periods=300, center=True).mean()
    ax.plot(sub.index, rolling, color=CITY_COLORS[ck],
            label=CITY_LABELS[ck], linewidth=1.8)
ax.set_title("Daily Minimum Temperature — 365-Day Rolling Mean by City (1990–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Tmin rolling mean (°C)")
ax.legend(loc="upper left", ncol=2)
plt.tight_layout()
save(fig, "07_tmin_timeseries_rolling.png")

# ── 4g: Monthly temperature patterns (median Tmax per city) ───────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
fig.suptitle("Median Daily Tmax by Month — Each City (1990–2025)", fontsize=13, fontweight="bold")
months = range(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for ax, ck in zip(axes, CITY_ORDER):
    sub = df[df["city_key"] == ck].copy()
    sub["month"] = sub["date"].dt.month
    med = sub.groupby("month")["temperature_2m_max"].median()
    ax.bar(list(months), [med.get(m, np.nan) for m in months],
           color=CITY_COLORS[ck], edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(months))
    ax.set_xticklabels(month_labels, rotation=45, fontsize=7)
    ax.set_title(CITY_LABELS[ck])
    ax.set_ylabel("Median Tmax (°C)")
    ax.set_ylim(10, 50)
plt.tight_layout()
save(fig, "08_monthly_tmax_per_city.png")

# ── 4h: Yearly mean Tmax (box) across all cities ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
df["year"] = df["date"].dt.year
yearly = df.groupby(["year", "city_key"])["temperature_2m_max"].mean().reset_index()
for ck in CITY_ORDER:
    sub = yearly[yearly["city_key"] == ck]
    ax.plot(sub["year"], sub["temperature_2m_max"], color=CITY_COLORS[ck],
            alpha=0.85, linewidth=1.6, label=CITY_LABELS[ck])
ax.set_title("Annual Mean Daily Tmax by City (1990–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean Tmax (°C)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "09_annual_mean_tmax_by_city.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — LONG-TERM TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — LONG-TERM TREND ANALYSIS")

trend_results = {}
for ck in CITY_ORDER:
    sub  = df[df["city_key"] == ck].copy()
    sub["year"] = sub["date"].dt.year
    ann  = sub.groupby("year").agg(
        mean_tmax=("temperature_2m_max", "mean"),
        max_tmax=("temperature_2m_max",  "max"),
        mean_tmin=("temperature_2m_min", "mean"),
        min_tmin=("temperature_2m_min",  "min"),
    ).reset_index()

    # Linear regression slope per metric
    trs = {}
    for metric in ["mean_tmax", "max_tmax", "mean_tmin", "min_tmin"]:
        slope, intercept, r, p, se = stats.linregress(ann["year"], ann[metric])
        trs[metric] = {"slope": round(slope, 4), "r2": round(r**2, 4), "p": round(p, 4)}

    trend_results[ck] = {"annual": ann, "trends": trs}

    rpt(f"\n  {CITY_LABELS[ck]}:")
    for metric, t in trs.items():
        sig = "significant (p<0.05)" if t["p"] < 0.05 else "not significant (p≥0.05)"
        rpt(f"    {metric:<14} slope={t['slope']:+.4f}°C/yr  R²={t['r2']:.3f}  {sig}")

# ── Trend plot: 2×2 subplots per city ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Long-Term Annual Temperature Trends (1990–2025)\nObserved trend lines — see Limitations section", fontsize=13, fontweight="bold")

trend_plots = [
    ("mean_tmax",  "Annual Mean Tmax (°C)"),
    ("max_tmax",   "Annual Maximum Tmax (°C)"),
    ("mean_tmin",  "Annual Mean Tmin (°C)"),
    ("min_tmin",   "Annual Minimum Tmin (°C)"),
]
for ax, (metric, ylabel) in zip(axes.flat, trend_plots):
    for ck in CITY_ORDER:
        ann = trend_results[ck]["annual"]
        t   = trend_results[ck]["trends"][metric]
        ax.plot(ann["year"], ann[metric], color=CITY_COLORS[ck],
                alpha=0.6, linewidth=1.2)
        # Trend line
        years = np.array(ann["year"])
        fit   = np.polyval(np.polyfit(years, ann[metric], 1), years)
        ax.plot(years, fit, color=CITY_COLORS[ck], linewidth=2,
                linestyle="--", label=f"{CITY_LABELS[ck]} ({t['slope']:+.3f}°C/yr)")
    ax.set_title(ylabel)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=1)
plt.tight_layout()
save(fig, "10_long_term_temperature_trends.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MONTHLY / SEASONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — MONTHLY / SEASONAL ANALYSIS")

df["month"]  = df["date"].dt.month
df["year"]   = df["date"].dt.year

def get_season(m):
    for s, months in SEASONS.items():
        if m in months:
            return s
    return "Unknown"

df["season"] = df["month"].apply(get_season)

# ── Monthly median Tmax heatmap (city × month) ───────────────────────────────
pivot = df.groupby(["city_key", "month"])["temperature_2m_max"].median().unstack()
pivot = pivot.loc[CITY_ORDER]
pivot.index = [CITY_LABELS[ck] for ck in CITY_ORDER]

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r",
               vmin=10, vmax=48)
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(5))
ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8, color="black" if 20 < val < 42 else "white")
plt.colorbar(im, ax=ax, label="Median Tmax (°C)")
ax.set_title("Median Daily Tmax by City × Month — ERA5 1990–2025")
plt.tight_layout()
save(fig, "11_monthly_tmax_heatmap.png")

# ── Seasonal boxplot — Tmax per city ──────────────────────────────────────────
season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
fig.suptitle("Tmax Distribution by Season — Each City (1990–2025)", fontsize=13, fontweight="bold")
for ax, ck in zip(axes, CITY_ORDER):
    data = [df.loc[(df["city_key"] == ck) & (df["season"] == s), "temperature_2m_max"].values
            for s in season_order]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    season_colors = ["#AED6F1", "#F9E79F", "#82E0AA", "#F0B27A"]
    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(["Win","Pre-M","Mon","Post-M"], rotation=30, fontsize=8)
    ax.set_title(CITY_LABELS[ck])
    ax.set_ylabel("Tmax (°C)" if ck == "delhi" else "")
plt.tight_layout()
save(fig, "12_seasonal_tmax_boxplot.png")

# ── Hottest months per city ────────────────────────────────────────────────────
subsection("Hottest months per city (median Tmax)")
for ck in CITY_ORDER:
    sub  = df[df["city_key"] == ck].groupby("month")["temperature_2m_max"].median()
    top3 = sub.nlargest(3)
    names = [month_labels[m-1] for m in top3.index]
    rpt(f"  {CITY_LABELS[ck]:<14}: {', '.join([f'{n} ({v:.1f}°C)' for n,v in zip(names,top3.values)])}")

# ── Seasonal mean Tmax per city (bar chart) ────────────────────────────────────
sea_means = df.groupby(["city_key", "season"])["temperature_2m_max"].mean().unstack()
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(season_order))
width = 0.15
for i, ck in enumerate(CITY_ORDER):
    vals = [sea_means.loc[ck].get(s, np.nan) for s in season_order]
    ax.bar(x + i * width, vals, width, label=CITY_LABELS[ck],
           color=CITY_COLORS[ck], edgecolor="white", linewidth=0.6)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(season_order)
ax.set_ylabel("Mean Tmax (°C)")
ax.set_title("Seasonal Mean Daily Tmax by City — ERA5 1990–2025")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
save(fig, "13_seasonal_mean_tmax_by_city.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — HUMIDITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — HUMIDITY ANALYSIS")

# ── Humidity distributions by city ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Relative Humidity Distributions by City — ERA5 1990–2025", fontsize=13, fontweight="bold")
for ax, col, title in zip(axes,
    ["relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean"],
    ["Daily Max RH (%)", "Daily Min RH (%)", "Daily Mean RH (%)"]):
    for ck in CITY_ORDER:
        sub = df[df["city_key"] == ck][col]
        ax.hist(sub, bins=50, alpha=0.5, color=CITY_COLORS[ck],
                label=CITY_LABELS[ck], density=True, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "14_humidity_distributions.png")

# ── Monthly mean humidity (mean RH) heatmap ───────────────────────────────────
pivot_rh = df.groupby(["city_key", "month"])["relative_humidity_2m_mean"].median().unstack()
pivot_rh = pivot_rh.loc[CITY_ORDER]
pivot_rh.index = [CITY_LABELS[ck] for ck in CITY_ORDER]

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(pivot_rh.values, aspect="auto", cmap="YlGnBu", vmin=20, vmax=100)
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(5))
ax.set_yticklabels(pivot_rh.index)
for i in range(pivot_rh.shape[0]):
    for j in range(pivot_rh.shape[1]):
        val = pivot_rh.iloc[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                color="black" if val < 70 else "white")
plt.colorbar(im, ax=ax, label="Median Mean RH (%)")
ax.set_title("Median Daily Mean Relative Humidity by City × Month — ERA5 1990–2025")
plt.tight_layout()
save(fig, "15_monthly_humidity_heatmap.png")

# ── Temperature vs Humidity scatter (subsampled) ──────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
fig.suptitle("Daily Tmax vs Mean Relative Humidity (random sample 2,000 days per city)", fontsize=12)
for ax, ck in zip(axes, CITY_ORDER):
    sub = df[df["city_key"] == ck][["temperature_2m_max", "relative_humidity_2m_mean"]].dropna()
    samp = sub.sample(min(2000, len(sub)), random_state=42)
    ax.scatter(samp["temperature_2m_max"], samp["relative_humidity_2m_mean"],
               alpha=0.2, s=8, color=CITY_COLORS[ck])
    # Pearson r
    r, p = stats.pearsonr(sub["temperature_2m_max"], sub["relative_humidity_2m_mean"])
    ax.set_title(f"{CITY_LABELS[ck]}\nr={r:.2f}")
    ax.set_xlabel("Tmax (°C)")
    ax.set_ylabel("Mean RH (%)" if ck == "delhi" else "")
plt.tight_layout()
save(fig, "16_tmax_vs_humidity_scatter.png")

# ── Report humidity stats per city ────────────────────────────────────────────
subsection("Humidity summary per city")
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]
    rpt(f"  {CITY_LABELS[ck]:<14} "
        f"RH_mean={sub['relative_humidity_2m_mean'].mean():.1f}%  "
        f"RH_max_p95={sub['relative_humidity_2m_max'].quantile(0.95):.0f}%  "
        f"RH_min_p05={sub['relative_humidity_2m_min'].quantile(0.05):.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PRECIPITATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 8 — PRECIPITATION ANALYSIS")

subsection("Zero-rain frequency per city")
for ck in CITY_ORDER:
    sub    = df[df["city_key"] == ck]["precipitation_sum"]
    zero   = (sub == 0).sum()
    total  = len(sub)
    rpt(f"  {CITY_LABELS[ck]:<14} zero-rain days={zero:,} / {total:,} ({zero/total*100:.1f}%)")

# ── Log-scale precipitation histogram ─────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Daily Precipitation Distribution — Log Scale (rain days only)", fontsize=13)
for ax, ck in zip(axes, CITY_ORDER):
    sub = df.loc[(df["city_key"] == ck) & (df["precipitation_sum"] > 0.1), "precipitation_sum"]
    ax.hist(sub, bins=50, color=CITY_COLORS[ck], edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_title(CITY_LABELS[ck])
    ax.set_xlabel("Precip (mm)")
    ax.set_ylabel("Count (log)" if ck == "delhi" else "")
    ax.text(0.97, 0.95, f"median={sub.median():.1f}mm\np99={sub.quantile(0.99):.0f}mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(facecolor="white", alpha=0.7, pad=2))
plt.tight_layout()
save(fig, "17_precipitation_histogram_logscale.png")

# ── Monthly precipitation heatmap ─────────────────────────────────────────────
pivot_pr = df.groupby(["city_key", "month"])["precipitation_sum"].mean().unstack()
pivot_pr = pivot_pr.loc[CITY_ORDER]
pivot_pr.index = [CITY_LABELS[ck] for ck in CITY_ORDER]

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(pivot_pr.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(5))
ax.set_yticklabels(pivot_pr.index)
for i in range(pivot_pr.shape[0]):
    for j in range(pivot_pr.shape[1]):
        val = pivot_pr.iloc[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="Mean daily precip (mm)")
ax.set_title("Mean Daily Precipitation by City × Month — ERA5 1990–2025")
plt.tight_layout()
save(fig, "18_monthly_precip_heatmap.png")

# ── Annual total precipitation ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].groupby("year")["precipitation_sum"].sum()
    ax.plot(sub.index, sub.values, color=CITY_COLORS[ck],
            alpha=0.75, linewidth=1.5, label=CITY_LABELS[ck])
ax.set_title("Annual Total Precipitation by City (1990–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Annual precipitation (mm)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "19_annual_total_precipitation.png")

# ── Monthly precip boxplot (all cities combined, by month) ────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
data_by_month = [df.loc[df["month"] == m, "precipitation_sum"].values for m in range(1, 13)]
bp = ax.boxplot(data_by_month, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))
rain_palette = plt.cm.Blues(np.linspace(0.3, 0.9, 12))
for patch, color in zip(bp["boxes"], rain_palette):
    patch.set_facecolor(color)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_title("Daily Precipitation by Month — All Cities Combined (1990–2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Daily precipitation (mm)")
plt.tight_layout()
save(fig, "20_monthly_precip_boxplot_all_cities.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — WIND / PRESSURE / RADIATION
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 9 — WIND / PRESSURE / RADIATION / ET0")

# ── Wind speed distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Wind Speed Distributions by City — ERA5 1990–2025")
for ax, col, title in zip(axes,
    ["wind_speed_10m_max", "wind_gusts_10m_max"],
    ["Max Wind Speed (km/h)", "Max Wind Gusts (km/h)"]):
    for ck in CITY_ORDER:
        sub = df[df["city_key"] == ck][col]
        ax.hist(sub, bins=60, alpha=0.5, density=True, color=CITY_COLORS[ck],
                label=CITY_LABELS[ck], edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("km/h")
    ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save(fig, "21_wind_distributions.png")

# ── Surface pressure by city (monthly) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].groupby("month")["surface_pressure_mean"].mean()
    ax.plot(list(months), [sub.get(m, np.nan) for m in months],
            marker="o", markersize=5, color=CITY_COLORS[ck], label=CITY_LABELS[ck])
ax.set_xticks(list(months))
ax.set_xticklabels(month_labels)
ax.set_title("Monthly Mean Surface Pressure by City — ERA5 1990–2025")
ax.set_xlabel("Month")
ax.set_ylabel("Surface pressure (hPa)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "22_monthly_pressure_by_city.png")

# ── Shortwave radiation monthly ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].groupby("month")["shortwave_radiation_sum"].mean()
    ax.plot(list(months), [sub.get(m, np.nan) for m in months],
            marker="o", markersize=5, color=CITY_COLORS[ck], label=CITY_LABELS[ck])
ax.set_xticks(list(months))
ax.set_xticklabels(month_labels)
ax.set_title("Monthly Mean Shortwave Radiation by City — ERA5 1990–2025")
ax.set_xlabel("Month")
ax.set_ylabel("Shortwave radiation (MJ/m²)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "23_monthly_radiation_by_city.png")

# ── ET0 evapotranspiration monthly ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].groupby("month")["et0_fao_evapotranspiration"].mean()
    ax.plot(list(months), [sub.get(m, np.nan) for m in months],
            marker="o", markersize=5, color=CITY_COLORS[ck], label=CITY_LABELS[ck])
ax.set_xticks(list(months))
ax.set_xticklabels(month_labels)
ax.set_title("Monthly Mean Reference Evapotranspiration (ET₀) by City — ERA5 1990–2025")
ax.set_xlabel("Month")
ax.set_ylabel("ET₀ (mm/day)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "24_monthly_et0_by_city.png")

# ── Wind vs Temperature scatter ────────────────────────────────────────────────
subsection("Wind and pressure stats per city")
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck]
    rpt(f"  {CITY_LABELS[ck]:<14} "
        f"wind_mean={sub['wind_speed_10m_max'].mean():.1f}km/h  "
        f"pressure_mean={sub['surface_pressure_mean'].mean():.1f}hPa  "
        f"radiation_mean={sub['shortwave_radiation_sum'].mean():.2f}MJ/m²  "
        f"et0_mean={sub['et0_fao_evapotranspiration'].mean():.2f}mm")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 10 — CORRELATION ANALYSIS")

# ── Full correlation matrix heatmap ───────────────────────────────────────────
corr = df[WEATHER_VARS].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle mask off
sns.heatmap(
    corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.4, linecolor="white",
    square=True,
    xticklabels=[c.replace("_", "\n") for c in WEATHER_VARS],
    yticklabels=[c.replace("_", "\n") for c in WEATHER_VARS],
)
ax.set_title("Pearson Correlation Matrix — All 16 Weather Variables (All Cities Combined)", pad=14)
plt.tight_layout()
save(fig, "25_correlation_matrix.png")

# Report strong correlations
subsection("Strong correlations (|r| > 0.7)")
for i, c1 in enumerate(WEATHER_VARS):
    for j, c2 in enumerate(WEATHER_VARS):
        if j <= i:
            continue
        r = corr.loc[c1, c2]
        if abs(r) > 0.7:
            rpt(f"  {c1:<35} ↔  {c2:<35}  r={r:+.3f}")

# ── Per-city correlation of Tmax with other vars ──────────────────────────────
subsection("Tmax correlation with other variables per city")
non_temp = [c for c in WEATHER_VARS if "temperature" not in c and "apparent" not in c]
rpt(f"  {'Variable':<35} " + "".join(f"{CITY_LABELS[ck]:>14}" for ck in CITY_ORDER))
rpt("  " + "-" * (35 + 14 * 5))
for col in non_temp:
    row_str = f"  {col:<35}"
    for ck in CITY_ORDER:
        sub = df[df["city_key"] == ck][["temperature_2m_max", col]].dropna()
        r, _ = stats.pearsonr(sub["temperature_2m_max"], sub[col])
        row_str += f"{r:>+14.3f}"
    rpt(row_str)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — EXTREME TEMPERATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 11 — EXTREME TEMPERATURE (PRELIMINARY THRESHOLD ANALYSIS)")
rpt("  NOTE: These thresholds are PRELIMINARY only. Final IMD labels = Phase 6.")

extreme_results = {}
for ck in CITY_ORDER:
    sub      = df[df["city_key"] == ck].copy().sort_values("date").reset_index(drop=True)
    thresh   = HW_THRESH[ck]
    exceed   = sub["temperature_2m_max"] >= thresh
    n_exceed = exceed.sum()
    pct      = n_exceed / len(sub) * 100
    hottest_idx = sub["temperature_2m_max"].idxmax()
    hottest_date = sub.loc[hottest_idx, "date"]
    hottest_val  = sub.loc[hottest_idx, "temperature_2m_max"]

    # Max consecutive exceedance days
    consec_max = 0
    consec_cur = 0
    for e in exceed:
        if e:
            consec_cur += 1
            consec_max = max(consec_max, consec_cur)
        else:
            consec_cur = 0

    extreme_results[ck] = {
        "threshold": thresh,
        "n_exceed":  int(n_exceed),
        "pct":       round(pct, 2),
        "hottest_date": str(hottest_date.date()),
        "hottest_val":  float(hottest_val),
        "consec_max":   consec_max,
        "annual":       sub.groupby("year").apply(lambda x: (x["temperature_2m_max"] >= thresh).sum()).to_dict(),
        "monthly":      sub.groupby("month").apply(lambda x: (x["temperature_2m_max"] >= thresh).sum()).to_dict(),
    }

    rpt(f"\n  {CITY_LABELS[ck]} (threshold ≥{thresh}°C):")
    rpt(f"    Exceedance days   : {n_exceed:,} / {len(sub):,}  ({pct:.2f}%)")
    rpt(f"    Hottest day       : {hottest_date.date()}  →  {hottest_val:.1f}°C")
    rpt(f"    Max consecutive   : {consec_max} days")

# ── Annual exceedance days per city ───────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
fig.suptitle("Annual Count of Threshold-Exceeding Tmax Days — Preliminary Only\n(Delhi/Lucknow/Nagpur/Ahmedabad ≥40°C | Mumbai ≥37°C)", fontsize=12, fontweight="bold")
for ax, ck in zip(axes, CITY_ORDER):
    ann_data = extreme_results[ck]["annual"]
    years    = sorted(ann_data.keys())
    counts   = [ann_data[y] for y in years]
    ax.bar(years, counts, color=CITY_COLORS[ck], edgecolor="none", alpha=0.85)
    # trend line
    slope, intercept, _, _, _ = stats.linregress(years, counts)
    fit = [slope * y + intercept for y in years]
    ax.plot(years, fit, "k--", linewidth=1.5, alpha=0.7)
    ax.set_title(CITY_LABELS[ck])
    ax.set_xlabel("Year")
    ax.set_ylabel("Days ≥ threshold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
plt.tight_layout()
save(fig, "26_annual_exceedance_days.png")

# ── Monthly distribution of exceedance days ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(12)
width = 0.15
for i, ck in enumerate(CITY_ORDER):
    monthly = extreme_results[ck]["monthly"]
    vals    = [monthly.get(m, 0) for m in range(1, 13)]
    ax.bar(x + i * width, vals, width, label=CITY_LABELS[ck],
           color=CITY_COLORS[ck], edgecolor="white", linewidth=0.5)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(month_labels)
ax.set_title("Monthly Distribution of Threshold-Exceeding Tmax Days (1990–2025)\n(Preliminary — not final IMD heatwave labels)")
ax.set_xlabel("Month")
ax.set_ylabel("Total days ≥ threshold (all years)")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
save(fig, "27_monthly_exceedance_days.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 12 — OUTLIER ANALYSIS")
rpt("  NOTE: No physical-range violations were found in Step 1.")
rpt("  Statistical outliers (IQR) are documented here. They are NOT removed.")
rpt("  Genuine extreme temperatures may be critical for heatwave detection.")

temp_vars = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
             "apparent_temperature_max"]

subsection("IQR-based outlier counts per city per temp variable")
rpt(f"  {'Variable':<32} {'City':<14} {'N_low':>7} {'N_high':>7} {'%_high':>8}  Interpretation")
rpt("  " + "-" * 90)
for col in temp_vars:
    for ck in CITY_ORDER:
        sub = df.loc[df["city_key"] == ck, col].dropna()
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_lo   = int((sub < lo).sum())
        n_hi   = int((sub > hi).sum())
        pct_hi = n_hi / len(sub) * 100
        interp = "Extreme hot — potential heatwave days" if "max" in col and pct_hi > 0 else "Normal range variation"
        rpt(f"  {col:<32} {CITY_LABELS[ck]:<14} {n_lo:>7} {n_hi:>7} {pct_hi:>7.2f}%  {interp}")

# ── Boxplot of Tmax with outlier overlay ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
city_data = [df.loc[df["city_key"] == ck, "temperature_2m_max"].values for ck in CITY_ORDER]
bp = ax.boxplot(city_data, patch_artist=True,
                medianprops=dict(color="black", linewidth=2.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.25))
for patch, ck in zip(bp["boxes"], CITY_ORDER):
    patch.set_facecolor(CITY_COLORS[ck])
    patch.set_alpha(0.7)
for ck_idx, ck in enumerate(CITY_ORDER):
    thresh = HW_THRESH[ck]
    ax.axhline(thresh, color=CITY_COLORS[ck], linestyle=":", linewidth=1.0, alpha=0.7)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([CITY_LABELS[ck] for ck in CITY_ORDER], rotation=10)
ax.set_title("Tmax Distribution with IQR Outliers — Points above dotted line = threshold days")
ax.set_ylabel("Tmax (°C)")
plt.tight_layout()
save(fig, "28_tmax_boxplot_outliers.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — COASTAL vs PLAINS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 13 — COASTAL (MUMBAI) vs PLAINS ANALYSIS")

plains_cities = [ck for ck in CITY_ORDER if ck != "mumbai"]
df["region_type_label"] = df["region_type"].map(lambda x: x.title())

subsection("Key metric comparison — Plains average vs Mumbai (Coastal)")
plains_df = df[df["city_key"].isin(plains_cities)]
mumbai_df = df[df["city_key"] == "mumbai"]

comp_vars = {
    "Tmax mean (°C)":         "temperature_2m_max",
    "Tmax std (°C)":          "temperature_2m_max",
    "Tmin mean (°C)":         "temperature_2m_min",
    "Mean RH (%)" :           "relative_humidity_2m_mean",
    "Mean Precip (mm/day)":   "precipitation_sum",
    "Mean Wind (km/h)":       "wind_speed_10m_max",
    "Mean Pressure (hPa)":    "surface_pressure_mean",
}
stat_map = {
    "Tmax mean (°C)":       "mean",
    "Tmax std (°C)":        "std",
    "Tmin mean (°C)":       "mean",
    "Mean RH (%)" :         "mean",
    "Mean Precip (mm/day)": "mean",
    "Mean Wind (km/h)":     "mean",
    "Mean Pressure (hPa)":  "mean",
}
rpt(f"  {'Metric':<25} {'Plains avg':>14} {'Mumbai':>14}")
rpt("  " + "-" * 55)
for label, col in comp_vars.items():
    fn = stat_map[label]
    p_val = getattr(plains_df[col], fn)()
    m_val = getattr(mumbai_df[col], fn)()
    rpt(f"  {label:<25} {p_val:>14.2f} {m_val:>14.2f}")

# ── Violin plots: Plains vs Coastal ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Plains Cities vs Coastal (Mumbai) — Distribution Comparison", fontsize=13, fontweight="bold")

for ax, (col, title) in zip(axes, [
    ("temperature_2m_max",      "Daily Tmax (°C)"),
    ("relative_humidity_2m_mean", "Mean Relative Humidity (%)"),
    ("precipitation_sum",        "Daily Precipitation (mm)"),
]):
    plot_df = df[["city_key", "region_type", col]].copy()
    plot_df["region"] = plot_df["city_key"].map(
        lambda ck: "Coastal (Mumbai)" if ck == "mumbai" else "Plains"
    )
    plot_df_clean = plot_df.dropna(subset=[col])
    # Cap precipitation at 95th pct for readability
    if col == "precipitation_sum":
        cap = plot_df_clean[col].quantile(0.95)
        plot_df_clean = plot_df_clean[plot_df_clean[col] <= cap]

    groups     = ["Plains", "Coastal (Mumbai)"]
    group_data = [plot_df_clean.loc[plot_df_clean["region"] == g, col].values for g in groups]
    vp = ax.violinplot(group_data, positions=[1, 2], showmedians=True, showextrema=True)
    for pc, color in zip(vp["bodies"], [REGION_COLORS["plains"], REGION_COLORS["coastal"]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(title)
plt.tight_layout()
save(fig, "29_coastal_vs_plains_violin.png")

# ── Seasonal temperature range comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
season_order_local = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
x = np.arange(len(season_order_local))
width = 0.15
for i, ck in enumerate(CITY_ORDER):
    vals = []
    for s in season_order_local:
        sub = df.loc[(df["city_key"] == ck) & (df["season"] == s), "temperature_2m_max"]
        vals.append(sub.mean())
    ls = "--" if ck == "mumbai" else "-"
    lw = 2.5 if ck == "mumbai" else 1.5
    ax.plot(x, vals, marker="o", linestyle=ls, linewidth=lw,
            color=CITY_COLORS[ck], label=CITY_LABELS[ck] + (" (coastal)" if ck == "mumbai" else ""))
ax.set_xticks(x)
ax.set_xticklabels(season_order_local)
ax.set_title("Seasonal Mean Tmax — Plains vs Coastal (1990–2025)")
ax.set_ylabel("Mean Tmax (°C)")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "30_seasonal_coastal_vs_plains.png")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
section("EDA COMPLETE — WRITING TEXT REPORT")

report_path = RESULTS_DIR / "eda_summary.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT_LINES))
rpt(f"  Report saved: {report_path.relative_to(ROOT)}")

# Save extreme results as JSON for docs use
er_path = RESULTS_DIR / "extreme_analysis.json"
er_save = {}
for ck, v in extreme_results.items():
    er_save[ck] = {k: val for k, val in v.items() if k != "annual" and k != "monthly"}
    er_save[ck]["annual"]  = {str(k): v for k, v in extreme_results[ck]["annual"].items()}
    er_save[ck]["monthly"] = {str(k): v for k, v in extreme_results[ck]["monthly"].items()}
with open(er_path, "w") as f:
    json.dump(er_save, f, indent=2)

# Save city summary — convert numpy types for JSON serialisation
def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

cs_path = RESULTS_DIR / "city_summary.json"
cs_safe = json.loads(json.dumps(city_summary, default=_json_safe))
with open(cs_path, "w") as f:
    json.dump(cs_safe, f, indent=2)

# Save trend results
tr_path = RESULTS_DIR / "trend_results.json"
tr_save = {ck: v["trends"] for ck, v in trend_results.items()}
with open(tr_path, "w") as f:
    json.dump(tr_save, f, indent=2)

rpt(f"\nAll plots saved to: results/plots/EDA/")
rpt("EDA script complete.")
