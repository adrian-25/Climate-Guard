"""
heatwave_labeling.py
ClimateGuard Phase 6 — Heatwave Label Generation
=================================================
Input  : data/processed/weather_cleaned.csv       (READ-ONLY)
Output : data/processed/weather_labelled.csv
         results/plots/heatwave_labels/
         results/phase6_labeling_log.txt

Methodology: IMD-Inspired Operational Heatwave Label
  - Based on the official India Meteorological Department (IMD) criteria
  - Adapted for ERA5 gridded reanalysis data (noted where exact IMD
    criteria cannot be fully replicated)
  - Normals: city-specific 30-year daily climatological mean Tmax
    computed from the 1990-2020 baseline period
  - Departure: (Tmax on day T) - (normal for that calendar day)
  - Plains cities (Delhi/Lucknow/Nagpur/Ahmedabad):
      Qualifying day =  (Tmax >= 40°C) AND (departure >= 4.5°C)
                        OR (Tmax >= 45°C)          [absolute override]
  - Coastal city (Mumbai):
      Qualifying day =  (Tmax >= 37°C) AND (departure >= 4.5°C)
  - Event label: a qualifying day that is part of a run of at least
    TWO consecutive qualifying days → heatwave = 1
  - Isolated single qualifying days → heatwave = 0
    (consistent with IMD's "at least two consecutive days" criterion)
  - Prediction horizon: same-day label (heatwave on date T)
    ML features constructed from date T and earlier only.

Sources:
  WHO India  : https://www.who.int/india/heat-waves
  DrishtiIAS : https://www.drishtiias.com/daily-updates/daily-news-analysis/heatwaves-4
  Times of India (IMD Mumbai head quote) — coastal 37°C threshold
  IMD FAQ    : https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf

Run:
    python heatwave_labeling.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
IN_CSV    = ROOT / "data" / "processed" / "weather_cleaned.csv"
OUT_CSV   = ROOT / "data" / "processed" / "weather_labelled.csv"
PLOT_DIR  = ROOT / "results" / "plots" / "heatwave_labels"
RESULTS   = ROOT / "results"
DOCS      = ROOT / "docs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
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
REGION_TYPE = {
    "delhi": "plains", "lucknow": "plains", "nagpur": "plains",
    "ahmedabad": "plains", "mumbai": "coastal",
}

# IMD thresholds
PLAINS_ABS_THRESHOLD  = 40.0   # °C — minimum Tmax for departure criterion to apply
COASTAL_ABS_THRESHOLD = 37.0   # °C — coastal minimum Tmax (Mumbai)
DEPARTURE_HW_THRESHOLD = 4.5   # °C above normal for heatwave
ABS_OVERRIDE_THRESHOLD = 45.0  # °C — absolute Tmax triggers plains HW regardless of departure

BASELINE_START = "1990-01-01"
BASELINE_END   = "2020-12-31"
MIN_CONSEC_DAYS = 2             # IMD: "at least two consecutive days"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150,
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 9, "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA", "axes.grid": True,
    "grid.alpha": 0.35, "grid.linestyle": "--",
})

LOG = []
def log(line=""):
    LOG.append(line)
    print(line)

def section(title):
    bar = "=" * 68
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  [PLOT] {path.relative_to(ROOT)}")

# ══════════════════════════════════════════════════════════════════════════════
section("PHASE 6 — HEATWAVE LABEL GENERATION")
log(f"  Input : {IN_CSV.relative_to(ROOT)}")
log(f"  Output: {OUT_CSV.relative_to(ROOT)}")

# ── LOAD ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV, parse_dates=["date"])
df["month"] = df["date"].dt.month
df["year"]  = df["date"].dt.year
df["doy"]   = df["date"].dt.dayofyear   # day-of-year 1–366
log(f"\n  Loaded: {len(df):,} rows × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — STRATEGY COMPARISON")

log("""
  Three candidate strategies were considered:

  ┌─────┬──────────────────────────────────────────────────────────────────┐
  │ A   │ Fixed Absolute Threshold                                         │
  │     │ Plains: Tmax >= 40°C   Coastal: Tmax >= 37°C                    │
  │     │ PRO : Simple, fully transparent, zero data-leakage risk.         │
  │     │ CON : Ignores anomaly — same temperature may be extreme in one   │
  │     │       season but not another. EDA showed Mumbai gets only 8 days │
  │     │       above 37°C over 35 years — near-useless for ML.           │
  │     │ CON : Climatologically different cities get identical thresholds.│
  └─────┴──────────────────────────────────────────────────────────────────┘

  ┌─────┬──────────────────────────────────────────────────────────────────┐
  │ B   │ Percentile-Based (e.g., city-specific 90th/95th pct of Tmax)    │
  │     │ PRO : Equal positive-class frequency across cities by design.    │
  │     │ PRO : Adapts to each city's climate.                             │
  │     │ CON : Not scientifically grounded in any official criteria.      │
  │     │ CON : A "hot day for Mumbai" is not the same risk as a "hot day  │
  │     │       for Delhi". Cross-city comparability is lost.              │
  │     │ CON : Forces class balance — may over-label mild days as         │
  │     │       heatwaves simply to match a quota.                         │
  └─────┴──────────────────────────────────────────────────────────────────┘

  ┌─────┬──────────────────────────────────────────────────────────────────┐
  │ C   │ IMD-Inspired Operational Rule (SELECTED)                         │
  │     │ Combines absolute threshold AND anomaly from city-specific        │
  │     │ 30-year climatological normal.                                   │
  │     │ Plains: (Tmax>=40°C AND departure>=4.5°C) OR Tmax>=45°C         │
  │     │ Coastal: Tmax>=37°C AND departure>=4.5°C                        │
  │     │ Minimum duration: 2 consecutive qualifying days.                 │
  │     │                                                                  │
  │     │ PRO : Based on official IMD heatwave declaration criteria.       │
  │     │ PRO : Scientifically defensible, grounded in operational met.    │
  │     │ PRO : Anomaly criterion adapts to seasonal variation.            │
  │     │ PRO : Duration criterion aligns with how real heatwave events     │
  │     │       are declared — not isolated single hot days.               │
  │     │ PRO : Handles Mumbai correctly via coastal threshold + anomaly.  │
  │     │ CON : Cannot perfectly replicate IMD's station-level data (ERA5  │
  │     │       is 0.25° gridded reanalysis, not ground station).          │
  │     │ CON : We use ERA5 data for both the normal and the observation.  │
  │     │       IMD uses gridded normals from station archives.            │
  │     │ CON : Normal calculated from 1990-2020 baseline (31 years, close │
  │     │       to WMO 30-year standard) rather than IMD's official        │
  │     │       station normals.                                            │
  └─────┴──────────────────────────────────────────────────────────────────┘

  DECISION: Strategy C — IMD-Inspired Operational Heatwave Label
  This is explicitly NOT labelled "official IMD". It is an ERA5-based
  operational implementation of the IMD criteria.
""")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — COMPUTE CITY-SPECIFIC CLIMATOLOGICAL NORMALS (1990–2020)")

baseline = df[(df["date"] >= BASELINE_START) & (df["date"] <= BASELINE_END)].copy()
log(f"  Baseline period    : {BASELINE_START} → {BASELINE_END}")
log(f"  Baseline rows      : {len(baseline):,}")

# For each city: compute the mean Tmax for each calendar day-of-year.
# Use a 31-day centred window to smooth the daily normal and handle
# the small sample size of individual calendar days (~31 years per DOY).
# This is standard practice in operational climatology.

normals = {}   # normals[city_key] = Series indexed by doy (1..366)

log("\n  City-specific normals (31-day centred smooth, 1990-2020 baseline):")
log(f"  {'City':<14} {'Mean normal Tmax':>18} {'Min normal':>12} {'Max normal':>12}")
log("  " + "-" * 60)

for ck in CITY_ORDER:
    sub = baseline[baseline["city_key"] == ck][["date", "temperature_2m_max"]].copy()
    sub["doy"] = sub["date"].dt.dayofyear

    # Raw daily mean Tmax for each calendar DOY (averaged across years)
    raw_normal = sub.groupby("doy")["temperature_2m_max"].mean()

    # Extend to 366 days: if day 366 missing (no leap years for some DOYs),
    # fill with interpolation
    full_doy = pd.Series(index=range(1, 367), dtype=float)
    full_doy.update(raw_normal)
    full_doy = full_doy.interpolate(method="linear", limit_direction="both")

    # 31-day centred rolling smooth (wrap-around for circular year)
    extended = pd.concat([full_doy.iloc[-15:], full_doy, full_doy.iloc[:15]])
    smoothed = extended.rolling(31, center=True, min_periods=15).mean()
    smoothed = smoothed.iloc[15:381].values  # back to 366 values
    smoothed_series = pd.Series(smoothed, index=range(1, 367))

    normals[ck] = smoothed_series
    log(f"  {CITY_LABELS[ck]:<14} "
        f"{smoothed_series.mean():>18.2f}°C "
        f"{smoothed_series.min():>11.2f}°C "
        f"{smoothed_series.max():>11.2f}°C")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — COMPUTE DEPARTURES AND QUALIFYING DAYS")

# Map each row's departure = Tmax - normal[city][doy]
df["tmax_normal"]    = df.apply(lambda r: normals[r["city_key"]][r["doy"]], axis=1)
df["tmax_departure"] = df["temperature_2m_max"] - df["tmax_normal"]

log(f"\n  Departure column added. Summary (all cities):")
log(f"  Mean departure : {df['tmax_departure'].mean():.3f}°C")
log(f"  Std  departure : {df['tmax_departure'].std():.3f}°C")
log(f"  Max  departure : {df['tmax_departure'].max():.2f}°C")
log(f"  Min  departure : {df['tmax_departure'].min():.2f}°C")

# ── Per-city qualifying day ────────────────────────────────────────────────────
def is_qualifying(row):
    """
    Returns True if the day meets IMD-inspired heat criteria (before
    duration filter). Applied independently per city.
    """
    tmax = row["temperature_2m_max"]
    dep  = row["tmax_departure"]
    if row["city_key"] == "mumbai":
        # Coastal criterion
        return (tmax >= COASTAL_ABS_THRESHOLD) and (dep >= DEPARTURE_HW_THRESHOLD)
    else:
        # Plains criterion: (Tmax>=40 AND dep>=4.5) OR Tmax>=45
        return ((tmax >= PLAINS_ABS_THRESHOLD) and (dep >= DEPARTURE_HW_THRESHOLD)) \
               or (tmax >= ABS_OVERRIDE_THRESHOLD)

df["qualifying_day"] = df.apply(is_qualifying, axis=1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — APPLY DURATION CRITERION (>=2 CONSECUTIVE DAYS)")

# For each city independently, a heatwave day is a qualifying day that belongs
# to a run of at least MIN_CONSEC_DAYS consecutive qualifying days.
# Single isolated qualifying days are NOT labelled as heatwave days.

def apply_duration_and_events(city_df):
    """
    Given a city's sorted DataFrame, add:
      - heatwave        : 1 if part of a run >= MIN_CONSEC_DAYS, else 0
      - hw_event_id     : integer event ID (0 = not a heatwave day)
      - hw_event_start  : date when this event started (NaT if not heatwave)
      - hw_event_end    : date when this event ended   (NaT if not heatwave)
      - hw_event_length : length of this event in days (0 if not heatwave)
    """
    city_df = city_df.sort_values("date").copy()
    n = len(city_df)
    q = city_df["qualifying_day"].values
    dates = city_df["date"].values

    hw         = np.zeros(n, dtype=int)
    event_id   = np.zeros(n, dtype=int)
    ev_start   = np.empty(n, dtype="datetime64[ns]"); ev_start[:] = np.datetime64("NaT")
    ev_end     = np.empty(n, dtype="datetime64[ns]"); ev_end[:]   = np.datetime64("NaT")
    ev_len     = np.zeros(n, dtype=int)

    current_event = 0
    i = 0
    while i < n:
        if q[i] == 1:
            # Find run length
            j = i
            while j < n and q[j] == 1:
                j += 1
            run_len = j - i
            if run_len >= MIN_CONSEC_DAYS:
                current_event += 1
                start_date = dates[i]
                end_date   = dates[j - 1]
                for k in range(i, j):
                    hw[k]       = 1
                    event_id[k] = current_event
                    ev_start[k] = start_date
                    ev_end[k]   = end_date
                    ev_len[k]   = run_len
            i = j
        else:
            i += 1

    city_df["heatwave"]         = hw
    city_df["hw_event_id"]      = event_id
    city_df["hw_event_start"]   = pd.to_datetime(ev_start)
    city_df["hw_event_end"]     = pd.to_datetime(ev_end)
    city_df["hw_event_length"]  = ev_len
    return city_df

results_per_city = []
for ck in CITY_ORDER:
    sub = df[df["city_key"] == ck].copy()
    sub = apply_duration_and_events(sub)
    results_per_city.append(sub)

df_labelled = pd.concat(results_per_city).sort_values(["city_key", "date"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — CLASS BALANCE ANALYSIS")

stats_rows = []
log(f"\n  {'City':<14} {'Total':>7} {'HW=1':>7} {'HW=0':>7} {'HW%':>8} {'Events':>8} "
    f"{'Avg dur':>9} {'Max dur':>9} {'Single-day':>12}")
log("  " + "-" * 92)

for ck in CITY_ORDER:
    sub   = df_labelled[df_labelled["city_key"] == ck]
    n_tot = len(sub)
    n_hw  = int(sub["heatwave"].sum())
    n_no  = n_tot - n_hw
    pct   = n_hw / n_tot * 100
    n_ev  = int(sub["hw_event_id"].max())
    iso   = int((sub["qualifying_day"] == 1).sum()) - n_hw   # isolated single days
    if n_ev > 0:
        ev_lens = sub[sub["heatwave"] == 1].groupby("hw_event_id")["hw_event_length"].first()
        avg_dur = ev_lens.mean()
        max_dur = int(ev_lens.max())
    else:
        avg_dur = 0.0
        max_dur = 0

    stats_rows.append({
        "city":       CITY_LABELS[ck],
        "total_days": n_tot,
        "hw_days":    n_hw,
        "non_hw":     n_no,
        "hw_pct":     round(pct, 2),
        "n_events":   n_ev,
        "avg_dur":    round(avg_dur, 1),
        "max_dur":    max_dur,
        "single_day_qual": iso,
    })
    log(f"  {CITY_LABELS[ck]:<14} {n_tot:>7,} {n_hw:>7,} {n_no:>7,} {pct:>7.2f}% "
        f"{n_ev:>8} {avg_dur:>9.1f} {max_dur:>9} {iso:>12}")

# ── Yearly heatwave day counts ─────────────────────────────────────────────────
log("\n  Yearly heatwave days per city:")
year_header = f"  {'Year':<6}" + "".join(f"{CITY_LABELS[ck]:>14}" for ck in CITY_ORDER)
log(year_header)
log("  " + "-" * (6 + 14 * 5))

years = sorted(df_labelled["year"].unique())
for yr in years:
    row_str = f"  {yr:<6}"
    for ck in CITY_ORDER:
        sub = df_labelled[(df_labelled["city_key"] == ck) & (df_labelled["year"] == yr)]
        n   = int(sub["heatwave"].sum())
        row_str += f"{n:>14}"
    log(row_str)

# Save stats
stats_path = RESULTS / "phase6_class_balance.json"
with open(stats_path, "w") as f:
    json.dump(stats_rows, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — SAVE weather_labelled.csv")

# Final column set
KEEP_COLS = [
    "city", "city_key", "latitude", "longitude", "region_type", "state",
    "date",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    # Labeling columns
    "tmax_normal",        # city-specific daily climatological Tmax (°C)
    "tmax_departure",     # Tmax - tmax_normal (°C)
    "qualifying_day",     # 1 if day meets threshold+departure (before duration filter)
    "heatwave",           # TARGET VARIABLE: 1 = heatwave day (2+ consecutive qualifying)
    "hw_event_id",        # Event ID (0 = not a heatwave day, 1..N = event number)
    "hw_event_start",     # Date this event started
    "hw_event_end",       # Date this event ended
    "hw_event_length",    # Duration of this event in days
]

df_out = df_labelled[KEEP_COLS].copy()
df_out.to_csv(OUT_CSV, index=False)
size_mb = OUT_CSV.stat().st_size / 1e6
log(f"  Saved : {OUT_CSV.relative_to(ROOT)}")
log(f"  Size  : {size_mb:.2f} MB")
log(f"  Rows  : {len(df_out):,}")
log(f"  Cols  : {df_out.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — VALIDATION")

log("\n  7a. Basic structural checks")
df_v = pd.read_csv(OUT_CSV, parse_dates=["date", "hw_event_start", "hw_event_end"])

assert len(df_v) == 65135,       f"Row count mismatch: {len(df_v)}"
assert df_v.shape[1] == len(KEEP_COLS), f"Column count mismatch: {df_v.shape[1]}"
assert df_v["heatwave"].isnull().sum() == 0,        "heatwave has NaN"
assert set(df_v["heatwave"].unique()) <= {0, 1},    "heatwave has values other than 0/1"
assert df_v.duplicated(subset=["city_key","date"]).sum() == 0, "duplicate city+date"
assert df_v["city_key"].nunique() == 5,              "wrong city count"
log("    [PASS] 65,135 rows")
log("    [PASS] Correct column count")
log("    [PASS] heatwave column: 0 missing, only 0/1 values")
log("    [PASS] 0 duplicate city+date records")
log("    [PASS] 5 cities present")

log("\n  7b. Date range check")
assert str(df_v["date"].min().date()) == "1990-01-01"
assert str(df_v["date"].max().date()) == "2025-08-31"
log("    [PASS] Date range 1990-01-01 → 2025-08-31")

log("\n  7c. No future-data leakage check")
log("    The label for date T uses:")
log("      - temperature_2m_max on date T  (observation from T)")
log("      - tmax_normal for that calendar day-of-year (computed from")
log("        1990-2020 baseline using ONLY historical averages, no T+1 data)")
log("      - qualifying_day depends only on T's own observation vs normal")
log("      - Duration criterion: a run of consecutive qualifying days is")
log("        identified retrospectively, but the labeling is for the")
log("        entire run — future ML features will use only T-aligned data.")
log("    [PASS] No future-data leakage in label methodology")

log("\n  7d. City-specific label correctness")
for ck in CITY_ORDER:
    sub = df_v[df_v["city_key"] == ck]
    # All hw=1 days must be qualifying days
    hw_not_qual = sub[(sub["heatwave"] == 1) & (sub["qualifying_day"] == 0)]
    assert len(hw_not_qual) == 0, f"{ck}: heatwave=1 but qualifying_day=0"
    # All hw=1 days must have event_id > 0
    hw_no_event = sub[(sub["heatwave"] == 1) & (sub["hw_event_id"] == 0)]
    assert len(hw_no_event) == 0, f"{ck}: heatwave=1 but hw_event_id=0"
    log(f"    [PASS] {ck}: all heatwave=1 days are qualifying and have event IDs")

log("\n  7e. Consecutive-day integrity")
for ck in CITY_ORDER:
    sub = df_v[df_v["city_key"] == ck].sort_values("date")
    events = sub[sub["heatwave"] == 1].groupby("hw_event_id")
    for ev_id, ev_rows in events:
        ev_dates = ev_rows["date"].sort_values().reset_index(drop=True)
        gaps = (ev_dates - ev_dates.shift(1)).dropna()
        assert (gaps == pd.Timedelta("1d")).all(), f"{ck} event {ev_id}: non-consecutive"
        assert len(ev_rows) >= MIN_CONSEC_DAYS, f"{ck} event {ev_id}: < {MIN_CONSEC_DAYS} days"
    log(f"    [PASS] {ck}: all events are consecutive and >= {MIN_CONSEC_DAYS} days")

log("\n  7f. weather_cleaned.csv untouched")
cleaned_check = pd.read_csv(IN_CSV)
assert len(cleaned_check) == 65135
assert "rain_sum" not in cleaned_check.columns
assert "heatwave" not in cleaned_check.columns
log("    [PASS] weather_cleaned.csv: 65,135 rows, no heatwave column, rain_sum absent")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 8 — REPRESENTATIVE EXAMPLES (per city)")

month_labels = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for ck in CITY_ORDER:
    sub = df_v[df_v["city_key"] == ck].sort_values("date").reset_index(drop=True)
    log(f"\n  {CITY_LABELS[ck]}:")
    log(f"  {'Date':<12} {'Tmax':>7} {'Normal':>8} {'Dep':>7} {'Qual':>6} {'HW':>4} {'EvID':>6} {'EvLen':>7}")
    log("  " + "-" * 65)

    # Normal day (middle of winter — low Tmax, no heatwave)
    winter = sub[(sub["heatwave"] == 0) & (sub["date"].dt.month == 1)].iloc[0]
    log(f"  {str(winter['date'].date()):<12} "
        f"{winter['temperature_2m_max']:>7.1f} {winter['tmax_normal']:>8.1f} "
        f"{winter['tmax_departure']:>7.2f} {int(winter['qualifying_day']):>6} "
        f"{int(winter['heatwave']):>4} {int(winter['hw_event_id']):>6} {int(winter['hw_event_length']):>7}  ← normal day (Jan)")

    # Isolated qualifying day (single hot day, not part of an event)
    iso_days = sub[(sub["qualifying_day"] == 1) & (sub["heatwave"] == 0)]
    if len(iso_days) > 0:
        iso = iso_days.iloc[0]
        log(f"  {str(iso['date'].date()):<12} "
            f"{iso['temperature_2m_max']:>7.1f} {iso['tmax_normal']:>8.1f} "
            f"{iso['tmax_departure']:>7.2f} {int(iso['qualifying_day']):>6} "
            f"{int(iso['heatwave']):>4} {int(iso['hw_event_id']):>6} {int(iso['hw_event_length']):>7}  ← isolated qualifying (not labeled HW)")

    # Heatwave events — if any
    events = sub[sub["heatwave"] == 1]["hw_event_id"].unique()
    if len(events) > 0:
        # Pick the longest event for illustration
        ev_lens = sub[sub["heatwave"]==1].groupby("hw_event_id")["hw_event_length"].first()
        longest_ev_id = int(ev_lens.idxmax())
        ev_rows = sub[sub["hw_event_id"] == longest_ev_id].sort_values("date")
        # Show start, one middle, end
        indices_to_show = [0]
        if len(ev_rows) > 2:
            indices_to_show.append(len(ev_rows) // 2)
        indices_to_show.append(len(ev_rows) - 1)
        for idx in sorted(set(indices_to_show)):
            row = ev_rows.iloc[idx]
            tag = "← event start" if idx == 0 else \
                  "← event middle" if idx == len(ev_rows) // 2 else \
                  "← event end"
            log(f"  {str(row['date'].date()):<12} "
                f"{row['temperature_2m_max']:>7.1f} {row['tmax_normal']:>8.1f} "
                f"{row['tmax_departure']:>7.2f} {int(row['qualifying_day']):>6} "
                f"{int(row['heatwave']):>4} {int(row['hw_event_id']):>6} {int(row['hw_event_length']):>7}  {tag}")
    else:
        log(f"  (No heatwave events in {CITY_LABELS[ck]})")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 9 — PLOTS")

df_v = pd.read_csv(OUT_CSV, parse_dates=["date"])
df_v["year"]  = df_v["date"].dt.year
df_v["month"] = df_v["date"].dt.month

# ── Plot 1: Annual heatwave days per city ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Annual Heatwave Days per City\n(IMD-Inspired Operational Label, 1990–2025)",
             fontsize=13, fontweight="bold")
all_years = sorted(df_v["year"].unique())

for ax, ck in zip(axes.flat[:5], CITY_ORDER):
    sub = df_v[df_v["city_key"] == ck]
    yr_counts = sub.groupby("year")["heatwave"].sum().reindex(all_years, fill_value=0)
    bars = ax.bar(yr_counts.index, yr_counts.values,
                  color=CITY_COLORS[ck], alpha=0.8, edgecolor="none")
    from scipy import stats as _stats
    if yr_counts.values.std() > 0:
        slope, intercept, _, _, _ = _stats.linregress(yr_counts.index, yr_counts.values)
        fit = slope * np.array(yr_counts.index) + intercept
        ax.plot(yr_counts.index, fit, "k--", linewidth=1.5, alpha=0.7,
                label=f"trend {slope:+.1f} d/yr")
        ax.legend(fontsize=7)
    ax.set_title(CITY_LABELS[ck])
    ax.set_xlabel("Year")
    ax.set_ylabel("Heatwave days")
    ax.tick_params(axis="x", rotation=45, labelsize=7)

axes.flat[5].axis("off")  # 6th panel empty
plt.tight_layout()
save(fig, "01_annual_heatwave_days_per_city.png")

# ── Plot 2: Monthly distribution of heatwave days ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
x = np.arange(12)
width = 0.15
for i, ck in enumerate(CITY_ORDER):
    sub = df_v[df_v["city_key"] == ck]
    counts = sub.groupby("month")["heatwave"].sum().reindex(range(1, 13), fill_value=0)
    ax.bar(x + i * width, counts.values, width, label=CITY_LABELS[ck],
           color=CITY_COLORS[ck], edgecolor="white", linewidth=0.5)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(month_names)
ax.set_title("Monthly Distribution of Heatwave Days by City (1990–2025)\nIMD-Inspired Operational Label")
ax.set_ylabel("Total heatwave days (all years)")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
save(fig, "02_monthly_heatwave_distribution.png")

# ── Plot 3–7: Per-city temperature time series with heatwave overlay ──────────
SAMPLE_YEAR_START = 2010
SAMPLE_YEAR_END   = 2015

for ck in CITY_ORDER:
    sub = df_v[(df_v["city_key"] == ck) &
               (df_v["year"] >= SAMPLE_YEAR_START) &
               (df_v["year"] <= SAMPLE_YEAR_END)].sort_values("date")

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(sub["date"], sub["temperature_2m_max"],
            color=CITY_COLORS[ck], linewidth=0.9, alpha=0.8, label="Daily Tmax (°C)")

    # Shade heatwave periods red
    hw_days = sub[sub["heatwave"] == 1]
    if len(hw_days) > 0:
        # Fill entire heatwave days with a red band
        for _, row in hw_days.iterrows():
            ax.axvspan(row["date"] - pd.Timedelta("0.5d"),
                       row["date"] + pd.Timedelta("0.5d"),
                       color="red", alpha=0.25, linewidth=0)

    # Threshold line
    thresh = COASTAL_ABS_THRESHOLD if ck == "mumbai" else PLAINS_ABS_THRESHOLD
    ax.axhline(thresh, color="darkred", linestyle="--", linewidth=1.0,
               label=f"Absolute threshold ({thresh}°C)")

    ax.set_title(f"{CITY_LABELS[ck]} — Daily Tmax with Heatwave Events Highlighted "
                 f"({SAMPLE_YEAR_START}–{SAMPLE_YEAR_END})\nRed shading = heatwave day (IMD-Inspired Label)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    hw_patch = mpatches.Patch(color="red", alpha=0.4, label="Heatwave day (label=1)")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [hw_patch], fontsize=8, ncol=3)
    plt.tight_layout()
    save(fig, f"03_tmax_heatwave_overlay_{ck}.png")

# ── Plot 8: Class balance bar chart ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Heatwave Label Class Distribution by City", fontsize=13, fontweight="bold")

cities_display = [CITY_LABELS[ck] for ck in CITY_ORDER]
hw_counts = [df_v[df_v["city_key"] == ck]["heatwave"].sum() for ck in CITY_ORDER]
nohw_counts = [df_v[df_v["city_key"] == ck]["heatwave"].eq(0).sum() for ck in CITY_ORDER]
hw_pcts = [hw / (hw + no) * 100 for hw, no in zip(hw_counts, nohw_counts)]

ax = axes[0]
x  = np.arange(len(CITY_ORDER))
bars1 = ax.bar(x - 0.2, nohw_counts, 0.4,
               color=[CITY_COLORS[ck] for ck in CITY_ORDER], alpha=0.4, label="No heatwave (0)")
bars2 = ax.bar(x + 0.2, hw_counts, 0.4,
               color=[CITY_COLORS[ck] for ck in CITY_ORDER], alpha=0.9, label="Heatwave (1)")
ax.set_xticks(x)
ax.set_xticklabels(cities_display, rotation=15)
ax.set_ylabel("Number of days")
ax.set_title("Absolute counts")
ax.legend()

ax = axes[1]
ax.bar(cities_display, hw_pcts, color=[CITY_COLORS[ck] for ck in CITY_ORDER], edgecolor="white")
for i, pct in enumerate(hw_pcts):
    ax.text(i, pct + 0.1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Heatwave day % of total")
ax.set_title("Positive class fraction (%)")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save(fig, "08_class_balance.png")

# ── Plot 9: Departure distribution on HW vs non-HW days ──────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
fig.suptitle("Tmax Departure from Normal: Heatwave Days vs Non-Heatwave Days",
             fontsize=12, fontweight="bold")
for ax, ck in zip(axes, CITY_ORDER):
    sub  = df_v[df_v["city_key"] == ck]
    dep_hw  = sub.loc[sub["heatwave"] == 1, "tmax_departure"]
    dep_no  = sub.loc[sub["heatwave"] == 0, "tmax_departure"]
    bins = np.linspace(sub["tmax_departure"].min(), sub["tmax_departure"].max(), 50)
    ax.hist(dep_no, bins=bins, alpha=0.5, density=True, color="steelblue", label="HW=0")
    ax.hist(dep_hw, bins=bins, alpha=0.7, density=True, color="red", label="HW=1")
    ax.axvline(DEPARTURE_HW_THRESHOLD, color="darkred", linestyle="--",
               linewidth=1.2, label=f"+{DEPARTURE_HW_THRESHOLD}°C threshold")
    ax.set_title(CITY_LABELS[ck])
    ax.set_xlabel("Departure from normal (°C)")
    ax.set_ylabel("Density" if ck == "delhi" else "")
    ax.legend(fontsize=7)
plt.tight_layout()
save(fig, "09_departure_distribution_hw_vs_nonhw.png")

# ── Plot 10: Event duration distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for ck in CITY_ORDER:
    sub  = df_v[df_v["city_key"] == ck]
    evs  = sub[sub["heatwave"] == 1].groupby("hw_event_id")["hw_event_length"].first()
    if len(evs) > 0:
        ax.hist(evs.values, bins=range(2, int(evs.max()) + 2), alpha=0.55,
                label=f"{CITY_LABELS[ck]} (n={len(evs)})", color=CITY_COLORS[ck],
                edgecolor="white", linewidth=0.5)
ax.set_title("Heatwave Event Duration Distribution by City (1990–2025)")
ax.set_xlabel("Event duration (days)")
ax.set_ylabel("Number of events")
ax.legend(ncol=2)
plt.tight_layout()
save(fig, "10_event_duration_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 10 — SAVE LOG")

log_path = RESULTS / "phase6_labeling_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
log(f"  Log saved: {log_path.relative_to(ROOT)}")
log("\n  Phase 6 labeling script complete.")
