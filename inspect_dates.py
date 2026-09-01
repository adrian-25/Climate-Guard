import pandas as pd

df = pd.read_csv('data/features/ml_baseline.csv', parse_dates=['date'], low_memory=False)
print(f"Total rows: {len(df)}")
print(f"Overall date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print()
for city, grp in df.groupby('city_key'):
    print(f"  {city:12s}  {grp['date'].min().date()} -> {grp['date'].max().date()}  ({len(grp)} rows)")
print()
# Show how many positives fall in each candidate split window
for label, y1, y2 in [('train (1990-2019)', 1990, 2019),
                       ('val   (2020-2022)', 2020, 2022),
                       ('test  (2023-2025)', 2023, 2025)]:
    mask = (df['date'].dt.year >= y1) & (df['date'].dt.year <= y2)
    sub = df[mask]
    pos = int(sub['heatwave_next_day'].sum())
    print(f"  {label}  rows={len(sub):,}  positives={pos}")
