import pandas as pd
import numpy as np

df = pd.read_csv('data/features/climateguard_features.csv',
                 parse_dates=['date'], low_memory=False)

nans = df[df['tmax_departure_zscore'].isna()]
print("NaN count per city:")
print(nans['city_key'].value_counts())
print()

print("Date ranges of NaN rows:")
for city, grp in nans.groupby('city_key'):
    dates = grp['date']
    print(f"  {city}: {dates.min().date()} to {dates.max().date()}  ({len(grp)} rows)")
print()

print("Row position within city block and tmax_departure value:")
for city, grp in df.groupby('city_key'):
    city_sorted = grp.sort_values('date').reset_index(drop=True)
    city_nans = city_sorted[city_sorted['tmax_departure_zscore'].isna()]
    if len(city_nans) == 0:
        continue
    for pos, row in city_nans.iterrows():
        td = row['tmax_departure']
        print(f"  {city} row {pos:5d}  date={row['date'].date()}  tmax_departure={td:.4f}")

print()
print("30-day trailing std of tmax_departure before each NaN row:")
for city, grp in df.groupby('city_key'):
    city_sorted = grp.sort_values('date').reset_index(drop=True)
    city_nans = city_sorted[city_sorted['tmax_departure_zscore'].isna()]
    if len(city_nans) == 0:
        continue
    for idx in city_nans.index:
        start = max(0, idx - 30)
        window = city_sorted.loc[start:idx-1, 'tmax_departure']
        std_val = window.std()
        mean_val = window.mean()
        print(f"  {city} row {idx:5d}  window_n={len(window)}  "
              f"window_std={std_val:.6f}  window_mean={mean_val:.4f}")
