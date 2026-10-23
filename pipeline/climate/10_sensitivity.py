"""Sensitivity of the marine heatwave results to the definition (Hobday et al. 2016 choices).

Variants: baseline = the original one (SST 1983-2012, reanalysis 1987-2016) or the WMO climate normal 1991-2020;
threshold percentile = 90 (Hobday 2016), 95 or 99; fixed baseline or detrended series (mhw.detrend). 12 variants.

A. Laganas Bay (GR2210002), shallow meadows 0-10 m, temperature at meadow depth (reanalysis), meadow-weighted as in
   04_summarise.py: mean MHW days 1987-2015 and 2016-2025, 2024 value and its rank (1 = highest year), trend per decade.
B. The 18 genotyped meadows (pipeline/genetics): which meadows fall in class 1 (6 highest exposure x 6 lowest genetic
   diversity) when the MHW-days part of the exposure score uses each variant (days above 28 C do not depend on it).

Usage: pixi run python pipeline/climate/10_sensitivity.py <ROOT climate> <ROOT genetics>
Outputs: <ROOT climate>/results_sensitivity.json, <ROOT climate>/tables/sensitivity_laganas.csv,
         <ROOT genetics>/tables/sensitivity_genetics.csv
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import mhw

RC, RG = sys.argv[1], sys.argv[2]
os.makedirs(f"{RC}/tables", exist_ok=True)
os.makedirs(f"{RG}/tables", exist_ok=True)
LAST = 2025
VARIANTS = [dict(base=b, pct=p, detrend=d) for b, p, d in itertools.product(("original", "1991-2020"), (90, 95, 99), (False, True))]


def label(v):
    return f"{v['base']} p{v['pct']} {'detrended' if v['detrend'] else 'fixed'}"


def mhw_days(t, x, base, v):
    t = pd.DatetimeIndex(t)
    if v["detrend"]:
        x, _ = mhw.detrend(t, x)
    b = base if v["base"] == "original" else (1991, 2020)
    seas, thr = mhw.climatology(t, x, b, pct=v["pct"])
    _, inm, _ = mhw.detect(t, x, seas, thr)
    return pd.Series(inm, index=t).groupby(t.year).sum()


def clean(s):
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated()].dropna()
    return s[s.index.year <= LAST]


# --- A. Laganas shallow meadows
cells = pd.read_parquet(f"{RC}/data/cell_metrics.parquet")
mc = cells[cells.meadow_ref & (cells.band == "0-10 m")]
w = mc.zt_key.value_counts()
ds = xr.open_dataset(f"{RC}/data/daily_series_obs.nc")
rows = []
for v in VARIANTS:
    per = {}
    for k in w.index:
        s = clean(ds[k].to_series())
        per[k] = mhw_days(s.index, s.values.astype(float), (1987, 2016), v)
    a = pd.DataFrame(per).loc[1987:LAST]
    days = (a * w / w.sum()).sum(axis=1)
    tr = mhw.trend(days.index, days.values)
    rows.append(dict(variant=label(v), mean_1987_2015=round(days.loc[:2015].mean(), 1), mean_2016_2025=round(days.loc[2016:].mean(), 1),
                     y2024=round(days.loc[2024], 1), rank_2024=int((days > days.loc[2024]).sum() + 1),
                     trend_per_decade=round(tr["ols"], 1), trend_lo=round(tr["ols_lo"], 1), trend_hi=round(tr["ols_hi"], 1)))
lag = pd.DataFrame(rows)
lag.to_csv(f"{RC}/tables/sensitivity_laganas.csv", index=False)

# --- B. Genotyped meadows
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genetics"))
ex = pd.read_csv(f"{RG}/data/exposure_by_meadow.csv")
gen = pd.read_csv(f"{RG}/data/genetics_by_meadow.csv")
K = 6
late = {}
for _, s in ex.iterrows():
    phy = xr.open_dataset(f"{RG}/data/cmems/{s.site_code}_phy_my_temp.nc")
    lev = int(np.abs(phy.depth.values - s.phy_level_m).argmin())
    ix = int(np.abs(phy.longitude.values - s.phy_lon).argmin())
    iy = int(np.abs(phy.latitude.values - s.phy_lat).argmin())
    x = phy.thetao.isel(depth=lev).values[:, iy, ix].astype(float)
    ser = clean(pd.Series(x, index=pd.DatetimeIndex(phy.time.values)))
    late[s.site_code] = {label(v): mhw_days(ser.index, ser.values, (1987, 2016), v).loc[2016:LAST].mean() for v in VARIANTS}
L = pd.DataFrame(late).T
m = gen.merge(ex[["site_code", "zt_days_gt28_2016_2025"]], on="site_code").set_index("site_code")
Gs = m[["R", "Hs"]].rank(pct=True).mean(axis=1)
loG = Gs.rank(method="first") <= K
cls = {}
for v in VARIANTS:
    E = pd.concat([L[label(v)].rank(pct=True), m.zt_days_gt28_2016_2025.rank(pct=True)], axis=1).mean(axis=1)
    hiE = E.rank(method="first", ascending=False) <= K
    cls[label(v)] = sorted((hiE & loG)[lambda b: b].index)
gtab = pd.DataFrame({"variant": list(cls), "class1_meadows": [", ".join(c) for c in cls.values()]})
gtab.to_csv(f"{RG}/tables/sensitivity_genetics.csv", index=False)
count = pd.Series([c for cs in cls.values() for c in cs]).value_counts()

res = {"laganas_shallow": lag.to_dict(orient="records"),
       "genetics_class1": {k: v for k, v in cls.items()},
       "genetics_class1_frequency_of_12": count.to_dict()}
json.dump(res, open(f"{RC}/results_sensitivity.json", "w"), indent=1)
print(lag.to_string(index=False))
print(gtab.to_string(index=False))
print(count.to_string())
