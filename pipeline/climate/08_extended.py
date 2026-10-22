"""Extended heat indicators for the meadows of GR2210002, from the daily series written by 03_observed.py:
  - marine heatwave days with a FIXED baseline (as in 03) and on the DETRENDED series (extremes beyond the trend);
  - days per Hobday et al. (2018) category (moderate, strong, severe, extreme), fixed baseline;
  - cumulative degree-days above 28 degC (absolute heat-stress threshold for Posidonia, see REPORT).
Meadow-weighted as in 04_summarise.py (equal-area HEALPix d17 meadow cells, each taking its assigned series).
Usage: pixi run python pipeline/climate/08_extended.py <ROOT>  -> results_obs_extended.json, tables/annual_extended_*.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import mhw

ROOT = sys.argv[1]
D, T = f"{ROOT}/data", f"{ROOT}/tables"
os.makedirs(T, exist_ok=True)
LAST_FULL, TH = 2025, 28.0
cells = pd.read_parquet(f"{D}/cell_metrics.parquet")
mc = cells[cells.meadow_ref].copy()
ds = xr.open_dataset(f"{D}/daily_series_obs.nc")


def series(key):
    s = ds[key].to_series()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated()].dropna()
    return s[s.index.year <= LAST_FULL]


def annual_extended(key):
    s = series(key)
    t, x = s.index, s.values.astype(float)
    base = (1983, 2012) if key.startswith("REP") else (1987, 2016)
    d = mhw.doy366(t)
    seas, thr = mhw.climatology(t, x, base)
    _, inm, _ = mhw.detect(t, x, seas, thr)
    cat = mhw.categories(x, seas, thr, d, inm)
    xd, slope = mhw.detrend(t, x)
    seas_d, thr_d = mhw.climatology(t, xd, base)
    _, inm_d, _ = mhw.detect(t, xd, seas_d, thr_d)
    df = pd.DataFrame({"x": x, "fix": inm, "det": inm_d, "cat": cat}, index=t)
    g = df.groupby(df.index.year)
    out = pd.DataFrame({"mhw_days_fixed": g.fix.sum(), "mhw_days_detrended": g.det.sum(),
                        **{f"days_cat{k}": g.cat.apply(lambda c, k=k: int((c == k).sum())) for k in (1, 2, 3, 4)},
                        "degree_days_gt_28": g.x.apply(lambda v: mhw.degree_days(v.values, TH)),
                        "n_days": g.x.count()})
    out["key"], out["trend_degC_per_decade"] = key, 10 * slope
    return out


keys = sorted(set(mc.rep_key.dropna()) | set(mc.zt_key.dropna()))
ann = pd.concat([annual_extended(k).reset_index(names="year") for k in keys], ignore_index=True)
ann.to_csv(f"{T}/annual_extended_by_series.csv", index=False)
IND = ["mhw_days_fixed", "mhw_days_detrended", "days_cat1", "days_cat2", "days_cat3", "days_cat4", "degree_days_gt_28"]


def weighted(sub, keycol):
    w = sub[keycol].value_counts()
    a = ann[ann.key.isin(w.index)].copy()
    a["w"] = a.key.map(w)
    return a.groupby("year").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in IND}))


res = {"method": {"baseline": "SST 1983-2012, reanalysis 1987-2016", "detrending": "linear, whole record",
                  "categories": "Hobday et al. 2018", "degree_days_threshold_degC": TH}, "groups": {}}
groups = {"SST (satellite), all meadows": ("rep_key", mc, 1982),
          "T at meadow depth (reanalysis), all meadows": ("zt_key", mc, 1987),
          "T at meadow depth (reanalysis), 0-10 m": ("zt_key", mc[mc.band == "0-10 m"], 1987)}
for name, (col, sub, y0) in groups.items():
    if not len(sub):
        continue
    a = weighted(sub, col).loc[y0:LAST_FULL]
    a.to_csv(f"{T}/annual_extended_{col}_{len(sub)}cells.csv")
    p1, p2 = a.loc[y0:y0 + 29], a.loc[2016:LAST_FULL]
    res["groups"][name] = {
        "cells": int(len(sub)),
        "mean_first_30y": {c: round(float(p1[c].mean()), 1) for c in IND},
        "mean_2016_2025": {c: round(float(p2[c].mean()), 1) for c in IND},
        "year_2024": {c: round(float(a.loc[2024, c]), 1) for c in IND},
        "trend_per_decade": {c: {k: round(v, 2) for k, v in mhw.trend(a.index, a[c]).items()
                                 if k in ("ols", "ols_lo", "ols_hi")} for c in ("mhw_days_fixed", "mhw_days_detrended", "degree_days_gt_28")}}
json.dump(res, open(f"{ROOT}/results_obs_extended.json", "w"), indent=1)
print(json.dumps(res["groups"], indent=1))

# Figure: fixed vs detrended marine heatwave days, and degree-days above 28 degC, shallow meadows (0-10 m)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
a = pd.read_csv(f"{T}/annual_extended_zt_key_{len(mc[mc.band == '0-10 m'])}cells.csv", index_col=0)
os.makedirs(f"{ROOT}/figures", exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True)
ax[0].plot(a.index, a.mhw_days_fixed, color="#eb6834", lw=2, label="fixed baseline 1987-2016 (departure from the past)")
ax[0].plot(a.index, a.mhw_days_detrended, color="#2a78d6", lw=2, label="detrended (extremes beyond the warming trend)")
ax[0].set_ylabel("marine heatwave days per year")
ax[0].legend(frameon=False, loc="upper left", fontsize=9)
ax[0].set_title("Shallow Posidonia meadows (0-10 m), Laganas Bay GR2210002: temperature at meadow depth (Med reanalysis)", fontsize=10, loc="left")
ax[1].bar(a.index, a.degree_days_gt_28, color="#b8364e", width=0.8)
ax[1].set_ylabel("degree-days above 28 degC")
ax[1].set_xlabel("year")
for x_ in ax:
    x_.spines[["top", "right"]].set_visible(False)
    x_.grid(axis="y", color="#e4e3df")
fig.text(0.01, -0.02, "Marine heatwaves after Hobday et al. 2016; detrending on deseasonalised anomalies. Exposure, not impact. "
         "Code: github.com/annefou/natura2000-laganas-posidonia-evidence", fontsize=7.5, color="#52514e")
fig.savefig(f"{ROOT}/figures/fig4_fixed_vs_detrended.png", dpi=150, bbox_inches="tight")
print("figure", f"{ROOT}/figures/fig4_fixed_vs_detrended.png")
