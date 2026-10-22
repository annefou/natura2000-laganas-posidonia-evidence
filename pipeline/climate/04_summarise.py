"""Meadow-weighted indicators, trends, depth bands, cross-checks.  Every meadow HEALPix d17 cell has equal area,
so a site value = mean over meadow cells of the value of the series assigned to each cell.
Usage: pixi run python pipeline/climate/04_summarise.py <ROOT> -> results_obs.json, tables/*.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import mhw

ROOT = sys.argv[1]
D = f"{ROOT}/data"
os.makedirs(f"{ROOT}/tables", exist_ok=True)
cells = pd.read_parquet(f"{D}/cell_metrics.parquet")
mc = cells[cells.meadow_ref].copy()
ann = pd.read_csv(f"{D}/annual_by_series.csv")
IND = ["mhw_days", "mhw_i_mean", "mhw_i_max", "mhw_i_cum", "n_events", "t_max", "days_gt_28", "days_gt_28.9", "days_gt_30", "run_gt_28.9"]
LAST_FULL = 2025


def weighted_annual(sub, keycol):
    w = sub[keycol].value_counts()
    a = ann[ann.key.isin(w.index)].copy()
    a["w"] = a.key.map(w)
    g = a.groupby("year")
    out = pd.DataFrame({c: g.apply(lambda d: np.average(d[c].fillna(0) if c != "mhw_i_mean" and c != "mhw_i_max" else d[c].fillna(np.nan).fillna(0), weights=d.w)) for c in IND})
    # intensities: average only where MHW days > 0
    for c in ("mhw_i_mean", "mhw_i_max"):
        out[c] = g.apply(lambda d: np.average(d[c][d.mhw_days > 0], weights=d.w[d.mhw_days > 0]) if (d.mhw_days > 0).any() else np.nan)
    out["n_days"] = g.n_days.min()
    return out


def trends(a, y0, y1=LAST_FULL):
    s = a.loc[y0:y1]
    return {c: mhw.trend(s.index, s[c]) for c in ["mhw_days", "mhw_i_cum", "t_max", "days_gt_28"]}


res = {}
groups = {"SST (REP, satellite)": ("rep_key", mc, 1982),
          "T at meadow depth (reanalysis), all": ("zt_key", mc, 1987)}
for b in ["0-10 m", "10-20 m", "20-40 m"]:
    groups[f"T at meadow depth (reanalysis), {b}"] = ("zt_key", mc[mc.band == b], 1987)
groups["bottomT of model column (reanalysis), all"] = ("bt_key", mc, 1987)
tab = []
for name, (kc, sub, y0) in groups.items():
    a = weighted_annual(sub, kc)
    a.to_csv(f"{ROOT}/tables/annual_{kc}_{name.split(',')[-1].strip().replace(' ', '').replace('(', '').replace(')', '')}.csv")
    tr = trends(a, max(y0, 1983 if kc == "rep_key" else y0))
    early, late = a.loc[1987:2016], a.loc[2016:2025]
    row = {"group": name, "n_cells": len(sub), "n_series": sub[kc].nunique(),
           "mhw_days_1987_2016": early.mhw_days.mean(), "mhw_days_2016_2025": late.mhw_days.mean(),
           "tmax_1987_2016": early.t_max.mean(), "tmax_2016_2025": late.t_max.mean(),
           "days_gt28_1987_2016": early.days_gt_28.mean(), "days_gt28_2016_2025": late.days_gt_28.mean(),
           "days_gt28.9_2016_2025": late["days_gt_28.9"].mean(), "days_gt30_2016_2025": late.days_gt_30.mean(),
           "years_tmax_gt28_1987_2025": int((a.loc[1987:2025].t_max > 28).sum()),
           "years_tmax_gt28.9_1987_2025": int((a.loc[1987:2025].t_max > 28.9).sum()),
           "years_icum_gt120_1987_2025": int((a.loc[1987:2025].mhw_i_cum > 120).sum()),
           "max_run_gt28.9_days": int(a.loc[1987:2026]["run_gt_28.9"].max()),
           "mhw_days_2026_partial": float(a.loc[2026].mhw_days) if 2026 in a.index else None,
           "tmax_2026": float(a.loc[2026].t_max) if 2026 in a.index else None,
           "trend_mhw_days_per_dec": tr["mhw_days"], "trend_tmax_per_dec": tr["t_max"],
           "trend_icum_per_dec": tr["mhw_i_cum"], "trend_days_gt28_per_dec": tr["days_gt_28"],
           "top5_years_mhw_days": a.loc[:2026].mhw_days.sort_values(ascending=False).head(5).round(0).to_dict(),
           "top5_years_tmax": a.loc[:2026].t_max.sort_values(ascending=False).head(5).round(2).to_dict()}
    tab.append(row)
res["groups"] = tab

# --- cross-checks
ds = xr.open_dataset(f"{D}/daily_series_obs.nc")
w_rep = mc.rep_key.value_counts()
w_zt0 = mc.zt_key.value_counts()
def wmean_daily(wcounts):
    arr = sum(ds[k] * n for k, n in wcounts.items()) / wcounts.sum()
    return arr.to_series()
rep_s = wmean_daily(w_rep)
nrt_s = wmean_daily(mc.nrt_key.value_counts())
# reanalysis top level (1.0 m) at the meadow columns, weighted by meadow cells
mc["zt_top"] = mc.zt_key.str.rsplit("_", n=1).str[0] + "_0"
zt_top = wmean_daily(mc.zt_top.value_counts())
cc = {}
for nm, s in (("NRT_minus_REP", nrt_s), ("reanalysis1m_minus_REP", zt_top)):
    j = pd.concat([s, rep_s], axis=1, keys=["x", "rep"]).dropna()
    j = j[j.index.year <= LAST_FULL]
    summer = j[j.index.month.isin([7, 8, 9])]
    cc[nm] = {"period": f"{j.index.min().date()}..{j.index.max().date()}", "bias_all": float((j.x - j.rep).mean()),
              "bias_JAS": float((summer.x - summer.rep).mean()), "rmsd": float(np.sqrt(((j.x - j.rep) ** 2).mean())),
              "r_anomaly": float(np.corrcoef(j.x - j.x.groupby(j.index.dayofyear).transform("mean"),
                                            j.rep - j.rep.groupby(j.index.dayofyear).transform("mean"))[0, 1]),
              "mean_days_gt28_per_yr_x": float((j.x > 28).groupby(j.index.year).sum().mean()),
              "mean_days_gt28_per_yr_rep": float((j.rep > 28).groupby(j.index.year).sum().mean()),
              "mean_annual_max_x": float(j.x.groupby(j.index.year).max().mean()),
              "mean_annual_max_rep": float(j.rep.groupby(j.index.year).max().mean())}
res["cross_checks"] = cc
# threshold sensitivity (SST REP, meadow-weighted, 2016-2025): +/-0.5 C
late = rep_s[(rep_s.index.year >= 2016) & (rep_s.index.year <= 2025)]
res["sst_days_gt_threshold_2016_2025_sensitivity"] = {f"{t:g}": float((late > t).groupby(late.index.year).sum().mean()) for t in (27.5, 28, 28.5, 28.9, 29.4, 30)}
rep_err = xr.open_dataset(f"{D}/cmems/sst_rep.nc").analysis_error
ev = []
for k in w_rep.index:
    a, b = map(int, k.split("_")[1:])
    v = rep_err.isel(latitude=a, longitude=b)
    v = v.where(v.time.dt.month.isin([7, 8, 9])).values
    ev.append(v[np.isfinite(v) & (v >= 0)])
ev = np.concatenate(ev)
res["sst_rep_analysis_error_JAS_meadow_K"] = {"median": float(np.median(ev)), "p90": float(np.percentile(ev, 90))}
# period of record
res["records"] = {"sst_rep": [str(ds.time.values[0])[:10], str(rep_s.dropna().index.max().date())],
                  "reanalysis": [str(zt_top.dropna().index.min().date()), str(zt_top.dropna().index.max().date())]}
json.dump(res, open(f"{ROOT}/results_obs.json", "w"), indent=1, default=float)
flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} | {
    "trend_mhw_days_per_dec": f"{r['trend_mhw_days_per_dec']['ols']:.1f} [{r['trend_mhw_days_per_dec']['ols_lo']:.1f}, {r['trend_mhw_days_per_dec']['ols_hi']:.1f}]",
    "trend_tmax_per_dec": f"{r['trend_tmax_per_dec']['ols']:.2f} [{r['trend_tmax_per_dec']['ols_lo']:.2f}, {r['trend_tmax_per_dec']['ols_hi']:.2f}]",
    "trend_days_gt28_per_dec": f"{r['trend_days_gt28_per_dec']['ols']:.1f} [{r['trend_days_gt28_per_dec']['ols_lo']:.1f}, {r['trend_days_gt28_per_dec']['ols_hi']:.1f}]"} for r in tab])
flat.to_csv(f"{ROOT}/tables/summary_obs.csv", index=False)
pd.set_option("display.width", 250)
print(flat.T.to_string())
print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1))
for r in tab:
    print(r["group"], r["top5_years_mhw_days"], r["top5_years_tmax"], {k: round(v, 2) for k, v in r["trend_mhw_days_per_dec"].items()})
