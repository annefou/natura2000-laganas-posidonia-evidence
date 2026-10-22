"""Climate DT SST (avg_tos) at the meadows: evaluation vs observations, bias correction, projection.

Meadow weighting: each meadow HEALPix d17 cell (HCMR map) is assigned to the DT level-10 cell containing its
centre (spherical HEALPix, as the DT is delivered and as Polytope resolves points); site series = weighted mean
of DT cells by number of meadow cells.  The DT is a free-running coupled simulation: years are not in phase with
the real ones, so only statistics (climatology, variability, extremes, MHW frequency) are compared.
Evaluation period 1990-2025 (hist 1990-2014 + SSP3-7.0 2015-2025), obs = SST REP meadow-weighted.
Bias correction (applied only if the climatological bias is material): T_bc = T_dt - [clim_dt(doy) - clim_obs(doy)],
both 1990-2019 day-of-year means smoothed as in Hobday (11-day window + 31-day running mean).  Mean only, no
variance or quantile scaling.
Projection indicators use the observed MHW threshold (SST REP 1983-2012 baseline, Hobday), i.e. a fixed baseline.
Usage: pixi run python pipeline/climate/05_dt.py <ROOT>
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import healpix_geo.nested as hn
import mhw

ROOT = sys.argv[1]
D = f"{ROOT}/data"
TH = (28.0, 28.9, 30.0)
cells = pd.read_parquet(f"{D}/cell_metrics.parquet")
mc = cells[cells.meadow_ref].copy()
d10 = np.uint8(10)
mc["dt_cell"] = hn.lonlat_to_healpix(mc.lon.values, mc.lat.values, d10, ellipsoid="sphere")
mc["dt_cell_wgs84"] = hn.lonlat_to_healpix(mc.lon.values, mc.lat.values, d10, ellipsoid="WGS84")
w = mc.dt_cell.value_counts()


def load(model):
    out = {}
    for c in w.index:
        parts = []
        for f in sorted(glob.glob(f"{D}/dt_raw/tos/{model}_{c}_*.covjson")):
            j = json.load(open(f))
            for cov in j["coverages"]:
                t = pd.to_datetime([s.replace("Z", "") for s in cov["domain"]["axes"]["t"]["values"]])
                v = np.array([np.nan if x is None else x for x in cov["ranges"]["avg_tos"]["values"]], float)
                parts.append(pd.Series(v - 273.15, index=t))
        if parts:
            s = pd.concat(parts).sort_index()
            out[int(c)] = s[~s.index.duplicated()]
    return out


ds_obs = xr.open_dataset(f"{D}/daily_series_obs.nc")
wr = mc.rep_key.value_counts()
obs = (sum(ds_obs[k] * n for k, n in wr.items()) / wr.sum()).to_series().dropna()
seas_o, thr_o = mhw.climatology(obs.index, obs.values, (1983, 2012))      # fixed obs baseline for projections
seas_o9, _ = mhw.climatology(obs.index, obs.values, (1990, 2019))


def indicators(s, seas, thr):
    ev, inm, anom = mhw.detect(s.index, s.values, seas, thr)
    a = mhw.annual(s.index, s.values, inm, anom, ev, TH)
    return a


res = {"membership": {"n_meadow_cells": int(len(mc)), "dt_cells": {int(k): int(v) for k, v in w.items()},
                      "cells_whose_dt_cell_differs_if_WGS84_authalic": int((mc.dt_cell != mc.dt_cell_wgs84).sum())}}
models = {}
for model in ("IFS-NEMO", "IFS-FESOM"):
    ser = load(model)
    if not ser:
        res[model] = "no data retrieved"; continue
    nan_cells = {c: float(s.isna().mean()) for c, s in ser.items()}
    ok = {c: s for c, s in ser.items() if s.notna().mean() > 0.99}
    ww = w[[c for c in w.index if int(c) in ok]]
    site = sum(ok[int(c)] * n for c, n in ww.items()) / ww.sum()
    site = site.dropna()
    res[model] = {"period": [str(site.index.min().date()), str(site.index.max().date())], "n_days": int(len(site)),
                  "nan_fraction_by_dt_cell": nan_cells,
                  "meadow_cells_covered": int(ww.sum())}
    models[model] = site

# ---------------- evaluation 1990-2025 -----------------
ev_out = {}
o = obs.loc["1990":"2025"]
oa = indicators(o, *mhw.climatology(o.index, o.values, (1990, 2019)))
for model, s in models.items():
    x = s.loc["1990":"2025"]
    if len(x) < 365 * 30:
        ev_out[model] = f"insufficient data ({len(x)} days)"; continue
    seas_d, thr_d = mhw.climatology(x.index, x.values, (1990, 2019))
    xa = indicators(x, seas_d, thr_d)                     # own-baseline MHW statistics
    mo = o.groupby(o.index.month).mean(); mx = x.groupby(x.index.month).mean()
    anom_o = o - seas_o9[mhw.doy366(o.index)]; anom_x = x - seas_d[mhw.doy366(x.index)]
    ev_out[model] = {
        "mean_bias": float(x.mean() - o.mean()),
        "monthly_bias": {int(m): round(float(mx[m] - mo[m]), 2) for m in range(1, 13)},
        "std_daily_anomaly_ratio": float(anom_x.std() / anom_o.std()),
        "annual_max_mean_dt_vs_obs": [float(xa.t_max.mean()), float(oa.t_max.mean())],
        "annual_max_sd_dt_vs_obs": [float(xa.t_max.std()), float(oa.t_max.std())],
        "mhw_days_per_yr_own_baseline_1990_2019_dt_vs_obs": [float(xa.loc[1990:2019].mhw_days.mean()), float(oa.loc[1990:2019].mhw_days.mean())],
        "mhw_days_per_yr_2016_2025_own_baseline_dt_vs_obs": [float(xa.loc[2016:2025].mhw_days.mean()), float(oa.loc[2016:2025].mhw_days.mean())],
        "mhw_i_max_mean_dt_vs_obs": [float(xa.mhw_i_max.mean()), float(oa.mhw_i_max.mean())],
        "mhw_duration_mean_dt_vs_obs": [float(xa.mhw_days.sum() / max(xa.n_events.sum(), 1)), float(oa.mhw_days.sum() / max(oa.n_events.sum(), 1))],
        "trend_tmax_1990_2025_per_dec_dt_vs_obs": [mhw.trend(xa.index, xa.t_max), mhw.trend(oa.index, oa.t_max)],
        "trend_mean_1990_2025_per_dec_dt_vs_obs": [mhw.trend(xa.index, xa.t_mean), mhw.trend(oa.index, oa.t_mean)],
    }
res["evaluation_1990_2025"] = ev_out

# ---------------- bias correction + projection -----------------
proj, annual_tab, daily_out = {}, [], {}
for model, s in models.items():
    if not isinstance(ev_out.get(model), dict):
        continue
    x = s.loc["1990":"2025"]
    seas_d, _ = mhw.climatology(x.index, x.values, (1990, 2019))
    delta = seas_d - seas_o9
    for corr in ("raw", "bc"):
        y = s if corr == "raw" else s - delta[mhw.doy366(s.index)]
        a = indicators(y, seas_o, thr_o)
        a["model"], a["corr"] = model, corr
        annual_tab.append(a.reset_index(names="year"))
        if corr == "bc":
            daily_out[model] = y
        dec = {}
        for y0, y1 in ((1990, 2019), (2016, 2025), (2020, 2029), (2030, 2039), (2040, 2049)):
            b = a.loc[y0:y1]
            dec[f"{y0}-{y1}"] = {"mhw_days": [float(b.mhw_days.mean()), float(b.mhw_days.quantile(.1)), float(b.mhw_days.quantile(.9))],
                                 "t_max": [float(b.t_max.mean()), float(b.t_max.min()), float(b.t_max.max())],
                                 "days_gt_28": float(b.days_gt_28.mean()), "days_gt_28.9": float(b["days_gt_28.9"].mean()),
                                 "days_gt_30": float(b.days_gt_30.mean()),
                                 "years_tmax_gt28": int((b.t_max > 28).sum()), "years_tmax_gt28.9": int((b.t_max > 28.9).sum()),
                                 "years_tmax_gt30": int((b.t_max > 30).sum()),
                                 "years_icum_gt120": int((b.mhw_i_cum > 120).sum()), "n_years": int(len(b)),
                                 "max_run_gt28.9": int(b["run_gt_28.9"].max())}
        proj[f"{model}_{corr}"] = {"decades": dec, "trend_tmax_2015_2049": mhw.trend(a.loc[2015:2049].index, a.loc[2015:2049].t_max),
                                   "trend_mhw_days_2015_2049": mhw.trend(a.loc[2015:2049].index, a.loc[2015:2049].mhw_days)}
res["projection_fixed_obs_baseline_1983_2012"] = proj
# observed reference with the same definitions, meadow-weighted SST
oo = indicators(obs, seas_o, thr_o)
res["obs_same_definitions"] = {p: {"mhw_days": float(oo.loc[a:b].mhw_days.mean()), "t_max": float(oo.loc[a:b].t_max.mean()),
                                   "days_gt_28": float(oo.loc[a:b].days_gt_28.mean()), "years_tmax_gt28": int((oo.loc[a:b].t_max > 28).sum()),
                                   "n_years": int(len(oo.loc[a:b]))}
                               for p, (a, b) in {"1990-2019": (1990, 2019), "2016-2025": (2016, 2025)}.items()}
oo["model"], oo["corr"] = "OBS_SST_REP", "obs"
annual_tab.append(oo.reset_index(names="year"))
pd.concat(annual_tab, ignore_index=True).to_csv(f"{ROOT}/tables/annual_dt_and_obs_sst.csv", index=False)
# derived daily indicator series (bias-corrected site mean) kept internal
if daily_out:
    xr.Dataset({k.replace("-", "_"): ("time", v.reindex(pd.date_range("1990-01-01", "2049-12-31")).values.astype("float32"))
                for k, v in daily_out.items()}, coords={"time": pd.date_range("1990-01-01", "2049-12-31")},
               attrs={"note": "INTERNAL: derived from DestinE Climate DT; do not redistribute"}).to_netcdf(f"{D}/dt_site_bc_INTERNAL.nc")
json.dump(res, open(f"{ROOT}/results_dt.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3))[:9000])
