"""Observed thermal exposure of Posidonia meadow cells (HEALPix d17) at GR2210002.

Sources (data/cmems): SST L4 REP 0.05 deg (1982-01-01..2026-09-22), SST L4 NRT 0.01 deg (2008-..., cross-check),
Med physics reanalysis 1/24 deg daily thetao (0-56 m) and bottomT (1987-01-01..2026-09-30).
Assignment: each HEALPix cell takes the value of the data grid cell that contains its centre (regular grid ->
nearest centre in lon and in lat separately).  No smoothing or interpolation.  If that grid cell is land/NaN
in the product, the nearest valid sea grid cell (great-circle) is used and the cell is flagged "fallback".
Temperature at the meadow = reanalysis thetao at the model level whose centre is nearest the meadow depth
(EMODnet), capped at the deepest wet level of the column (flag "below_model_bottom" when meadow is deeper).
MHW (Hobday 2016): baseline 1983-2012 for SST REP; 1987-2016 for the reanalysis (it starts 1987).
Usage: pixi run python pipeline/climate/03_observed.py <ROOT>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import mhw

ROOT = sys.argv[1]
D = f"{ROOT}/data"
cells = pd.read_parquet(f"{D}/meadow_cells.parquet")
TH = (28.0, 28.9, 30.0)


def gc(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def assign(glon, glat, valid):
    """valid: 2-D bool [lat, lon]. Returns (ilat, ilon, fallback, dist_km) per cell."""
    ix = np.abs(cells.lon.values[:, None] - glon[None, :]).argmin(1)
    iy = np.abs(cells.lat.values[:, None] - glat[None, :]).argmin(1)
    fb = ~valid[iy, ix]
    vy, vx = np.nonzero(valid)
    dist = np.zeros(len(cells))
    for k in np.nonzero(fb)[0]:
        dd = gc(cells.lon.values[k], cells.lat.values[k], glon[vx], glat[vy])
        j = dd.argmin(); iy[k], ix[k], dist[k] = vy[j], vx[j], dd[j]
    return iy, ix, fb, dist


series, src = {}, []
# --- SST REP
rep = xr.open_dataset(f"{D}/cmems/sst_rep.nc")
sst = rep.analysed_sst.values - 273.15
iy, ix, fb, dist = assign(rep.longitude.values, rep.latitude.values, np.isfinite(sst).mean(0) > 0.95)
cells["rep_key"] = [f"REP_{a}_{b}" for a, b in zip(iy, ix)]
cells["rep_fallback"], cells["rep_fb_km"] = fb, dist
for k in cells.rep_key.unique():
    a, b = map(int, k.split("_")[1:])
    series[k] = ("SST REP", rep.time.values, sst[:, a, b], (1983, 2012), float(rep.longitude[b]), float(rep.latitude[a]), 0.0)
# --- SST NRT (cross-check only)
nrt = xr.open_dataset(f"{D}/cmems/sst_nrt.nc")
sn = nrt.analysed_sst.values - 273.15
iy, ix, fb, dist = assign(nrt.longitude.values, nrt.latitude.values, np.isfinite(sn).mean(0) > 0.95)
cells["nrt_key"] = [f"NRT_{a}_{b}" for a, b in zip(iy, ix)]
cells["nrt_fallback"] = fb
for k in cells[cells.meadow_ref].nrt_key.unique():
    a, b = map(int, k.split("_")[1:])
    series[k] = ("SST NRT", nrt.time.values, sn[:, a, b], None, float(nrt.longitude[b]), float(nrt.latitude[a]), 0.0)
# --- Reanalysis: T at meadow depth and bottomT
phy = xr.open_dataset(f"{D}/cmems/phy_my_temp.nc")
th = phy.thetao.values          # time, depth, lat, lon
bt = phy.bottomT.values
z = phy.depth.values
wet = np.isfinite(th[:400]).mean(0) > 0.95          # depth, lat, lon
st = xr.open_dataset(f"{D}/cmems/phy_my_static.nc")
iy, ix, fb, dist = assign(phy.longitude.values, phy.latitude.values, wet[0])
nwet = wet.sum(0)[iy, ix]
lev = np.abs(cells.depth_m.values[:, None] - z[None, :]).argmin(1)
cells["below_model_bottom"] = cells.depth_m.values > z[np.maximum(nwet - 1, 0)] + 0.5 * np.r_[np.diff(z), 5][np.maximum(nwet - 1, 0)]
lev = np.minimum(lev, nwet - 1)
cells["phy_fallback"], cells["phy_fb_km"] = fb, dist
cells["phy_model_depth_m"] = st.deptho.values[iy, ix]
cells["phy_level_m"] = z[lev]
cells["zt_key"] = [f"ZT_{a}_{b}_{l}" for a, b, l in zip(iy, ix, lev)]
cells["bt_key"] = [f"BT_{a}_{b}" for a, b in zip(iy, ix)]
for k in cells.zt_key.unique():
    a, b, l = map(int, k.split("_")[1:])
    series[k] = ("T at meadow depth (reanalysis)", phy.time.values, th[:, l, a, b], (1987, 2016),
                 float(phy.longitude[b]), float(phy.latitude[a]), float(z[l]))
for k in cells.bt_key.unique():
    a, b = map(int, k.split("_")[1:])
    series[k] = ("bottomT (reanalysis)", phy.time.values, bt[:, a, b], (1987, 2016),
                 float(phy.longitude[b]), float(phy.latitude[a]), float(st.deptho.values[a, b]))

# --- MHW + annual indicators per distinct series
ann_all, ev_all, clim_all, meta = [], [], {}, []
for k, (name, t, x, base, lo, la, zz) in series.items():
    x = np.asarray(x, float)
    if base is not None:
        seas, thr = mhw.climatology(t, x, base)
        ev, inm, anom = mhw.detect(t, x, seas, thr)
        clim_all[k] = (seas, thr)
    else:
        ev, inm, anom = pd.DataFrame(), np.zeros(len(x), bool), np.full(len(x), np.nan)
    a = mhw.annual(t, x, inm, anom, ev, TH)
    a["key"] = k; a["source"] = name
    ann_all.append(a.reset_index(names="year"))
    if len(ev):
        ev["key"] = k; ev_all.append(ev)
    meta.append(dict(key=k, source=name, lon=lo, lat=la, depth_m=zz, base=str(base),
                     n_meadow_cells=int(((cells[[c for c in cells if c.endswith("_key")]] == k).any(axis=1) & cells.meadow_ref).sum())))
ann = pd.concat(ann_all, ignore_index=True)
ann.to_csv(f"{D}/annual_by_series.csv", index=False)
pd.concat(ev_all, ignore_index=True).to_csv(f"{D}/mhw_events_by_series.csv", index=False)
pd.DataFrame(meta).to_csv(f"{D}/series_meta.csv", index=False)

# daily series archive (NetCDF, self-describing)
dss = []
for k, (name, t, x, base, lo, la, zz) in series.items():
    da = xr.DataArray(np.asarray(x, "float32"), dims="time", coords={"time": t}, name=k,
                      attrs={"source": name, "lon": lo, "lat": la, "depth_m": zz, "units": "degC", "mhw_baseline": str(base)})
    dss.append(da)
xr.merge(dss, join="outer").to_netcdf(f"{D}/daily_series_obs.nc")
np.save(f"{D}/_clim_keys.npy", np.array(list(clim_all)))  # keys only; climatologies saved as NetCDF below
xr.Dataset({k: (("doy", "stat"), np.c_[v[0], v[1]]) for k, v in clim_all.items()},
           coords={"doy": np.arange(1, 367), "stat": ["seas_mean", "thresh_p90"]}).to_netcdf(f"{D}/mhw_climatology_obs.nc")

# --- per-cell metrics (meadow and non-meadow) for two periods
def per_key(src_key_col, period, cols):
    a = ann[(ann.year >= period[0]) & (ann.year <= period[1])].groupby("key")[cols].mean()
    return cells[[src_key_col]].join(a, on=src_key_col)[cols]

full = ["mhw_days", "mhw_i_cum", "t_max", "days_gt_28", "days_gt_28.9", "days_gt_30", "run_gt_28.9"]
for pname, per in {"1987_2016": (1987, 2016), "2016_2025": (2016, 2025)}.items():
    for sk, tag in (("rep_key", "sst"), ("zt_key", "zt")):
        m = per_key(sk, per, full)
        for c in full:
            cells[f"{tag}_{c}_{pname}"] = m[c].values
yrs_gt28 = ann.assign(gt=ann.t_max > 28).query("year <= 2025").groupby("key").gt.sum()
cells["zt_years_tmax_gt28_1987_2025"] = cells.zt_key.map(yrs_gt28)
cells["sst_years_tmax_gt28_1987_2025"] = cells.rep_key.map(ann.assign(gt=ann.t_max > 28).query("1987 <= year <= 2025").groupby("key").gt.sum())
cells.to_parquet(f"{D}/cell_metrics.parquet", index=False)

# --- summary
mc = cells[cells.meadow_ref]
res = {"n_meadow_cells": int(len(mc)), "n_series": {s: int(sum(1 for v in series.values() if v[0] == s)) for s in {v[0] for v in series.values()}},
       "assignment": {"rep_fallback_cells": int(mc.rep_fallback.sum()), "rep_fallback_km_max": float(mc.rep_fb_km.max()),
                      "phy_fallback_cells": int(mc.phy_fallback.sum()), "phy_fallback_km_max": float(mc.phy_fb_km.max()),
                      "nrt_fallback_cells": int(mc.nrt_fallback.sum()),
                      "below_model_bottom_cells": int(mc.below_model_bottom.sum())}}
json.dump(res, open(f"{ROOT}/results_obs_assign.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print(pd.DataFrame(meta).to_string())
