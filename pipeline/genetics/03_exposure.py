"""Observed heat exposure at each genotyped meadow: satellite SST and reanalysis temperature at the sampling depth.

Usage: pixi run python pipeline/genetics/03_exposure.py <ROOT>
Inputs: data/inputs/sites.csv, data/cmems/<SITE>_{sst_rep,phy_my_temp,phy_static}.nc (02_download_cmems.py).
Method (same conventions as pipeline/climate/03_observed.py):
- Assignment: the product grid cell containing the published site coordinate (nearest centre in lon and lat).
  If that cell is land/NaN (SST: <95 % valid days; reanalysis: level nearest the sampling depth not wet), the nearest
  valid sea cell (great circle) is used and flagged with its distance. No interpolation.
- Depth: reanalysis thetao at the model level whose centre is nearest the published sampling depth (4-15 m);
  sampling depth is a single value per meadow, the meadow's real depth range is unknown.
- MHW after Hobday et al. 2016 (pipeline/climate/mhw.py): baseline 1983-2012 for SST REP, 1987-2016 for reanalysis.
- Indicators per calendar year: MHW days, days with T > 28 C and > 28.9 C, annual maximum. Trend of MHW days
  1982(SST)/1987(reanalysis)-2025 by Theil-Sen and OLS with Newey-West 95 % CI (per decade). 2026 is partial
  (SST to 2026-09-22, reanalysis to 2026-09-30) and is not used in means or trends.
- HEALPix cell of each meadow: healpix-geo, WGS84 ellipsoid, NESTED, depth 12 (cell ~1.5 km), matching the 0.01 deg
  precision of the published coordinates.
Outputs: data/exposure_by_meadow.csv, data/annual_by_meadow.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "climate"))
import numpy as np, pandas as pd, xarray as xr
import healpix_geo.nested as hn
import mhw

ROOT = sys.argv[1]
D = f"{ROOT}/data"
sites = pd.read_csv(f"{D}/inputs/sites.csv")
HPX_DEPTH = np.uint8(12)
LAST = 2025
TH = (28.0, 28.9, 30.0)


def gc(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def assign(lon, lat, glon, glat, valid):
    ix, iy = int(np.abs(glon - lon).argmin()), int(np.abs(glat - lat).argmin())
    if valid[iy, ix]:
        return iy, ix, False, gc(lon, lat, glon[ix], glat[iy])
    vy, vx = np.nonzero(valid)
    dd = gc(lon, lat, glon[vx], glat[vy])
    j = dd.argmin()
    return int(vy[j]), int(vx[j]), True, float(dd[j])


def indicators(t, x, base):
    seas, thr = mhw.climatology(t, x, base)
    ev, inm, anom = mhw.detect(t, x, seas, thr)
    return mhw.annual(t, x, inm, anom, ev, TH)


rows, annual = [], []
for _, s in sites.iterrows():
    r = dict(site_code=s.site_code)
    r["hpx12"] = int(hn.lonlat_to_healpix(np.array([s.lon]), np.array([s.lat]), HPX_DEPTH, ellipsoid="WGS84")[0])
    # --- SST REP
    rep = xr.open_dataset(f"{D}/cmems/{s.site_code}_sst_rep.nc")
    sst = rep.analysed_sst.values - 273.15
    iy, ix, fb, dist = assign(s.lon, s.lat, rep.longitude.values, rep.latitude.values, np.isfinite(sst).mean(0) > 0.95)
    r.update(sst_lon=float(rep.longitude[ix]), sst_lat=float(rep.latitude[iy]), sst_fallback=fb, sst_dist_km=dist)
    a_sst = indicators(rep.time.values, sst[:, iy, ix].astype(float), (1983, 2012))
    # --- reanalysis at sampling depth
    phy = xr.open_dataset(f"{D}/cmems/{s.site_code}_phy_my_temp.nc")
    st = xr.open_dataset(f"{D}/cmems/{s.site_code}_phy_static.nc")
    z = phy.depth.values
    lev = int(np.abs(z - s.depth_m).argmin())
    th = phy.thetao.isel(depth=lev).values
    iy, ix, fb, dist = assign(s.lon, s.lat, phy.longitude.values, phy.latitude.values, np.isfinite(th[:400]).mean(0) > 0.95)
    dep = st.deptho.sel(longitude=phy.longitude[ix], latitude=phy.latitude[iy], method="nearest")
    r.update(phy_lon=float(phy.longitude[ix]), phy_lat=float(phy.latitude[iy]), phy_fallback=fb, phy_dist_km=dist,
             phy_level_m=float(z[lev]), phy_model_depth_m=float(dep))
    a_zt = indicators(phy.time.values, th[:, iy, ix].astype(float), (1987, 2016))
    for tag, a, y0 in (("sst", a_sst, 1982), ("zt", a_zt, 1987)):
        full = a.loc[y0:LAST]
        tr = mhw.trend(full.index, full.mhw_days)
        late, early = a.loc[2016:LAST], a.loc[y0:2015]
        r.update({f"{tag}_mhw_days_trend_sen": tr["sen"], f"{tag}_mhw_days_trend_sen_lo": tr["sen_lo"],
                  f"{tag}_mhw_days_trend_sen_hi": tr["sen_hi"], f"{tag}_mhw_days_trend_ols": tr["ols"],
                  f"{tag}_mhw_days_trend_ols_p": tr["ols_p"], f"{tag}_trend_period": f"{y0}-{LAST}",
                  f"{tag}_mhw_days_{y0}_2015": early.mhw_days.mean(), f"{tag}_mhw_days_2016_2025": late.mhw_days.mean(),
                  f"{tag}_days_gt28_2016_2025": late.days_gt_28.mean(), f"{tag}_days_gt28.9_2016_2025": late["days_gt_28.9"].mean(),
                  f"{tag}_days_gt28_{y0}_2015": early.days_gt_28.mean(),
                  f"{tag}_tmax_2016_2025": late.t_max.mean(),
                  f"{tag}_mhw_days_2024": a.loc[2024].mhw_days, f"{tag}_days_gt28_2024": a.loc[2024].days_gt_28,
                  f"{tag}_tmax_2024": a.loc[2024].t_max,
                  f"{tag}_tmax_record_year": int(full.t_max.idxmax()), f"{tag}_tmax_record": full.t_max.max(),
                  f"{tag}_mhw_days_record_year": int(full.mhw_days.idxmax()), f"{tag}_mhw_days_record": full.mhw_days.max(),
                  f"{tag}_2024_tmax_rank": int((full.t_max > a.loc[2024].t_max).sum() + 1),
                  f"{tag}_2024_mhw_days_rank": int((full.mhw_days > a.loc[2024].mhw_days).sum() + 1),
                  f"{tag}_n_years": len(full)})
        annual.append(a.reset_index(names="year").assign(site_code=s.site_code, source=tag))
    rows.append(r)
    print(s.site_code, f"SST fb={r['sst_fallback']} {r['sst_dist_km']:.1f} km; phy fb={r['phy_fallback']} {r['phy_dist_km']:.1f} km "
          f"level {r['phy_level_m']:.1f} m (sampled {s.depth_m} m, model depth {r['phy_model_depth_m']:.0f} m)", flush=True)
ex = sites.merge(pd.DataFrame(rows), on="site_code")
ex.to_csv(f"{D}/exposure_by_meadow.csv", index=False)
pd.concat(annual, ignore_index=True).to_csv(f"{D}/annual_by_meadow.csv", index=False)
print(ex[["site_code", "depth_m", "zt_mhw_days_2016_2025", "zt_days_gt28_2016_2025", "zt_mhw_days_trend_sen", "zt_tmax_2024",
          "zt_tmax_record_year", "sst_mhw_days_2016_2025", "sst_days_gt28_2016_2025"]].round(2).to_string())
