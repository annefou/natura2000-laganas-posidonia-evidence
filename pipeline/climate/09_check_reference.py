"""Check our marine heatwave detection (mhw.py) against the reference implementation of Hobday et al. (2016):
E. C. J. Oliver, marineHeatWaves (https://github.com/ecjoliver/marineHeatWaves), pinned commit below.
The reference module is downloaded at run time (its licence is not stated as an SPDX identifier, so it is not
vendored here) and two NumPy-2 incompatibilities are patched in memory: np.NaN -> np.nan and a size-1 array
assigned to a scalar (doy lookup).  Detection logic is untouched.
Usage: pixi run python pipeline/climate/09_check_reference.py <ROOT>  -> results_reference_check.json
Exits with status 1 if day-by-day agreement on any series is below 0.99.
"""
import sys, os, json, types, datetime as dt, urllib.request, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import mhw

COMMIT = "d7292bf08ade0af213fa760b0d7e4adfe5f52894"
URL = f"https://raw.githubusercontent.com/ecjoliver/marineHeatWaves/{COMMIT}/marineHeatWaves.py"
ROOT = sys.argv[1]
warnings.simplefilter("ignore")

src = urllib.request.urlopen(URL, timeout=120).read().decode()
src = src.replace("np.NaN", "np.nan").replace(
    "doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]",
    "doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])][0]")
ref = types.ModuleType("marineHeatWaves")
exec(compile(src, URL, "exec"), ref.__dict__)

ds = xr.open_dataset(f"{ROOT}/data/daily_series_obs.nc")
keys = [k for k in ds.data_vars if k.startswith("REP_")][:1] + [k for k in ds.data_vars if k.startswith("ZT_")][:1]
out, ok = {"reference": URL}, True
for k in keys:
    s = ds[k].to_series()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated()].dropna()
    s = s[s.index.year <= 2025]
    full = pd.date_range(s.index.min(), "2025-12-31", freq="D")
    s = s.reindex(full).interpolate(limit_direction="both")   # the reference needs a gap-free daily series
    t, x = s.index, s.values.astype(float)
    base = (1983, 2012) if k.startswith("REP") else (1987, 2016)
    seas, thr = mhw.climatology(t, x, base)
    ev, inm, _ = mhw.detect(t, x, seas, thr)
    m, clim = ref.detect(np.array([dt.date(i.year, i.month, i.day).toordinal() for i in t]), x, climatologyPeriod=list(base))
    rf = np.zeros(len(t), bool)
    for a, b in zip(m["index_start"], m["index_end"]):
        rf[a:b + 1] = True
    agree = float((inm == rf).mean())
    ok &= agree >= 0.99
    out[k] = {"period": [str(t[0].date()), str(t[-1].date())], "threshold_max_abs_diff_degC": float(np.nanmax(np.abs(clim["thresh"] - thr[mhw.doy366(t)]))),
              "events_ours": int(len(ev)), "events_ref": int(m["n_events"]), "mhw_days_ours": int(inm.sum()),
              "mhw_days_ref": int(rf.sum()), "daily_agreement": round(agree, 5)}
json.dump(out, open(f"{ROOT}/results_reference_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
