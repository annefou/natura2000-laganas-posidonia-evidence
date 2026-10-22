"""Probe DestinE Climate DT (generation 2) for ocean SST at GR2210002 via Polytope feature extraction.

Request keys copied from polytope-examples climate-dt/ (commit 3a13f05, 2026-10-07):
  explorer/polytope_zarr.py (base_request, clte date/time fields, timeseries feature with time_axis "date",
  per-model address: IFS-NEMO -> polytope.mn5..., others -> polytope.lumi...), explorer/destine_portfolio.py
  (o2d param avg_tos, daily mean, K; o3d avg_thetao, levels 1..75).
Usage: pixi run python pipeline/climate/10_dt_probe.py <ROOT> <model> <experiment> <date or date/to/date> [param] [levtype] [levelist]
Writes raw covjson to data/dt_raw/ (internal only, not redistributed).
"""
import sys, os, json, time
from polytope.api import Client

ROOT, model, exp, date = sys.argv[1:5]
param = sys.argv[5] if len(sys.argv) > 5 else "avg_tos"
levtype = sys.argv[6] if len(sys.argv) > 6 else "o2d"
levelist = sys.argv[7] if len(sys.argv) > 7 else None
POINT = [37.700, 20.900]       # lat, lon: centre of Laganas Bay meadows (committed, not fetched)
address = "polytope.mn5.apps.dte.destination-earth.eu" if model == "IFS-NEMO" else "polytope.lumi.apps.dte.destination-earth.eu"
req = {
    "class": "d1", "dataset": "climate-dt", "type": "fc", "expver": "0001", "generation": "2",
    "realization": "1",
    "activity": "baseline" if exp in ("hist", "cont") else "projections",
    "experiment": exp, "levtype": levtype, "resolution": "high", "stream": "clte",
    "model": model, "param": param, "date": date, "time": "0000",
    "feature": {"type": "timeseries", "points": [POINT], "axes": ["latitude", "longitude"], "time_axis": "date"},
}
if levelist:
    req["levelist"] = levelist
os.makedirs(f"{ROOT}/data/dt_raw", exist_ok=True)
tag = f"{model}_{exp}_{param}_{(levelist or 'sfc').replace('/', '-')}_{date.replace('/', '-')}"
out = f"{ROOT}/data/dt_raw/{tag}.covjson"
c = Client(address=address, quiet=True)
t0 = time.time()
try:
    c.retrieve("destination-earth", req, out)
    print("OK", tag, f"{time.time()-t0:.0f}s", os.path.getsize(out), "bytes")
except Exception as e:
    print("FAIL", tag, type(e).__name__, str(e)[:2000])
json.dump({"address": address, "request": req}, open(f"{ROOT}/data/dt_raw/{tag}.request.json", "w"), indent=1)
