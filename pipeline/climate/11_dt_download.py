"""Daily ocean SST (avg_tos, o2d, clte stream) from DestinE Climate DT generation 2 at one DT HEALPix cell
(level 10, nside 1024, spherical HEALPix as delivered) that contains HCMR-mapped Posidonia cells of GR2210002.

Keys follow polytope-examples climate-dt/explorer/polytope_zarr.py + destine_portfolio.py (commit 3a13f05):
  hist 1990-2014: activity "baseline", experiment "hist"; SSP3-7.0 2015-2049: activity "projections".
  IFS-NEMO -> polytope.mn5..., IFS-FESOM -> polytope.lumi....  Server-side feature: timeseries at the cell
  centre (centres committed below, computed with healpix_geo sphere centres). No regridding.
A single request with all 6 points failed with HTTP 400 "list index out of range" (2026-10-22), so one point per
request, in <=10-year chunks.
Raw covjson stays in data/dt_raw/ (internal; Climate DT redistribution restricted).
Usage: pixi run python pipeline/climate/11_dt_download.py <ROOT> <MODEL> <CELL>
"""
import sys, os, json, time
from polytope.api import Client

ROOT, MODEL, CELL = sys.argv[1], sys.argv[2], int(sys.argv[3])
CELLS = {587822: (37.6385, 20.8740), 587823: (37.6856, 20.9180), 587832: (37.6856, 21.0059),
         587834: (37.7327, 20.9619), 587908: (37.6856, 20.8301), 587909: (37.7327, 20.8740)}
CHUNKS = [("hist", 1990, 1999), ("hist", 2000, 2009), ("hist", 2010, 2014),
          ("SSP3-7.0", 2015, 2024), ("SSP3-7.0", 2025, 2034), ("SSP3-7.0", 2035, 2042), ("SSP3-7.0", 2043, 2049)]
address = "polytope.mn5.apps.dte.destination-earth.eu" if MODEL == "IFS-NEMO" else "polytope.lumi.apps.dte.destination-earth.eu"
OUT = f"{ROOT}/data/dt_raw/tos"
os.makedirs(OUT, exist_ok=True)
c = Client(address=address, quiet=True)
la, lo = CELLS[CELL]
for exp, y0, y1 in CHUNKS:
    fn = f"{OUT}/{MODEL}_{CELL}_{exp}_{y0}_{y1}.covjson"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        continue
    req = {"class": "d1", "dataset": "climate-dt", "type": "fc", "expver": "0001", "generation": "2",
           "realization": "1", "activity": "baseline" if exp == "hist" else "projections",
           "experiment": exp, "levtype": "o2d", "resolution": "high", "stream": "clte", "model": MODEL,
           "param": "avg_tos", "date": f"{y0}0101/to/{y1}1231", "time": "0000",
           "feature": {"type": "timeseries", "points": [[la, lo]], "axes": ["latitude", "longitude"], "time_axis": "date"}}
    json.dump({"address": address, "request": req}, open(fn.replace(".covjson", ".request.json"), "w"))
    for attempt in range(3):
        t0 = time.time()
        try:
            c.retrieve("destination-earth", req, fn)
            print("OK", MODEL, CELL, exp, y0, y1, f"{time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:
            msg = str(e).split("Details:")[-1][:300].replace("\n", " ")
            print("FAIL", MODEL, CELL, exp, y0, y1, attempt, type(e).__name__, msg, flush=True)
            time.sleep(30)
