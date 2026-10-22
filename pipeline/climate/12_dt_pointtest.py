"""Test which point specs the Polytope timeseries feature accepts. Usage: 12_dt_pointtest.py <ROOT> <MODEL> <lat,lon;lat,lon...>"""
import sys, os
from polytope.api import Client
ROOT, MODEL, pts = sys.argv[1], sys.argv[2], sys.argv[3]
P = [[float(a) for a in p.split(",")] for p in pts.split(";")]
address = "polytope.mn5.apps.dte.destination-earth.eu" if MODEL == "IFS-NEMO" else "polytope.lumi.apps.dte.destination-earth.eu"
req = {"class": "d1", "dataset": "climate-dt", "type": "fc", "expver": "0001", "generation": "2", "realization": "1",
       "activity": "baseline", "experiment": "hist", "levtype": "o2d", "resolution": "high", "stream": "clte",
       "model": MODEL, "param": "avg_tos", "date": os.environ.get("DTDATE", "20100801/to/20100802"), "time": "0000",
       "feature": {"type": "timeseries", "points": P, "axes": ["latitude", "longitude"], "time_axis": "date"}}
fn = f"{ROOT}/data/dt_raw/test/{MODEL}_{abs(hash(pts))}.covjson"
try:
    Client(address=address, quiet=True).retrieve("destination-earth", req, fn); print("OK", pts, open(fn).read()[:0])
except Exception as e:
    print("FAIL", pts, str(e).split("Details:")[-1][:150].replace("\n", " "))
