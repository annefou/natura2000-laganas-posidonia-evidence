"""Map and reference on HEALPix cells (healpix-geo, WGS84 ellipsoid, NESTED), the service's output grid.

The 10 m Sentinel-2 grid (EPSG:32634) is only the working grid: each pixel centre is assigned to the
HEALPix cell that contains it, and each cell reports the Posidonia share in the reference and the map.
Usage: pixi run python pipeline/accuracy/05_healpix.py <ROOT = work/accuracy> [depth]   -> results_healpix.json, data/cells_d<depth>.parquet
"""
import json, sys
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
import healpix_geo.nested as hn

ROOT = sys.argv[1]
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 17   # ~48 m cells, ~23 pixels each
d = np.load(f"{ROOT}/data/maps.npz")
W = int(d["W"])
with rasterio.open(f"{ROOT}/data/s2_median_2025.tif") as src:
    T = src.transform
rows, cols = np.divmod(d["idx"], W)
x, y = rasterio.transform.xy(T, rows, cols, offset="center")
lon, lat = Transformer.from_crs(32634, 4326, always_xy=True).transform(np.asarray(x), np.asarray(y))
cell = hn.lonlat_to_healpix(lon, lat, np.uint8(DEPTH), ellipsoid="WGS84")

df = pd.DataFrame({"cell": cell, "ref": d["y"], "map_oof": d["oofA"], "map": d["full"]})
cells = df.groupby("cell").agg(n=("ref", "size"), ref_share=("ref", "mean"),
                               map_oof_share=("map_oof", "mean"), map_share=("map", "mean")).reset_index()
cells = cells[cells.n >= 10]                       # drop cells cut by the domain edge
cells.to_parquet(f"{ROOT}/data/cells_d{DEPTH}.parquet", index=False)

ref_c, map_c = cells.ref_share >= 0.5, cells.map_oof_share >= 0.5
out = {
    "grid": {"library": "healpix-geo", "ellipsoid": "WGS84", "indexing": "nested", "depth": DEPTH,
             "approx_cell_edge_m": float(np.sqrt(4 * np.pi * 6371008.8**2 / (12 * 4**DEPTH)))},
    "cells": int(len(cells)),
    "cell_overall_accuracy_majority": float((ref_c == map_c).mean()),
    "cell_share_mae": float((cells.map_oof_share - cells.ref_share).abs().mean()),
    "cell_share_bias": float((cells.map_oof_share - cells.ref_share).mean()),
}
json.dump(out, open(f"{ROOT}/results_healpix.json", "w"), indent=2)
print(json.dumps(out, indent=2))
