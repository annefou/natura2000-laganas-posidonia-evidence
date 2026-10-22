"""Meadow cells on HEALPix depth 17 (healpix-geo, WGS84, NESTED) with centre lon/lat and depth.

Inputs (from ../accuracy): data/cells_d17.parquet (ref_share = Posidonia share in HCMR map GR000019,
map_share = Sentinel-2 map), data/maps.npz (10 m grid: idx, depth = EMODnet DTM depth in m, positive down),
data/s2_median_2025.tif (grid transform only).
Meadow cell = ref_share >= 0.5 (field-based HCMR map, primary); map_share >= 0.5 kept as alternative flag.
Depth of a cell = median of the 10 m pixel depths whose centres fall in the cell (same assignment as 05_healpix.py).
Usage: pixi run python pipeline/climate/02_cells.py <ROOT>  -> data/meadow_cells.parquet
"""
import sys
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
import healpix_geo.nested as hn

ROOT = sys.argv[1]
ACC = f"{ROOT}/../accuracy"
D = np.uint8(17)
cells = pd.read_parquet(f"{ACC}/data/cells_d17.parquet")
m = np.load(f"{ACC}/data/maps.npz")
W = int(m["W"])
with rasterio.open(f"{ACC}/data/s2_median_2025.tif") as src:
    T = src.transform
rows, cols = np.divmod(m["idx"], W)
x, y = rasterio.transform.xy(T, rows, cols, offset="center")
lon, lat = Transformer.from_crs(32634, 4326, always_xy=True).transform(np.asarray(x), np.asarray(y))
pc = hn.lonlat_to_healpix(lon, lat, D, ellipsoid="WGS84")
dep = pd.DataFrame({"cell": pc, "depth": m["depth"].ravel()[m["idx"]]}).groupby("cell").depth.median()
cells = cells.merge(dep.rename("depth_m"), left_on="cell", right_index=True, how="left")
clon, clat = hn.healpix_to_lonlat(cells.cell.values.astype(np.uint64), D, ellipsoid="WGS84")
cells["lon"], cells["lat"] = clon, clat
cells["meadow_ref"] = cells.ref_share >= 0.5
cells["meadow_map"] = cells.map_share >= 0.5
cells["band"] = pd.cut(cells.depth_m, [-np.inf, 10, 20, 40], labels=["0-10 m", "10-20 m", "20-40 m"])
cells.to_parquet(f"{ROOT}/data/meadow_cells.parquet", index=False)
mc = cells[cells.meadow_ref]
print("all cells", len(cells), "meadow_ref", len(mc), "meadow_map", int(cells.meadow_map.sum()))
print(mc.band.value_counts().sort_index())
print("lon", mc.lon.min(), mc.lon.max(), "lat", mc.lat.min(), mc.lat.max(), "depth", mc.depth_m.describe().round(1).to_dict())
