"""Step 1 (HEALPix-native): processing grid, overlap weights and references on depth-19 cells.

Grid: healpix-geo, WGS84 ellipsoid, NESTED, depth 19 (~12.4 m edge). The window is the one of
pipeline/accuracy/01_composite.py (site bbox + 500 m, EPSG:32634) snapped outward to 60 m so the native
Sentinel-2 10/20/60 m grids all cover it; only cells lying wholly inside the window are kept.

Writes <ROOT>/data/:
  cells.parquet       one row per cell: id, centre lon/lat, depth (EMODnet, conservative area mean),
                      site_share, share of each GR000019 class, refc (plurality class), pos_share,
                      sg2015_share
  weights_{10,20,60}.parquet   exact cell x native-pixel area-overlap weights for the S2 grids
  grid.json           window, grid sizes, cell area, self-tests of conservation
References (shared inputs, not re-downloaded): work/accuracy/data/ (00_fetch_inputs.py).
  * GR000019 (HCMR 1999-2002) and the 2015 survey: area share per cell from healpix-geo polygon_coverage
    (fully-covered flag) + exact intersection for partially covered cells; the cell takes the class with the
    largest share, "unmapped" counting as a class (plurality; = majority whenever a share exceeds 0.5).
  * site mask: site area share > 0.5.
  * bathymetry: conservative area mean of the EMODnet DTM pixels (lon/lat grid) overlapping the cell.
Usage: pixi run python pipeline/accuracy_healpix/00_grid.py <ROOT = work/accuracy_healpix> [SHARED = work/accuracy/data]
"""
import json, os, sys, time
import numpy as np, pandas as pd, geopandas as gpd, rasterio, shapely
import healpix_geo.nested as hn
from common import (DEPTH, ELLIPSOID, UTM, authalic_radius, cell_area_m2, window_cells, grid_weights,
                    conservative_mean, polygon_share)

ROOT = sys.argv[1]
SH = sys.argv[2] if len(sys.argv) > 2 else "work/accuracy/data"
os.makedirs(f"{ROOT}/data", exist_ok=True)
t0 = time.time()

site = gpd.read_file(f"{SH}/site.geojson").to_crs(UTM)
x0, y0, x1, y1 = site.total_bounds
X0, Y0 = np.floor((x0 - 500) / 60) * 60, np.floor((y0 - 500) / 60) * 60
X1, Y1 = np.ceil((x1 + 500) / 60) * 60, np.ceil((y1 + 500) / 60) * 60
ids, vlon, vlat, quads = window_cells((X0, Y0, X1, Y1))
N = len(ids)
print("cells", N, f"{time.time()-t0:.0f}s", flush=True)

# --- native S2 grids (exact overlap weights)
grids = {}
for res in (10, 20, 60):
    W, H = int((X1 - X0) / res), int((Y1 - Y0) / res)
    fw = f"{ROOT}/data/weights_{res}.parquet"
    if os.path.exists(fw):                 # cache (deterministic from the window)
        ci = pd.read_parquet(fw, columns=["ci"])["ci"].to_numpy()
    else:
        ci, pi, w = grid_weights(quads, X0, Y1, res, res, W, H)
        pd.DataFrame({"ci": ci, "pi": pi, "w": w}).to_parquet(fw, index=False)
    grids[res] = {"W": W, "H": H, "n_pairs": int(ci.size), "mean_pixels_per_cell": float(ci.size / N)}
    print("weights", res, grids[res], f"{time.time()-t0:.0f}s", flush=True)

# --- conservation self-test on the real 10 m grid (random field and constant field)
ci, pi, w = (pd.read_parquet(f"{ROOT}/data/weights_10.parquet")[c].to_numpy() for c in ("ci", "pi", "w"))
A_plane = shapely.area(shapely.polygons(quads))                    # planar UTM cell area
rng = np.random.default_rng(0)
v = rng.random(grids[10]["W"] * grids[10]["H"])
mean, _ = conservative_mean(ci, pi, w, v, N)
lhs = float((mean * A_plane).sum())
rhs = float(np.bincount(pi, w * A_plane[ci] * v[pi]).sum())        # sum_p v_p * area(p ∩ cells)
const, _ = conservative_mean(ci, pi, w, np.full_like(v, 0.37), N)
selftest = {"random_field_integral_rel_err": abs(lhs - rhs) / rhs,
            "constant_field_max_abs_err": float(np.abs(const - 0.37).max()),
            "pixel_area_covered_by_cells_m2": float(np.bincount(pi, w * A_plane[ci]).sum()),
            "sum_cell_area_plane_m2": float(A_plane.sum())}
assert selftest["random_field_integral_rel_err"] < 1e-9 and selftest["constant_field_max_abs_err"] < 1e-9
print("conservation", selftest, flush=True)

# --- bathymetry: conservative area mean on the lon/lat grid
with rasterio.open(f"{SH}/bathy/emodnet_mean.tif") as b:
    bt, bz = b.transform, b.read(1).astype(np.float64)
bq = np.stack([vlon, vlat], -1)
bci, bpi, bw = grid_weights(bq, bt.c, bt.f, bt.a, -bt.e, bz.shape[1], bz.shape[0])
bathy, _ = conservative_mean(bci, bpi, bw, bz.ravel(), N)
depth = np.clip(-bathy, 0, None)

# --- references
site_share, d_site = polygon_share(ids, quads, shapely.union_all(site.geometry.buffer(0).values))
ref = gpd.read_file(f"{SH}/zip/GR000019.shp").to_crs(UTM)
ref["geometry"] = ref.buffer(0)
codes = {"1120": 1, "1110": 2, "1170": 3}
ref["code"] = [(4 if "no Posidonia" in r.habsubtype else 5) if r.hab_type == "Na" else codes[r.hab_type]
               for r in ref.itertuples()]
# 1 Posidonia(1120) 2 Cymodocea/Halophila sand(1110) 3 reef algae(1170) 4 vegetated soft no Posidonia 5 bare soft
shares, diags = {}, {"site": d_site}
for c in range(1, 6):
    shares[c], diags[f"class_{c}"] = polygon_share(ids, quads, shapely.union_all(ref.geometry[ref.code == c].values))
    print("class", c, diags[f"class_{c}"], f"{time.time()-t0:.0f}s", flush=True)
S = np.stack([shares[c] for c in range(1, 6)], 1)
covered = S.sum(1)
S_all = np.c_[np.clip(1 - covered, 0, None), S]                    # column 0 = unmapped
refc = S_all.argmax(1).astype(np.int8)
g15 = gpd.read_file(f"{SH}/greek_seagrass_meadows_v0906.geojson").to_crs(UTM)
sg2015_share, diags["sg2015"] = polygon_share(ids, quads, shapely.union_all(g15.geometry.buffer(0).values))

clon, clat = hn.healpix_to_lonlat(ids, DEPTH, ellipsoid=ELLIPSOID)
cells = pd.DataFrame({"cell": ids, "lon": clon, "lat": clat, "depth": depth.astype(np.float32),
                      "site_share": site_share, "refc": refc, "ref_share_max": S_all.max(1),
                      "ref_covered": covered, "sg2015_share": sg2015_share})
for c in range(1, 6):
    cells[f"share_c{c}"] = shares[c]
cells["pos_share"] = cells["share_c1"]
cells.to_parquet(f"{ROOT}/data/cells.parquet", index=False)

A = cell_area_m2(DEPTH)
# independent check of the cell area: geodesic area of a few densified cells
from pyproj import Geod
gd = Geod(ellps="WGS84")
sl, sa = hn.vertices(ids[:: max(1, N // 50)], DEPTH, ellipsoid=ELLIPSOID, step=8)
geod_areas = np.array([abs(gd.polygon_area_perimeter(lo, la)[0]) for lo, la in zip(sl, sa)])
ha = A / 1e4
out = {"grid": {"library": "healpix-geo", "ellipsoid": ELLIPSOID, "indexing": "nested", "depth": DEPTH,
                "authalic_radius_m": authalic_radius(), "cell_area_m2": A, "cell_edge_m_sqrt_area": float(np.sqrt(A)),
                "cell_area_check_geodesic_m2_mean": float(geod_areas.mean()),
                "cell_area_check_geodesic_rel_spread": float(np.ptp(geod_areas) / A),
                "n_cells": int(N), "window_utm": [X0, Y0, X1, Y1], "native_grids": grids},
       "conservation_selftest": selftest, "reference_diagnostics": diags,
       "reference_bookkeeping_ha": {
           "site_area_share_sum": float(site_share.sum() * ha), "site_cells_gt05": float((site_share > 0.5).sum() * ha),
           "site_polygon_area": float(site.area.sum() / 1e4),
           "pos_area_share_sum_window": float(shares[1].sum() * ha),
           "pos_cells_plurality_window": float((refc == 1).sum() * ha),
           "max_class_share_sum_per_cell": float(covered.max()),
           "n_cells_overlapping_classes_gt1.01": int((covered > 1.01).sum())},
       "seconds": time.time() - t0}
json.dump(out, open(f"{ROOT}/data/grid.json", "w"), indent=1)
print(json.dumps(out, indent=1))
