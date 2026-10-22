"""Step 2 (HEALPix-native): every Sentinel-2 L2A scene straight onto depth-19 cells, then a per-cell median.

Same pinned scene IDs as pipeline/accuracy/01_composite.py (10 scenes 2025, 9 scenes 2024), same bands
(coastal B1 60 m; blue B2, green B3, red B4, nir B8 10 m; swir16 B11 20 m), same SCL classes kept (4, 5, 6),
same DN -> reflectance (DN * 1e-4 - 0.1). There is no UTM working grid: each band is read at its native
resolution over the window and aggregated onto cells with exact area-overlap weights (00_grid.py):

  * reflectance: conservative area-weighted mean of the valid native pixels in the cell;
    a native pixel is valid if its DN > 0 and at least half its area has SCL in {4, 5, 6};
  * SCL: area-weighted majority per cell (categorical, never averaged);
  * a cell is a valid observation if its SCL majority is in {4, 5, 6} and every band has valid native
    pixels over at least half the cell area.
Composite: per-cell median of valid observations; n_valid_obs counted on the blue band (as in 01_composite).
Effective resolution: ~12.4 m (cell, sqrt of 154.6 m2) for the 10 m bands; 20 m for B11 and 60 m for B1,
whose native pixels are larger than the cell (a cell is the area-weighted mean of the 1-4 pixels it overlaps).
Writes <ROOT>/data/scenes/<id>.parquet (cache) and <ROOT>/data/composite_<tag>.parquet.
Usage: pixi run python pipeline/accuracy_healpix/01_scenes.py <ROOT = work/accuracy_healpix> <tag 2025|2024>
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from pystac_client import Client
from common import conservative_mean, area_majority

IDS = {"2025": ["S2B_34SDG_20250611_0_L2A", "S2C_34SDG_20250626_0_L2A", "S2C_34SDG_20250706_0_L2A",
                "S2C_34SDG_20250716_0_L2A", "S2C_34SDG_20250726_0_L2A", "S2A_34SDG_20250807_0_L2A",
                "S2B_34SDG_20250810_0_L2A", "S2C_34SDG_20250828_0_L2A", "S2B_34SDG_20250902_0_L2A",
                "S2B_34SDG_20250912_0_L2A"],
       "2024": ["S2A_34SDG_20240601_0_L2A", "S2A_34SDG_20240621_0_L2A", "S2A_34SDG_20240701_0_L2A",
                "S2A_34SDG_20240711_0_L2A", "S2A_34SDG_20240721_0_L2A", "S2B_34SDG_20240716_0_L2A",
                "S2B_34SDG_20240805_0_L2A", "S2B_34SDG_20240815_0_L2A", "S2B_34SDG_20240825_0_L2A"]}
BANDS = {"coastal": 60, "blue": 10, "green": 10, "red": 10, "nir": 10, "swir16": 20}
ROOT, TAG = sys.argv[1], sys.argv[2]
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
G = json.load(open(f"{ROOT}/data/grid.json"))["grid"]
X0, Y0, X1, Y1 = G["window_utm"]
NG = {int(k): v for k, v in G["native_grids"].items()}
N = G["n_cells"]
WT = {r: tuple(pd.read_parquet(f"{ROOT}/data/weights_{r}.parquet")[c].to_numpy() for c in ("ci", "pi", "w"))
      for r in (10, 20, 60)}
cells = pd.read_parquet(f"{ROOT}/data/cells.parquet", columns=["cell"])["cell"].to_numpy()


def read(href, res):
    with rasterio.open(href) as src:
        assert src.crs.to_epsg() == 32634 and abs(src.res[0] - res) < 1e-6, (href, src.crs, src.res)
        assert (X0 - src.transform.c) % res == 0 and (src.transform.f - Y1) % res == 0, "grid not aligned"
        win = from_bounds(X0, Y0, X1, Y1, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win)
    assert a.shape == (NG[res]["H"], NG[res]["W"]), (a.shape, NG[res])
    return a


def scl_valid_fraction(scl20, res):
    """Share of each native pixel's area with SCL in {4,5,6} (SCL is 20 m; the 10/20/60 m grids nest)."""
    good = np.isin(scl20, [4, 5, 6]).astype(np.float32)
    if res == 10:
        return np.repeat(np.repeat(good, 2, 0), 2, 1)
    if res == 20:
        return good
    H, W = NG[60]["H"], NG[60]["W"]
    return good.reshape(H, 3, W, 3).mean((1, 3))


def scene(it):
    f = f"{ROOT}/data/scenes/{it.id}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f), json.load(open(f + ".json"))
    with ThreadPoolExecutor(7) as ex:
        fut = {b: ex.submit(read, it.assets[b].href, r) for b, r in BANDS.items()}
        fs = ex.submit(read, it.assets["scl"].href, 20)
        arr = {b: fut[b].result() for b in BANDS}
        scl = fs.result()
    ci, pi, w = WT[20]
    scl_cell, scl_share = area_majority(ci, pi, w, scl.ravel(), N, list(range(12)))
    ok = np.isin(scl_cell, [4, 5, 6])
    df = pd.DataFrame({"cell": cells, "scl": scl_cell.astype(np.int8), "scl_share": scl_share.astype(np.float32)})
    for b, r in BANDS.items():
        dn = arr[b].ravel()
        valid = (dn > 0) & (scl_valid_fraction(scl, r).ravel() >= 0.5)
        m, frac = conservative_mean(*WT[r], dn * 1e-4 - 0.1, N, valid)
        ok &= frac >= 0.5
        df[b] = m.astype(np.float32)
    for b in BANDS:
        df.loc[~ok, b] = np.nan
    log = {"id": it.id, "datetime": str(it.datetime), "cloud": it.properties.get("eo:cloud_cover"),
           "valid_frac_cells": float(ok.mean()),
           "valid_frac_native_scl20": float(np.isin(scl, [4, 5, 6]).mean())}
    df.to_parquet(f, index=False)
    json.dump(log, open(f + ".json", "w"))
    return df, log


def main():
    os.makedirs(f"{ROOT}/data/scenes", exist_ok=True)
    t0 = time.time()
    items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
        collections=["sentinel-2-l2a"], ids=IDS[TAG]).items())
    assert len(items) == len(IDS[TAG]), [i.id for i in items]
    stack = np.full((len(items), len(BANDS), N), np.nan, np.float32)
    logs = []
    for k, it in enumerate(sorted(items, key=lambda i: i.id)):
        df, log = scene(it)
        assert (df.cell.to_numpy() == cells).all()
        stack[k] = df[list(BANDS)].to_numpy().T
        logs.append(log)
        print(log, f"{time.time()-t0:.0f}s", flush=True)
    med = np.nanmedian(stack, axis=0)
    out = pd.DataFrame({"cell": cells})
    for j, b in enumerate(BANDS):
        out[b] = med[j]
    out["n_valid_obs"] = np.isfinite(stack[:, 1]).sum(0).astype(np.int16)
    out.to_parquet(f"{ROOT}/data/composite_{TAG}.parquet", index=False)
    json.dump(logs, open(f"{ROOT}/data/s2_scenes_{TAG}.json", "w"), indent=1)
    print("done", TAG, out.n_valid_obs.describe().to_dict())


if __name__ == "__main__":
    import warnings; warnings.simplefilter("ignore", RuntimeWarning)
    main()
