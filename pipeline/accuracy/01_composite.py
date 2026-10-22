"""Step 2: median Sentinel-2 L2A composite over GR2210002 (Laganas Bay) from Earth Search COGs (no account).

Reads a window (site bbox + 500 m buffer, EPSG:32634, 10 m grid) from each scene, applies SCL mask,
resamples all bands to 10 m (nearest for 20/60 m bands), converts DN -> reflectance with the STAC
offset/scale (-0.1, 1e-4), and takes the per-pixel median. Output: data/s2_median_{TAG}.tif
Usage: pixi run python pipeline/accuracy/01_composite.py <ROOT = work/accuracy> [tag ids_comma_separated]
"""
import sys, json, os
import numpy as np, rasterio, geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pystac_client import Client

ROOT = sys.argv[1]
IDS = ["S2B_34SDG_20250611_0_L2A", "S2C_34SDG_20250626_0_L2A", "S2C_34SDG_20250706_0_L2A",
       "S2C_34SDG_20250716_0_L2A", "S2C_34SDG_20250726_0_L2A", "S2A_34SDG_20250807_0_L2A",
       "S2B_34SDG_20250810_0_L2A", "S2C_34SDG_20250828_0_L2A", "S2B_34SDG_20250902_0_L2A",
       "S2B_34SDG_20250912_0_L2A"]
TAG = "2025"
if len(sys.argv) > 2:  # optional: <tag> <id1,id2,...>  (used for the 2024 stability composite)
    TAG, IDS = sys.argv[2], sys.argv[3].split(",")
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16"]

site = gpd.read_file(f"{ROOT}/data/site.geojson").to_crs(32634)
x0, y0, x1, y1 = site.total_bounds
x0, y0 = np.floor((x0 - 500) / 10) * 10, np.floor((y0 - 500) / 10) * 10
x1, y1 = np.ceil((x1 + 500) / 10) * 10, np.ceil((y1 + 500) / 10) * 10
W, H = int((x1 - x0) / 10), int((y1 - y0) / 10)
T = from_origin(x0, y1, 10, 10)

items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
    collections=["sentinel-2-l2a"], ids=IDS).items())
assert len(items) == len(IDS), [i.id for i in items]
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

def read(href, resamp):
    out = np.zeros((H, W), np.float32)
    with rasterio.open(href) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=T, dst_crs="EPSG:32634", resampling=resamp)
    return out

stack = np.full((len(items), len(BANDS), H, W), np.nan, np.float32)
log = []
for k, it in enumerate(items):
    scl = read(it.assets["scl"].href, Resampling.nearest)
    # keep vegetation(4), bare(5), water(6); drop cloud/shadow/cirrus/saturated/nodata
    good = np.isin(scl, [4, 5, 6])
    for b, name in enumerate(BANDS):
        dn = read(it.assets[name].href, Resampling.nearest)
        refl = dn * 1e-4 - 0.1
        refl[(~good) | (dn == 0)] = np.nan
        stack[k, b] = refl
    log.append({"id": it.id, "datetime": str(it.datetime), "cloud": it.properties.get("eo:cloud_cover"),
                "valid_frac": float(good.mean())})
    print(log[-1], flush=True)

med = np.nanmedian(stack, axis=0)
nvalid = np.sum(~np.isnan(stack[:, 1]), axis=0).astype(np.float32)
prof = dict(driver="GTiff", width=W, height=H, count=len(BANDS) + 1, dtype="float32", crs="EPSG:32634",
            transform=T, compress="deflate", nodata=np.nan)
with rasterio.open(f"{ROOT}/data/s2_median_{TAG}.tif", "w", **prof) as dst:
    for b in range(len(BANDS)):
        dst.write(med[b], b + 1); dst.set_band_description(b + 1, BANDS[b])
    dst.write(nvalid, len(BANDS) + 1); dst.set_band_description(len(BANDS) + 1, "n_valid_obs")
json.dump(log, open(f"{ROOT}/data/s2_scenes_{TAG}.json", "w"), indent=1)
print("done", med.shape)
