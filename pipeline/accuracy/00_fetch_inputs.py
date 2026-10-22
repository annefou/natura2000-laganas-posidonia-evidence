"""Open inputs for the Sentinel-2 Posidonia accuracy test at GR2210002 Laganas Bay.

Usage: python pipeline/accuracy/00_fetch_inputs.py <ROOT>     (ROOT = work/accuracy)
Writes <ROOT>/data/:
  site.geojson                           Natura 2000 GR2210002 boundary, EEA Natura2000Sites MapServer layer 2
  zip/GR000019.*                         R1: HCMR habitat map of the site (1999-2002), EMODnet Seabed Habitats
                                         survey map GR000019, CC BY 4.0
  greek_seagrass_meadows_v0906.geojson   R2: Panayotidis et al. 2022, doi:10.17882/87740 (SEANOE), CC BY 4.0,
                                         clipped to the study box
  bathy/emodnet_mean.tif                 EMODnet Bathymetry DTM (mean depth), WCS, CC BY 4.0
Sentinel-2 scenes are fetched by 01_composite.py.
"""
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

import geopandas as gpd
from shapely.geometry import box

ROOT = sys.argv[1]
D = f"{ROOT}/data"
BOX = (20.75, 37.60, 21.05, 37.80)   # lon/lat study box around GR2210002
SITE = ("https://bio.discomap.eea.europa.eu/arcgis/rest/services/ProtectedSites/Natura2000Sites/MapServer/2/"
        "query?where=SITECODE%3D%27GR2210002%27&outFields=*&returnGeometry=true&outSR=4326&f=geojson")
R1 = "https://files.emodnet-seabedhabitats.eu/data/EMODnetSBHsurvey_GR000019.zip"
R2 = "https://www.seanoe.org/data/00765/87740/data/93289.zip"
R2_SHP = "Greek_Seagrass_Meadows_v0906/Greek Seagrass meadows v0906.shp"
BATHY = ("https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1&request=GetCoverage"
         "&coverageId=emodnet__mean&format=image/tiff&subset=Lat(37.60,37.80)&subset=Long(20.75,21.05)")


def get(url: str, tries: int = 5) -> bytes:
    """Download with retries; EMODnet services return 502/504 at times."""
    req = urllib.request.Request(url, headers={"User-Agent": "natura2000-laganas-posidonia-evidence"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1:
                raise SystemExit(f"download failed after {tries} tries: {url} ({e})")
            time.sleep(30 * (i + 1))


def main() -> None:
    os.makedirs(f"{D}/zip", exist_ok=True)
    os.makedirs(f"{D}/bathy", exist_ok=True)
    if not os.path.exists(f"{D}/site.geojson"):
        b = get(SITE)
        open(f"{D}/site.geojson", "wb").write(b)
    if not os.path.exists(f"{D}/zip/GR000019.shp"):
        zipfile.ZipFile(io.BytesIO(get(R1))).extractall(f"{D}/zip")
    if not os.path.exists(f"{D}/greek_seagrass_meadows_v0906.geojson"):
        z = f"{D}/r2_seanoe.zip"
        b = get(R2)
        open(z, "wb").write(b)
        g = gpd.read_file(f"zip://{os.path.abspath(z)}!{R2_SHP}").to_crs(4326)
        g = g[g.intersects(box(*BOX))]
        g.to_file(f"{D}/greek_seagrass_meadows_v0906.geojson", driver="GeoJSON")
        os.remove(z)
    if not os.path.exists(f"{D}/bathy/emodnet_mean.tif"):
        b = get(BATHY)
        open(f"{D}/bathy/emodnet_mean.tif", "wb").write(b)
    print("inputs in", D)


if __name__ == "__main__":
    main()
