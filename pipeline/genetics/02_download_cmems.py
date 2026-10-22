"""Download small Copernicus Marine subsets around each genotyped meadow (Litsi-Mizan et al. 2024, 18 Greek sites).

Usage: pixi run python pipeline/genetics/02_download_cmems.py <ROOT> [SITE_CODE ...]
Credentials are read by the toolbox from ~/.copernicusmarine (never printed here). Free account:
https://data.marine.copernicus.eu/register . Licence: Copernicus Marine Service licence
(https://marine.copernicus.eu/user-corner/service-commitments-and-licence), free use with attribution.

Products (same as pipeline/climate/01_download_cmems.py):
- SST_MED_SST_L4_REP_OBSERVATIONS_010_021, dataset cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021 (0.05 deg, 1982-),
  doi:10.48670/moi-00173
- MEDSEA_MULTIYEAR_PHY_006_004, dataset cmems_mod_med_phy-temp_my_4.2km_P1D-m (1/24 deg, 141 levels, 1987-),
  doi:10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1 ; thetao 0-25 m (meadows sampled at 4-15 m) and bottomT
- cmems_mod_med_phy_my_4.2km_static (model depth, mask)
Box per site: +/-0.15 deg around the published coordinates (given to 0.01 deg), enough for a nearest-sea-cell fallback.
"""
import sys, os, json
import pandas as pd
import copernicusmarine as cm

ROOT = sys.argv[1]
OUT = f"{ROOT}/data/cmems"
os.makedirs(OUT, exist_ok=True)
sites = pd.read_csv(f"{ROOT}/data/inputs/sites.csv")
if len(sys.argv) > 2:
    sites = sites[sites.site_code.isin(sys.argv[2:])]
H = 0.15
for _, s in sites.iterrows():
    box = dict(minimum_longitude=s.lon - H, maximum_longitude=s.lon + H, minimum_latitude=s.lat - H, maximum_latitude=s.lat + H)
    jobs = [
        dict(dataset_id="cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021", variables=["analysed_sst"],
             start_datetime="1982-01-01", end_datetime="2026-12-31", output_filename=f"{s.site_code}_sst_rep.nc"),
        dict(dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m", variables=["bottomT", "thetao"], minimum_depth=0, maximum_depth=25,
             start_datetime="1987-01-01", end_datetime="2026-12-31", output_filename=f"{s.site_code}_phy_my_temp.nc"),
        dict(dataset_id="cmems_mod_med_phy_my_4.2km_static", variables=["deptho", "mask"], output_filename=f"{s.site_code}_phy_static.nc"),
    ]
    for j in jobs:
        fn = f"{OUT}/{j['output_filename']}"
        if os.path.exists(fn):
            print("exists", fn); continue
        kw = {**box, **j}
        cm.subset(**kw, output_directory=OUT, overwrite=True, disable_progress_bar=True)
        with open(f"{OUT}/requests.jsonl", "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")
        print("ok", fn, flush=True)
