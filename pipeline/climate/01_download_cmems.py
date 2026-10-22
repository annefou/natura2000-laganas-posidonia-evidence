"""Download small Copernicus Marine subsets around GR2210002 (Laganas Bay, Zakynthos).

Usage: pixi run python pipeline/climate/01_download_cmems.py <ROOT>
Credentials are read by the toolbox from ~/.copernicusmarine (never printed here).
Box: 20.70-21.10 E, 37.55-37.80 N (site bbox 20.825-20.991 E, 37.663-37.743 N, plus one-two grid cells margin).
"""
import sys, os, json
import copernicusmarine as cm

ROOT = sys.argv[1]
OUT = f"{ROOT}/data/cmems"
os.makedirs(OUT, exist_ok=True)
BOX = dict(minimum_longitude=20.70, maximum_longitude=21.10, minimum_latitude=37.55, maximum_latitude=37.80)
jobs = [
    # Observed SST, L4 reprocessed (ESA CCI/C3S-based, 0.05 deg), 1982 onward
    dict(dataset_id="cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021", variables=["analysed_sst", "analysis_error"],
         start_datetime="1982-01-01", end_datetime="2026-12-31", output_filename="sst_rep.nc"),
    # Observed SST, L4 NRT high resolution (0.01 deg), 2008 onward: used only for months after REP ends + cross-check
    dict(dataset_id="SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2", variables=["analysed_sst"],
         start_datetime="2008-01-01", end_datetime="2026-12-31", output_filename="sst_nrt.nc"),
    # Physics reanalysis (Med MFC, 4.2 km, 141 levels), daily bottom temperature + temperature 0-60 m
    dict(dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m", variables=["bottomT", "thetao"],
         minimum_depth=0, maximum_depth=60,
         start_datetime="1987-01-01", end_datetime="2026-12-31", output_filename="phy_my_temp.nc"),
    # Static: model bathymetry and mask
    dict(dataset_id="cmems_mod_med_phy_my_4.2km_static", variables=["deptho", "mask"],
         output_filename="phy_my_static.nc"),
]
log = []
for j in jobs:
    fn = f"{OUT}/{j['output_filename']}"
    if os.path.exists(fn):
        print("exists", fn); continue
    kw = {**BOX, **j}
    r = cm.subset(**kw, output_directory=OUT, overwrite=True, disable_progress_bar=True)
    log.append({k: v for k, v in kw.items()})
    print("ok", fn)
json.dump(log, open(f"{OUT}/requests.json", "a"), indent=1, default=str)
