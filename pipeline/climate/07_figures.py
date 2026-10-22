"""Figure set.  fig1 observed time series + MHW; fig2 exposure map on HEALPix d17 cells; fig3 projection vs observed.
Palette: validated reference palette (dataviz skill, light mode): categorical #2a78d6 blue, #eb6834 orange,
#1baf7a aqua; sequential blue ramp; ink #0b0b0b / #52514e; grid #e1e0d9.
Usage: pixi run python pipeline/climate/07_figures.py <ROOT>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import healpix_geo.nested as hn
import mhw

ROOT = sys.argv[1]
D, F, T = f"{ROOT}/data", f"{ROOT}/figures", f"{ROOT}/tables"
os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, INK, INK2, GRID, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left"})
cells = pd.read_parquet(f"{D}/cell_metrics.parquet")
mc = cells[cells.meadow_ref]

# ---------------- fig1: observed -----------------
fig, ax = plt.subplots(3, 1, figsize=(10, 10.5), gridspec_kw={"height_ratios": [1, 1, 1.1]})
groups = [("SST (satellite, REP)", "annual_rep_key_satellite.csv", BLUE),
          ("Meadow 0–10 m (reanalysis)", "annual_zt_key_0-10m.csv", ORANGE),
          ("Meadow 20–40 m (reanalysis)", "annual_zt_key_20-40m.csv", AQUA)]
for lab, fn, col in groups:
    a = pd.read_csv(f"{T}/{fn}", index_col=0)
    full = a[a.index <= 2025]
    ax[0].plot(full.index, full.mhw_days, color=col, lw=2, label=lab)
    if 2026 in a.index:
        ax[0].plot([2026], [a.loc[2026, "mhw_days"]], "o", mfc="none", mec=col, ms=8)
    ax[1].plot(a.index, a.t_max, color=col, lw=2, label=lab)
ax[0].set_title("a  Marine-heatwave days per year (Hobday 2016; fixed baseline: SST 1983–2012, reanalysis 1987–2016)")
ax[0].set_ylabel("MHW days / year")
ax[0].annotate("open circles: 2026 to 22–30 Sep only", xy=(0.01, 0.9), xycoords="axes fraction", color=INK2, fontsize=9)
ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.85))
for th, ls, txt in ((28, "--", "28 °C  field / synthesis T-limit"), (28.9, ":", "28.9 °C  lab LT50 (Savva 2018)")):
    ax[1].axhline(th, color=INK2, lw=1, ls=ls)
    ax[1].text(1982.3, th + 0.08, txt, color=INK2, fontsize=8.5)
ax[1].set_title("b  Annual maximum of daily mean temperature")
ax[1].set_ylabel("°C")
ax[1].set_ylim(22.5, 30)
# c: 2024 daily, meadow-weighted SST REP + 0-10 m
ds = xr.open_dataset(f"{D}/daily_series_obs.nc")
wr = mc.rep_key.value_counts()
sst = (sum(ds[k] * n for k, n in wr.items()) / wr.sum()).to_series().dropna()
seas, thr = mhw.climatology(sst.index, sst.values, (1983, 2012))
ev, inm, anom = mhw.detect(sst.index, sst.values, seas, thr)
s = sst.loc["2024-05-01":"2024-11-30"]
d = mhw.doy366(s.index)
m = pd.Series(inm, index=sst.index).loc[s.index]
ax[2].fill_between(s.index, seas[d], s.values, where=m.values, color="#f7c6b0", lw=0, label="MHW days (SST)")
ax[2].plot(s.index, seas[d], color=MUTED, lw=1.2, label="Climatology 1983–2012")
ax[2].plot(s.index, thr[d], color=INK2, lw=1, ls="--", label="90th percentile threshold")
ax[2].plot(s.index, s.values, color=BLUE, lw=2, label="Daily SST, meadow mean")
wz = mc[mc.band == "0-10 m"].zt_key.value_counts()
z = (sum(ds[k] * n for k, n in wz.items()) / wz.sum()).to_series().loc[s.index]
ax[2].plot(z.index, z.values, color=ORANGE, lw=1.5, label="Daily T at 0–10 m meadows (reanalysis)")
ax[2].axhline(28, color=INK2, lw=0.8, ls=":")
ax[2].set_title("c  2024, the most extreme year so far: daily temperature and MHW periods")
ax[2].set_ylabel("°C")
ax[2].legend(frameon=False, ncol=2, fontsize=8.5, loc="lower center")
ax[2].set_ylim(14, 30.5)
fig.tight_layout()
fig.savefig(f"{F}/fig1_observed_timeseries.png", dpi=170)
plt.close(fig)

# ---------------- fig2: map on HEALPix cells -----------------
r = pd.read_csv(f"{T}/meadow_cells_ranked.csv")
allc = cells
lonv, latv = hn.vertices(allc.cell.values.astype(np.uint64), np.uint8(17), ellipsoid="WGS84")
polys = {c: np.c_[lo, la] for c, lo, la in zip(allc.cell.values, lonv, latv)}
site = json.load(open(f"{ROOT}/../accuracy/data/site.geojson"))
fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
for a_ in ax:
    nm = allc[~allc.meadow_ref]
    a_.add_collection(PolyCollection([polys[c] for c in nm.cell], facecolor="#ecebe6", edgecolor="none"))
    for f in site["features"]:
        g = f["geometry"]
        rings = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in rings:
            xy = np.array(poly[0]); a_.plot(xy[:, 0], xy[:, 1], color=INK2, lw=0.8)
    a_.set_xlim(20.82, 21.0); a_.set_ylim(37.655, 37.75)
    a_.set_aspect(1 / np.cos(np.deg2rad(37.7)))
    a_.grid(False)
    a_.set_xlabel("Longitude °E"); a_.set_ylabel("Latitude °N")
    a_.set_xticks([20.84, 20.88, 20.92, 20.96, 21.0])
# a: days > 28 C at meadow depth
bins = [0, 0.5, 1, 2, 3, 4, 5, 7]
cmap = ListedColormap(SEQ); norm = BoundaryNorm(bins, cmap.N)
pc = PolyCollection([polys[c] for c in r.cell], array=r.zt_days_gt_28_2016_2025.values, cmap=cmap, norm=norm, edgecolor="none")
ax[0].add_collection(pc)
cb = fig.colorbar(pc, ax=ax[0], shrink=0.75, pad=0.02)
cb.set_label("days/yr with T > 28 °C at meadow depth\n(mean 2016–2025)")
ax[0].set_title("a  Observed heat exposure at the seabed, per HEALPix cell")
# b: tiers
tc = {"A": "#184f95", "B": "#6da7ec", "C": "#cde2fb"}
ax[1].add_collection(PolyCollection([polys[c] for c in r.cell], facecolor=[tc[t] for t in r.tier], edgecolor="none"))
ax[1].legend(handles=[Patch(color=tc["A"], label="A  ≥1 day/yr above 28 °C"), Patch(color=tc["B"], label="B  summer max ≥27 °C"),
                      Patch(color=tc["C"], label="C  lower exposure"), Patch(color="#ecebe6", label="no Posidonia (HCMR map)")],
             frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=8.5)
ax[1].set_title("b  Exposure tier (ranking of meadow cells)")
fig.text(0.01, 0.01, "HEALPix depth 17 (~50 m), healpix-geo WGS84 NESTED. Meadow = HCMR map share ≥ 0.5. Outline: Natura 2000 GR2210002. "
         "\nValues from the 1/24° reanalysis column containing each cell (nearest sea column where the model has land).",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f"{F}/fig2_exposure_map_healpix.png", dpi=170)
plt.close(fig)

# ---------------- fig3: projection vs observed -----------------
p = f"{T}/annual_dt_and_obs_sst.csv"
if os.path.exists(p):
    a = pd.read_csv(p)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    obs = a[(a.model == "OBS_SST_REP") & (a.year <= 2025)]
    series = [("Observed SST (REP)", obs, INK, 2.2)]
    for mdl, col in (("IFS-NEMO", BLUE), ("IFS-FESOM", ORANGE)):
        b = a[(a.model == mdl) & (a["corr"] == "bc")]
        if len(b):
            series.append((f"Climate DT {mdl}, bias-corrected (hist → SSP3-7.0)", b, col, 1.6))
    for lab, b, col, lw in series:
        ax[0].plot(b.year, b.mhw_days, color=col, lw=lw, label=lab)
        ax[1].plot(b.year, b.t_max, color=col, lw=lw, label=lab)
    for th, ls, txt in ((28, "--", "28 °C"), (28.9, ":", "28.9 °C"), (30, "-.", "30 °C  Crete upper growth limit (Bennett 2022)")):
        ax[1].axhline(th, color=INK2, lw=1, ls=ls); ax[1].text(1983, th + 0.08, txt, color=INK2, fontsize=8.5)
    for a_ in ax:
        a_.axvspan(2025.5, 2049.5, color="#f3f2ee", zorder=0)
    ax[0].text(2026, ax[0].get_ylim()[1] * 0.93 if ax[0].get_ylim()[1] > 0 else 1, "projection", color=INK2, fontsize=9)
    ax[0].set_title("a  Sea-surface MHW days per year at the meadows (fixed observed baseline 1983–2012)")
    ax[0].set_ylabel("MHW days / year"); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[1].set_title("b  Annual maximum daily SST at the meadows")
    ax[1].set_ylabel("°C"); ax[1].set_xlim(1982, 2050)
    fig.text(0.01, 0.005, "Climate DT years are not in phase with real years: compare statistics, not individual years. One realisation per model. "
             "Surface only: DT ocean level depths undocumented.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(f"{F}/fig3_projection_vs_observed.png", dpi=170)
    plt.close(fig)
print("figures written")
