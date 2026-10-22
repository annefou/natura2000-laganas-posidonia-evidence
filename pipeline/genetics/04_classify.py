"""Exposure x genetic-diversity screening of the 18 genotyped Greek Posidonia oceanica meadows: table, map, sensitivity.

Usage: pixi run python pipeline/genetics/04_classify.py <ROOT>
Inputs: data/genetics_by_meadow.csv (01), data/exposure_by_meadow.csv (03), data/inputs/ne_10m_land.zip.

Classification rules (fixed before looking at results; relative to these 18 meadows only, not absolute thresholds):
  Exposure score E   = mean of the percentile ranks (0-1, ties averaged) of
                       (a) mean MHW days/yr 2016-2025 at the sampling depth (reanalysis, Hobday 2016, 1987-2016 baseline)
                       (b) mean days/yr with T > 28 C at the sampling depth, 2016-2025 (reanalysis)
  Diversity score G  = mean of the percentile ranks of clonal richness R and gene diversity Hs (10 SSR loci)
  High exposure      = E in the top third of the 18 meadows (6 highest)
  Low diversity      = G in the bottom third (6 lowest)
  Class: "1 high exposure + low diversity"; "2 high exposure"; "3 low diversity"; "4 neither".
This is a screening of co-occurrence. It makes no claim that low diversity causes heat damage or that these meadows
are, or will be, harmed: no condition, mortality or flowering data enter the classification.
Sensitivity (reported, not used for the class): R replaced by R_ident (identical MLGs collapsed); exposure from
satellite SST instead of temperature at depth; exposure from MHW days only; from days > 28 C only.
Outputs: tables/meadow_vulnerability.csv, figures/fig_genetic_vulnerability_map.png, results.json
"""
import sys, os, json, zipfile
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = sys.argv[1]
D, F, T = f"{ROOT}/data", f"{ROOT}/figures", f"{ROOT}/tables"
os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
gen = pd.read_csv(f"{D}/genetics_by_meadow.csv")
ex = pd.read_csv(f"{D}/exposure_by_meadow.csv")
m = gen.merge(ex.drop(columns=["sea", "region", "site", "lat", "lon", "depth_m"]), on="site_code")
assert len(m) == 18
K = len(m) // 3
LAB = {1: "1  high exposure + low diversity", 2: "2  high exposure", 3: "3  low diversity", 4: "4  neither"}


def classify(df, ex_cols, div_cols):
    E = df[ex_cols].rank(pct=True).mean(axis=1)
    G = df[div_cols].rank(pct=True).mean(axis=1)
    hiE = E.rank(method="first", ascending=False) <= K
    loG = G.rank(method="first", ascending=True) <= K
    cls = np.where(hiE & loG, 1, np.where(hiE, 2, np.where(loG, 3, 4)))
    return E, G, hiE, loG, cls


EX = ["zt_mhw_days_2016_2025", "zt_days_gt28_2016_2025"]
DIV = ["R", "Hs"]
m["E"], m["G"], m["high_exposure"], m["low_diversity"], m["class"] = classify(m, EX, DIV)
sens = {"R_ident instead of R": (EX, ["R_ident", "Hs"]),
        "SST (satellite) exposure": (["sst_mhw_days_2016_2025", "sst_days_gt28_2016_2025"], DIV),
        "MHW days only": (["zt_mhw_days_2016_2025"], DIV),
        "days > 28 C only": (["zt_days_gt28_2016_2025"], DIV)}
for name, (e, d) in sens.items():
    m[f"class_{name}"] = classify(m, e, d)[4]
m["class_stable_n_of_4"] = sum((m[f"class_{n}"] == 1) == (m["class"] == 1) for n in sens)

cols = {"site_code": "code", "site": "meadow", "region": "region", "lat": "lat", "lon": "lon", "depth_m": "depth_m",
        "hpx12": "healpix_d12_nested_wgs84", "N": "N_ramets", "G": "MLG", "R": "R", "R_ident": "R_ident", "Ar": "Ar",
        "Hs": "Hs", "Ho": "Ho", "phy_level_m": "model_level_m",
        "zt_mhw_days_2016_2025": "mhw_days_yr_depth_2016_2025", "zt_mhw_days_1987_2015": "mhw_days_yr_depth_1987_2015",
        "zt_mhw_days_trend_sen": "mhw_days_trend_depth_per_dec", "zt_mhw_days_trend_sen_lo": "trend_lo",
        "zt_mhw_days_trend_sen_hi": "trend_hi", "zt_days_gt28_2016_2025": "days_gt28_yr_depth_2016_2025",
        "zt_tmax_2024": "tmax_depth_2024", "zt_tmax_record_year": "tmax_depth_record_year",
        "zt_mhw_days_2024": "mhw_days_depth_2024", "zt_2024_mhw_days_rank": "mhw_days_2024_rank_of_39",
        "sst_mhw_days_2016_2025": "mhw_days_yr_sst_2016_2025", "sst_mhw_days_trend_sen": "mhw_days_trend_sst_per_dec",
        "sst_days_gt28_2016_2025": "days_gt28_yr_sst_2016_2025", "sst_tmax_2024": "tmax_sst_2024",
        "sst_tmax_record_year": "tmax_sst_record_year",
        "E": "exposure_score", "G": "diversity_score", "class": "class", "class_stable_n_of_4": "class1_membership_stable_n_of_4"}
tab = m[list(cols)].rename(columns=cols).sort_values(["class", "exposure_score"], ascending=[True, False])
tab["class_label"] = tab["class"].map(LAB)
tab.round(3).to_csv(f"{T}/meadow_vulnerability.csv", index=False)

# ---------------- figure: map + rule plane -----------------
INK, INK2, GRID, MUTED, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#898781", "#fcfcfb"
COL = {1: "#eb6834", 2: "#eda100", 3: "#2a78d6", 4: "#b5b4ad"}
MRK = {1: "D", 2: "^", 3: "s", 4: "o"}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlesize": 10.5,
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})
zf = f"{D}/inputs/ne_10m_land.zip"
land = gpd.read_file(f"zip://{zf}").cx[19.5:28.6, 34.3:41.2]
fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
a.set_facecolor("#eef4fb")
land.plot(ax=a, color="#e7e6df", edgecolor="#c3c2b7", linewidth=0.4)
for c in (4, 3, 2, 1):
    s = m[m["class"] == c]
    a.scatter(s.lon, s.lat, s=70, marker=MRK[c], c=COL[c], edgecolor=SURF, linewidth=1.5, zorder=3, label=LAB[c])
for _, r in m.iterrows():
    off = {"SAN": (-24, -11), "PER": (6, -2)}.get(r.site_code, (5, 4))
    a.annotate(r.site_code, (r.lon, r.lat), xytext=off, textcoords="offset points", fontsize=8, color=INK, zorder=4)
a.set_xlim(19.6, 28.4); a.set_ylim(34.5, 40.8); a.set_aspect(1 / np.cos(np.deg2rad(37.5)))
a.set_xlabel("longitude (°E)"); a.set_ylabel("latitude (°N)")
a.set_title("Genotyped P. oceanica meadows (Litsi-Mizan et al. 2024) by screening class")
a.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=8.5)
b = ax[1]
Es, Gs = np.sort(m.E.values)[::-1], np.sort(m.G.values)
e_cut, g_cut = (Es[K - 1] + Es[K]) / 2, (Gs[K - 1] + Gs[K]) / 2          # midway between 6th and 7th meadow
b.axvspan(e_cut, 1.02, color="#fbe3d6", zorder=0, lw=0)
b.axhspan(-0.02, g_cut, color="#dbe8f8", zorder=0, lw=0)
for _, r in m.iterrows():
    b.scatter(r.E, r.G, s=70, marker=MRK[r["class"]], c=COL[r["class"]], edgecolor=SURF, linewidth=1.5, zorder=3)
    b.annotate(r.site_code, (r.E, r.G), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
b.grid(color=GRID, lw=0.6); b.spines[["top", "right"]].set_visible(False)
b.set_xlabel("exposure score E\n(mean rank of MHW days/yr and days >28 °C/yr at sampling depth, 2016–2025)")
b.set_ylabel("diversity score G (rank of clonal richness R and Hs)")
b.set_title("Rule: 6 highest E (orange band) × 6 lowest G (blue band)")
fig.text(0.01, 0.01, "Exposure: Copernicus Marine Med reanalysis at sampling depth, MHW after Hobday et al. 2016. Genetics: 10 SSR loci, "
         "sampled 2014–2021. Ranks are relative to these 18 meadows; co-occurrence screening, not a causal or impact claim.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{F}/fig_genetic_vulnerability_map.png", dpi=170)

# ---------------- results -----------------
cor = {f"spearman_{x}_vs_{y}": float(m[x].corr(m[y], method="spearman")) for x in ("R", "Hs") for y in EX}
res = {"n_meadows": 18, "rules": {"high_exposure": "top 6 of 18 by E", "low_diversity": "bottom 6 of 18 by G"},
       "class_counts": {LAB[k]: int(v) for k, v in m["class"].value_counts().sort_index().items()},
       "class1": m.loc[m["class"] == 1, "site_code"].tolist(),
       "sensitivity_class1": {n: m.loc[m[f"class_{n}"] == 1, "site_code"].tolist() for n in sens},
       "correlations_genetics_vs_exposure": cor,
       "range": {c: [float(m[c].min()), float(m[c].max())] for c in EX + ["sst_mhw_days_2016_2025", "R", "R_ident", "Hs",
                                                                       "zt_mhw_days_trend_sen", "sst_mhw_days_trend_sen"]},
       "median": {c: float(m[c].median()) for c in EX + ["zt_mhw_days_1987_2015", "sst_mhw_days_2016_2025",
                                                       "sst_mhw_days_1982_2015", "zt_mhw_days_trend_sen", "sst_mhw_days_trend_sen"]},
       "n_meadows_tmax_record_2024": {"depth": int((m.zt_tmax_record_year == 2024).sum()), "sst": int((m.sst_tmax_record_year == 2024).sum())},
       "n_meadows_mhw_days_record_2024": {"depth": int((m.zt_mhw_days_record_year == 2024).sum()), "sst": int((m.sst_mhw_days_record_year == 2024).sum())},
       "trend_ci_excludes_zero": {"depth": int((m.zt_mhw_days_trend_sen_lo > 0).sum()), "sst": int((m.sst_mhw_days_trend_sen_lo > 0).sum())},
       "assignment": {"sst_fallback": int(m.sst_fallback.sum()), "sst_fallback_km_max": float(m.sst_dist_km[m.sst_fallback].max()) if m.sst_fallback.any() else 0,
                      "phy_fallback": int(m.phy_fallback.sum()), "phy_fallback_km_max": float(m.phy_dist_km[m.phy_fallback].max()) if m.phy_fallback.any() else 0}}
json.dump(res, open(f"{ROOT}/results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
pd.set_option("display.width", 250)
print(tab[["code", "region", "depth_m", "R", "R_ident", "Hs", "mhw_days_yr_depth_2016_2025", "days_gt28_yr_depth_2016_2025",
           "mhw_days_trend_depth_per_dec", "tmax_depth_2024", "tmax_depth_record_year", "mhw_days_yr_sst_2016_2025",
           "exposure_score", "diversity_score", "class", "class1_membership_stable_n_of_4"]].round(2).to_string(index=False))
