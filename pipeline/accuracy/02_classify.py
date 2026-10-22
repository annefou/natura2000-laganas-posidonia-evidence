"""Step 3: Posidonia (1120) vs other-seabed classification at GR2210002 with spatial-block hold-out.

Inputs (all open): data/s2_median_2025.tif (01_composite.py), data/zip/GR000019.shp (HCMR 1999-2002
Natura habitat map, EMODnet SBH, CC BY 4.0), data/greek_seagrass_meadows_v0906.geojson (Panayotidis et
al. 2022 SEANOE doi:10.17882/87740, 2015 acoustic/camera survey, CC BY 4.0), data/bathy/emodnet_mean.tif
(EMODnet Bathymetry DTM, WCS), data/site.geojson (EEA Natura 2000 boundary).

Pre-processing in the spirit of Traganos et al. 2018 (Remote Sens. 10:1227):
  * sun-glint correction (Hedley et al. 2005) of B1/B2/B3 against NIR (B8), regression over optically
    deep water (EMODnet depth > 60 m);
  * deep-water correction X_i = ln(R_i - R_deep_i);
  * Lyzenga depth-invariant indices DII_ij = X_i - (k_i/k_j) X_j for (B1,B2),(B1,B3),(B2,B3), with k_i/k_j
    estimated from reference "non-vegetated soft bottom" pixels in TRAINING blocks only;
  * Random Forest (300 trees, balanced class weights).
Feature set A ("spectral"): X1,X2,X3, DII12, DII13, DII23, B2/B3 ratio.  Feature set B: A + EMODnet depth.
Validation: 1 km x 1 km blocks, 5-fold GroupKFold; every pixel is predicted by a model that never saw
its block (out-of-fold).  Test metrics on all held-out pixels (no sub-sampling of test).
Usage: pixi run python pipeline/accuracy/02_classify.py <ROOT = work/accuracy>
"""
import sys, json
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, cohen_kappa_score

ROOT = sys.argv[1]
rng = np.random.default_rng(42)
src = rasterio.open(f"{ROOT}/data/s2_median_2025.tif")
S = src.read().astype(np.float32)  # coastal, blue, green, red, nir, swir16, nvalid
T, H, W = src.transform, src.height, src.width
B1, B2, B3, B4, B8 = S[0], S[1], S[2], S[3], S[4]
nvalid = S[6]

def rast(gdf, vals, fill=0):
    shapes = [(g, v) for g, v in zip(gdf.geometry, vals) if g is not None and not g.is_empty]
    return rasterize(shapes, out_shape=(H, W), transform=T, fill=fill, dtype="int16")

site = gpd.read_file(f"{ROOT}/data/site.geojson").to_crs(32634)
insite = rast(site, [1] * len(site)).astype(bool)
ref = gpd.read_file(f"{ROOT}/data/zip/GR000019.shp").to_crs(32634)
ref["geometry"] = ref.buffer(0)
codes = {"1120": 1, "1110": 2, "1170": 3}
def refcode(r):
    if r.hab_type == "Na":
        return 4 if "no Posidonia" in r.habsubtype else 5
    return codes[r.hab_type]
# 1 Posidonia(1120) 2 Cymodocea/Halophila sand(1110) 3 reef algae(1170) 4 vegetated soft no Posidonia 5 bare soft
refc = rast(ref, [refcode(r) for r in ref.itertuples()])
g15 = gpd.read_file(f"{ROOT}/data/greek_seagrass_meadows_v0906.geojson").to_crs(32634)
g15["geometry"] = g15.buffer(0)
sg2015 = rast(g15, [1] * len(g15)).astype(bool)

bsrc = rasterio.open(f"{ROOT}/data/bathy/emodnet_mean.tif")
bathy = np.full((H, W), np.nan, np.float32)
reproject(bsrc.read(1), bathy, src_transform=bsrc.transform, src_crs=bsrc.crs, dst_transform=T,
          dst_crs="EPSG:32634", resampling=Resampling.bilinear)
depth = np.clip(-bathy, 0, None)

ok = np.isfinite(B1) & np.isfinite(B2) & np.isfinite(B3) & np.isfinite(B8)
water = ok & (B8 < 0.05)
deep = water & (depth > 60)
# --- sun-glint correction (Hedley): R_i' = R_i - b_i (NIR - min NIR) over deep water
nir_min = np.nanpercentile(B8[deep], 1)
deg = {}
for name, R in (("B1", B1), ("B2", B2), ("B3", B3)):
    b = np.polyfit(B8[deep], R[deep], 1)[0]
    deg[name] = R - b * (B8 - nir_min)
glint_slopes = {n: float(np.polyfit(B8[deep], R[deep], 1)[0]) for n, R in (("B1", B1), ("B2", B2), ("B3", B3))}
Rdeep = {n: float(np.nanmedian(deg[n][deep])) for n in deg}
X = {n: np.log(np.clip(deg[n] - Rdeep[n], 1e-4, None)) for n in deg}

domain = water & insite & (refc > 0) & (depth <= 40) & (nvalid >= 3)
yall = (refc == 1).astype(np.int8)
by, bx = np.indices((H, W)); block = (by // 100) * 10000 + (bx // 100)  # 1 km blocks
idx = np.flatnonzero(domain.ravel())
groups = block.ravel()[idx]; y = yall.ravel()[idx]
print("domain px", idx.size, "Posidonia frac", y.mean(), "blocks", np.unique(groups).size)

def kratio(xi, xj, m):
    vi, vj = np.var(xi[m]), np.var(xj[m]); cij = np.cov(xi[m], xj[m])[0, 1]
    a = (vi - vj) / (2 * cij); return a + np.sqrt(a * a + 1)

def features(train_mask_flat, with_depth):
    sand = np.zeros(H * W, bool); sand[train_mask_flat] = True
    sand = sand.reshape(H, W) & (refc == 5)
    F = [X["B1"], X["B2"], X["B3"]]
    for a, b in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
        F.append(X[a] - kratio(X[a], X[b], sand) * X[b])
    F.append(deg["B2"] / np.clip(deg["B3"], 1e-4, None))
    if with_depth: F.append(depth)
    return np.stack([f.ravel()[idx] for f in F], 1)

gkf = GroupKFold(n_splits=5)
out = {"glint_slopes": glint_slopes, "R_deep": Rdeep, "n_domain_px": int(idx.size)}
preds = {}
for fs, wd in (("A_spectral", False), ("B_spectral+depth", True)):
    oof = np.zeros(idx.size, np.int8)
    for k, (tr, te) in enumerate(gkf.split(idx, y, groups)):
        trflat = idx[tr]
        Fm = features(trflat, wd)
        sub = rng.choice(tr, size=min(30000, tr.size), replace=False)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                    n_jobs=-1, random_state=k).fit(Fm[sub], y[sub])
        oof[te] = rf.predict(Fm[te])
    preds[fs] = oof
    cm = confusion_matrix(y, oof, labels=[1, 0])  # rows=reference [Pos, Other], cols=map
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    PA_p, UA_p = tp / (tp + fn), tp / (tp + fp); PA_o, UA_o = tn / (tn + fp), tn / (tn + fn)
    res = {"confusion_rows_ref_cols_map[Pos,Other]": cm.tolist(), "OA": float((tp + tn) / cm.sum()),
           "kappa": float(cohen_kappa_score(y, oof)),
           "Posidonia": {"PA": float(PA_p), "UA": float(UA_p), "F1": float(2 * PA_p * UA_p / (PA_p + UA_p))},
           "Other": {"PA": float(PA_o), "UA": float(UA_o), "F1": float(2 * PA_o * UA_o / (PA_o + UA_o))},
           "mapped_Posidonia_ha_oof": float(oof.sum() * 0.01), "ref_Posidonia_ha_in_domain": float(y.sum() * 0.01)}
    # by depth band
    d = depth.ravel()[idx]; bands = {}
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 40)):
        m = (d >= lo) & (d < hi) if hi < 40 else (d >= lo) & (d <= hi)
        if m.sum() == 0: continue
        yy, pp = y[m], oof[m]
        tp_ = ((yy == 1) & (pp == 1)).sum()
        bands[f"{lo}-{hi} m"] = {"n_px": int(m.sum()), "ref_Pos_frac": float(yy.mean()), "OA": float((yy == pp).mean()),
                                 "Pos_PA": float(tp_ / max(yy.sum(), 1)), "Pos_UA": float(tp_ / max(pp.sum(), 1))}
    res["by_depth"] = bands
    # by reference class: share mapped as Posidonia
    rc = refc.ravel()[idx]
    res["share_mapped_Pos_by_ref_class"] = {str(c): float(oof[rc == c].mean()) for c in np.unique(rc)}
    # against the 2015 acoustic/camera survey (presence-only): share of 2015 seagrass mapped as Posidonia
    s15 = sg2015.ravel()[idx]
    res["vs_2015_survey"] = {"seagrass2015_px": int(s15.sum()), "share_mapped_Pos": float(oof[s15].mean()),
                             "share_outside2015_mapped_Pos": float(oof[~s15].mean())}
    out[fs] = res
    print(fs, json.dumps({k: res[k] for k in ("OA", "kappa", "Posidonia", "Other", "mapped_Posidonia_ha_oof")}))

# Full-domain final model (what an operational service would produce), feature set A, trained on all reference
Fm = features(idx, False)
sub = rng.choice(idx.size, 40000, replace=False)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                            random_state=0).fit(Fm[sub], y[sub])
full = rf.predict(Fm)
out["final_model_A_mapped_Posidonia_ha"] = float(full.sum() * 0.01)

# Reference-vs-reference agreement inside domain (1999-2002 map vs 2015 survey)
s15 = sg2015.ravel()[idx]
out["ref_vs_ref"] = {"Pos1999_ha": float(y.sum() * .01), "seagrass2015_ha": float(s15.sum() * .01),
                     "both_ha": float((s15 & (y == 1)).sum() * .01),
                     "share_of_1999Pos_in_2015": float((s15 & (y == 1)).sum() / y.sum()),
                     "share_of_2015_in_1999Pos": float((s15 & (y == 1)).sum() / s15.sum())}
# area bookkeeping
dsite = depth[insite & water]
out["areas_ha"] = {"site_water_px_ha": float((insite & water).sum() * .01),
                   "site_water_0_40m_ha": float((insite & water & (depth <= 40)).sum() * .01),
                   "ref_Pos_in_site_all_depths_ha": float(((refc == 1) & insite).sum() * .01),
                   "ref_Pos_in_site_deeper_40m_ha": float(((refc == 1) & insite & (depth > 40)).sum() * .01),
                   "domain_ha": float(idx.size * .01)}
json.dump(out, open(f"{ROOT}/results.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/data/maps.npz", idx=idx, y=y, oofA=preds["A_spectral"], oofB=preds["B_spectral+depth"],
                    full=full, H=H, W=W, refc=refc, sg2015=sg2015, insite=insite, depth=depth,
                    rgb=np.stack([deg["B3"] * 0 + B4, B3, B2]))
print(json.dumps({k: out[k] for k in ("final_model_A_mapped_Posidonia_ha", "ref_vs_ref", "areas_ha")}, indent=1))
