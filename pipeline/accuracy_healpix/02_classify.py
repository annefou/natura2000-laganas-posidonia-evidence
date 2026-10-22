"""Step 3 (HEALPix-native): corrections, Random Forest and spatial-block validation on depth-19 cells.

Same method as pipeline/accuracy/02_classify.py, applied to cells instead of 10 m UTM pixels:
  glint (Hedley) of B1/B2/B3 against NIR over deep water (depth > 60 m), deep-water correction
  X_i = ln(R_i - R_deep_i), Lyzenga DII for (B1,B2),(B1,B3),(B2,B3) with k-ratios from bare-sand reference
  cells of the TRAINING blocks, B2/B3 ratio; RF 300 trees, min_samples_leaf 5, balanced, 30,000 training
  samples per fold; feature sets A (spectral) and B (A + EMODnet depth).
Domain: water (B8 < 0.05), site area share > 0.5, reference class mapped (plurality), depth <= 40 m,
  >= 3 valid observations.  Label: Posidonia if the plurality class of the cell is 1120.
Blocks: HEALPix depth-13 parents (~795 m edge, 0.63 km2) of the depth-19 cells; 5-fold GroupKFold.
Areas: n_cells x cell area on the WGS84 ellipsoid (computed in 00_grid.py).
Writes <ROOT>/results.json (classification part) and <ROOT>/data/oof.parquet.
Usage: pixi run python pipeline/accuracy_healpix/02_classify.py <ROOT = work/accuracy_healpix> [block depth, default 13]
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, cohen_kappa_score

ROOT = sys.argv[1]
BLOCK_DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 13   # other values: block-size sensitivity only
RES = f"{ROOT}/results.json" if BLOCK_DEPTH == 13 else f"{ROOT}/results_blocks_d{BLOCK_DEPTH}.json"
rng = np.random.default_rng(42)
G = json.load(open(f"{ROOT}/data/grid.json"))
DEPTH = G["grid"]["depth"]
A_HA = G["grid"]["cell_area_m2"] / 1e4
C = pd.read_parquet(f"{ROOT}/data/cells.parquet")
S = pd.read_parquet(f"{ROOT}/data/composite_2025.parquet")
assert (C.cell.to_numpy() == S.cell.to_numpy()).all()
B1, B2, B3, B4, B8 = (S[b].to_numpy(np.float64) for b in ("coastal", "blue", "green", "red", "nir"))
nvalid = S.n_valid_obs.to_numpy()
depth = C.depth.to_numpy(np.float64)
refc = C.refc.to_numpy()
insite = C.site_share.to_numpy() > 0.5
sg2015 = C.sg2015_share.to_numpy() > 0.5
cell = C.cell.to_numpy()

ok = np.isfinite(B1) & np.isfinite(B2) & np.isfinite(B3) & np.isfinite(B8)
water = ok & (B8 < 0.05)
deep = water & (depth > 60)
nir_min = np.nanpercentile(B8[deep], 1)
deg, glint_slopes = {}, {}
for name, R in (("B1", B1), ("B2", B2), ("B3", B3)):
    b = np.polyfit(B8[deep], R[deep], 1)[0]
    glint_slopes[name] = float(b)
    deg[name] = R - b * (B8 - nir_min)
Rdeep = {n: float(np.nanmedian(deg[n][deep])) for n in deg}
with np.errstate(invalid="ignore"):
    X = {n: np.log(np.clip(deg[n] - Rdeep[n], 1e-4, None)) for n in deg}

domain = water & insite & (refc > 0) & (depth <= 40) & (nvalid >= 3)
idx = np.flatnonzero(domain)
y = (refc[idx] == 1).astype(np.int8)
groups = (cell[idx] >> np.uint64(2 * (DEPTH - BLOCK_DEPTH))).astype(np.int64)
print("domain cells", idx.size, "Posidonia frac", y.mean(), "blocks", np.unique(groups).size, flush=True)


def kratio(xi, xj, m):
    vi, vj = np.var(xi[m]), np.var(xj[m]); cij = np.cov(xi[m], xj[m])[0, 1]
    a = (vi - vj) / (2 * cij); return a + np.sqrt(a * a + 1)


def features(train_cells, with_depth):
    sand = np.zeros(len(cell), bool); sand[train_cells] = True
    sand &= refc == 5
    F = [X["B1"], X["B2"], X["B3"]]
    for a, b in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
        F.append(X[a] - kratio(X[a], X[b], sand) * X[b])
    F.append(deg["B2"] / np.clip(deg["B3"], 1e-4, None))
    if with_depth: F.append(depth)
    return np.stack([f[idx] for f in F], 1)


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=[1, 0])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    PA_p, UA_p = tp / (tp + fn), tp / (tp + fp); PA_o, UA_o = tn / (tn + fp), tn / (tn + fn)
    return {"confusion_rows_ref_cols_map[Pos,Other]": cm.tolist(), "OA": float((tp + tn) / cm.sum()),
            "kappa": float(cohen_kappa_score(y, p)),
            "Posidonia": {"PA": float(PA_p), "UA": float(UA_p), "F1": float(2 * PA_p * UA_p / (PA_p + UA_p))},
            "Other": {"PA": float(PA_o), "UA": float(UA_o), "F1": float(2 * PA_o * UA_o / (PA_o + UA_o))}}


gkf = GroupKFold(n_splits=5)
out = {"grid": G["grid"], "block_depth": BLOCK_DEPTH,
       "block_area_km2": G["grid"]["cell_area_m2"] * 4 ** (DEPTH - BLOCK_DEPTH) / 1e6,
       "glint_slopes": glint_slopes, "R_deep": Rdeep, "n_domain_cells": int(idx.size),
       "n_blocks": int(np.unique(groups).size)}
preds = {}
for fs, wd in (("A_spectral", False), ("B_spectral+depth", True)):
    oof = np.zeros(idx.size, np.int8)
    for k, (tr, te) in enumerate(gkf.split(idx, y, groups)):
        Fm = features(idx[tr], wd)
        sub = rng.choice(tr, size=min(30000, tr.size), replace=False)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                    n_jobs=-1, random_state=k).fit(Fm[sub], y[sub])
        oof[te] = rf.predict(Fm[te])
    preds[fs] = oof
    res = metrics(y, oof)
    res["mapped_Posidonia_ha_oof"] = float(oof.sum() * A_HA)
    res["ref_Posidonia_ha_in_domain"] = float(y.sum() * A_HA)
    res["ref_Posidonia_ha_in_domain_area_share"] = float(C.pos_share.to_numpy()[idx].sum() * A_HA)
    d = depth[idx]; bands = {}
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 40)):
        m = (d >= lo) & (d < hi) if hi < 40 else (d >= lo) & (d <= hi)
        if m.sum() == 0: continue
        yy, pp = y[m], oof[m]; tp_ = ((yy == 1) & (pp == 1)).sum()
        bands[f"{lo}-{hi} m"] = {"n_cells": int(m.sum()), "ref_Pos_frac": float(yy.mean()), "OA": float((yy == pp).mean()),
                                 "Pos_PA": float(tp_ / max(yy.sum(), 1)), "Pos_UA": float(tp_ / max(pp.sum(), 1))}
    res["by_depth"] = bands
    rc = refc[idx]
    res["share_mapped_Pos_by_ref_class"] = {str(c): float(oof[rc == c].mean()) for c in np.unique(rc)}
    s15 = sg2015[idx]
    res["vs_2015_survey"] = {"seagrass2015_cells": int(s15.sum()), "share_mapped_Pos": float(oof[s15].mean()),
                             "share_outside2015_mapped_Pos": float(oof[~s15].mean())}
    # label purity: how mixed are the cells the labels come from
    ps = C.pos_share.to_numpy()[idx]
    res["mixed_cells_share_0.1_0.9"] = float(((ps > 0.1) & (ps < 0.9)).mean())
    out[fs] = res
    print(fs, json.dumps({k: res[k] for k in ("OA", "kappa", "Posidonia", "Other", "mapped_Posidonia_ha_oof")}), flush=True)

Fm = features(idx, False)
sub = rng.choice(idx.size, 40000, replace=False)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                            random_state=0).fit(Fm[sub], y[sub])
full = rf.predict(Fm)
out["final_model_A_mapped_Posidonia_ha"] = float(full.sum() * A_HA)
s15 = sg2015[idx]
out["ref_vs_ref"] = {"Pos1999_ha": float(y.sum() * A_HA), "seagrass2015_ha": float(s15.sum() * A_HA),
                     "both_ha": float((s15 & (y == 1)).sum() * A_HA),
                     "share_of_1999Pos_in_2015": float((s15 & (y == 1)).sum() / y.sum()),
                     "share_of_2015_in_1999Pos": float((s15 & (y == 1)).sum() / s15.sum())}
out["areas_ha"] = {"site_water_ha": float((insite & water).sum() * A_HA),
                   "site_water_0_40m_ha": float((insite & water & (depth <= 40)).sum() * A_HA),
                   "ref_Pos_in_site_all_depths_ha": float(((refc == 1) & insite).sum() * A_HA),
                   "ref_Pos_in_site_deeper_40m_ha": float(((refc == 1) & insite & (depth > 40)).sum() * A_HA),
                   "domain_ha": float(idx.size * A_HA)}

# aggregation to depth 17 (same cells as pipeline/accuracy/05_healpix.py): parent = cell >> 4
par = (cell[idx] >> np.uint64(4)).astype(np.int64)
df = pd.DataFrame({"p": par, "ref": y, "ref_area": C.pos_share.to_numpy()[idx], "oof": preds["A_spectral"]})
g = df.groupby("p").agg(n=("ref", "size"), ref=("ref", "mean"), ref_area=("ref_area", "mean"), oof=("oof", "mean"))
g = g[g.n >= 7]          # >= 7 of 16 children in the domain (~ the >= 10 of ~24.7 px rule of 05_healpix.py)
out["healpix_d17"] = {"cells": int(len(g)), "cell_overall_accuracy_majority": float(((g.ref >= .5) == (g.oof >= .5)).mean()),
                      "cell_share_mae": float((g.oof - g.ref).abs().mean()), "cell_share_bias": float((g.oof - g.ref).mean()),
                      "cell_share_mae_vs_area_share_ref": float((g.oof - g.ref_area).abs().mean())}
json.dump(out, open(RES, "w"), indent=1)
if BLOCK_DEPTH == 13:
    pd.DataFrame({"cell": cell[idx], "y": y, "oofA": preds["A_spectral"], "oofB": preds["B_spectral+depth"],
                  "full": full}).to_parquet(f"{ROOT}/data/oof.parquet", index=False)
print(json.dumps({k: out[k] for k in ("final_model_A_mapped_Posidonia_ha", "ref_vs_ref", "areas_ha", "healpix_d17")}, indent=1))
