"""Step 3b: year-to-year stability (noise floor for change monitoring), 2024 vs 2025 summer composites.

Posidonia oceanica grows a few cm per year, so real meadow-extent change between two consecutive summers
at 10 m is expected to be close to zero; any mapped difference is (mostly) classification noise.
(i) transfer: RF trained on the 2025 composite (all reference pixels, feature set A) applied to 2024;
(ii) per-year out-of-fold maps (same 1 km block 5-fold CV as 02_classify.py) compared pixel by pixel.
Same pre-processing as 02_classify.py (glint, deep-water, DII; k-ratios from reference bare-sand pixels).
Usage: pixi run python pipeline/accuracy/03_stability.py <ROOT = work/accuracy>
"""
import sys, json
import numpy as np, rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = sys.argv[1]
M = np.load(f"{ROOT}/data/maps.npz")
idx, y, refc, depth, H, W = M["idx"], M["y"], M["refc"], M["depth"], int(M["H"]), int(M["W"])

def feats(path):
    S = rasterio.open(path).read().astype(np.float32)
    B1, B2, B3, B8, nv = S[0], S[1], S[2], S[4], S[6]
    deep = np.isfinite(B8) & (B8 < 0.05) & (depth > 60) & np.isfinite(B1)
    nmin = np.nanpercentile(B8[deep], 1); X = {}; deg = {}
    for n, R in (("B1", B1), ("B2", B2), ("B3", B3)):
        b = np.polyfit(B8[deep], R[deep], 1)[0]; deg[n] = R - b * (B8 - nmin)
        X[n] = np.log(np.clip(deg[n] - np.nanmedian(deg[n][deep]), 1e-4, None))
    sand = (refc == 5) & np.isfinite(X["B1"])
    F = [X["B1"], X["B2"], X["B3"]]
    for a, c in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
        vi, vj = np.var(X[a][sand]), np.var(X[c][sand]); cij = np.cov(X[a][sand], X[c][sand])[0, 1]
        q = (vi - vj) / (2 * cij); F.append(X[a] - (q + np.sqrt(q * q + 1)) * X[c])
    F.append(deg["B2"] / np.clip(deg["B3"], 1e-4, None))
    Fm = np.stack([f.ravel()[idx] for f in F], 1)
    return Fm, np.isfinite(Fm).all(1) & (nv.ravel()[idx] >= 3)

F25, ok25 = feats(f"{ROOT}/data/s2_median_2025.tif")
F24, ok24 = feats(f"{ROOT}/data/s2_median_2024.tif")
ok = ok25 & ok24
rng = np.random.default_rng(1)
by, bx = np.indices((H, W)); groups = ((by // 100) * 10000 + (bx // 100)).ravel()[idx]

def oa(p): return float((p[ok] == y[ok]).mean())
out = {"n_px_both_years": int(ok.sum())}
# (i) transfer
sub = rng.choice(np.flatnonzero(ok), 40000, replace=False)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                            random_state=0).fit(F25[sub], y[sub])
p25, p24 = rf.predict(F25), rf.predict(F24)
out["transfer_2025model"] = {"area_2025_ha": float(p25[ok].sum() * .01), "area_2024_ha": float(p24[ok].sum() * .01),
                             "pixel_disagreement": float((p25[ok] != p24[ok]).mean()),
                             "OA_2024_vs_ref": oa(p24)}
# (ii) per-year OOF
oof = {}
for tag, F in (("2025", F25), ("2024", F24)):
    o = np.zeros(idx.size, np.int8)
    for k, (tr, te) in enumerate(GroupKFold(5).split(idx, y, groups)):
        tr = tr[ok[tr]]; s = rng.choice(tr, min(30000, tr.size), replace=False)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                                   random_state=k).fit(F[s], y[s])
        te_ok = te[ok[te]]; o[te_ok] = m.predict(F[te_ok])
    oof[tag] = o
out["oof_per_year"] = {"area_2025_ha": float(oof["2025"][ok].sum() * .01), "area_2024_ha": float(oof["2024"][ok].sum() * .01),
                       "OA_2025": oa(oof["2025"]), "OA_2024": oa(oof["2024"]),
                       "pixel_disagreement": float((oof["2025"][ok] != oof["2024"][ok]).mean())}
# disagreement aggregated to 1 ha (10x10 px) cells: share of cells whose Posidonia cover differs by >25 points
def agg(p):
    g = np.full(H * W, np.nan); g[idx[ok]] = p[ok]; g = g.reshape(H, W)[: H // 10 * 10, : W // 10 * 10]
    return np.nanmean(g.reshape(H // 10, 10, W // 10, 10), axis=(1, 3))
import warnings; warnings.simplefilter("ignore")
a, b = agg(oof["2025"]), agg(oof["2024"]); m = np.isfinite(a) & np.isfinite(b)
out["oof_per_year"]["share_1ha_cells_cover_diff_gt25pt"] = float((np.abs(a - b)[m] > 0.25).mean())
json.dump(out, open(f"{ROOT}/results_stability.json", "w"), indent=1)
print(json.dumps(out, indent=1))
