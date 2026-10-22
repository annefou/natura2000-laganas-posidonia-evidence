"""Step 4 (HEALPix-native): 2024 vs 2025 stability on depth-19 cells (same design as pipeline/accuracy/03_stability.py).

(i) transfer: RF trained on 2025 (all domain cells, feature set A, 40,000 samples) applied to 2024;
(ii) per-year out-of-fold maps (depth-13 blocks, 5-fold GroupKFold, re-trained per year) compared cell by cell;
1 ha comparison on depth-16 parents (0.99 ha, computed): share of parents whose Posidonia cover differs by > 25 points.
Adds key "stability" to <ROOT>/results.json.
Usage: pixi run python pipeline/accuracy_healpix/03_stability.py <ROOT = work/accuracy_healpix>
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = sys.argv[1]
R = json.load(open(f"{ROOT}/results.json"))
DEPTH, A_HA = R["grid"]["depth"], R["grid"]["cell_area_m2"] / 1e4
C = pd.read_parquet(f"{ROOT}/data/cells.parquet")
O = pd.read_parquet(f"{ROOT}/data/oof.parquet")
cell = C.cell.to_numpy()
idx = np.searchsorted(cell, O.cell.to_numpy()); assert (cell[idx] == O.cell.to_numpy()).all()
y, refc, depth = O.y.to_numpy(), C.refc.to_numpy(), C.depth.to_numpy(np.float64)


def feats(tag):
    S = pd.read_parquet(f"{ROOT}/data/composite_{tag}.parquet")
    B1, B2, B3, B8 = (S[b].to_numpy(np.float64) for b in ("coastal", "blue", "green", "nir"))
    nv = S.n_valid_obs.to_numpy()
    deep = np.isfinite(B8) & (B8 < 0.05) & (depth > 60) & np.isfinite(B1)
    nmin = np.nanpercentile(B8[deep], 1); X = {}; deg = {}
    with np.errstate(invalid="ignore"):
        for n, Rb in (("B1", B1), ("B2", B2), ("B3", B3)):
            b = np.polyfit(B8[deep], Rb[deep], 1)[0]; deg[n] = Rb - b * (B8 - nmin)
            X[n] = np.log(np.clip(deg[n] - np.nanmedian(deg[n][deep]), 1e-4, None))
    sand = (refc == 5) & np.isfinite(X["B1"])
    F = [X["B1"], X["B2"], X["B3"]]
    for a, c in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
        vi, vj = np.var(X[a][sand]), np.var(X[c][sand]); cij = np.cov(X[a][sand], X[c][sand])[0, 1]
        q = (vi - vj) / (2 * cij); F.append(X[a] - (q + np.sqrt(q * q + 1)) * X[c])
    F.append(deg["B2"] / np.clip(deg["B3"], 1e-4, None))
    Fm = np.stack([f[idx] for f in F], 1)
    return Fm, np.isfinite(Fm).all(1) & (nv[idx] >= 3)


F25, ok25 = feats("2025")
F24, ok24 = feats("2024")
ok = ok25 & ok24
rng = np.random.default_rng(1)
groups = (O.cell.to_numpy() >> np.uint64(2 * (DEPTH - R["block_depth"]))).astype(np.int64)


def oa(p): return float((p[ok] == y[ok]).mean())


out = {"n_cells_both_years": int(ok.sum())}
sub = rng.choice(np.flatnonzero(ok), 40000, replace=False)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                            random_state=0).fit(F25[sub], y[sub])
p25, p24 = rf.predict(np.nan_to_num(F25)), rf.predict(np.nan_to_num(F24))
out["transfer_2025model"] = {"area_2025_ha": float(p25[ok].sum() * A_HA), "area_2024_ha": float(p24[ok].sum() * A_HA),
                             "cell_disagreement": float((p25[ok] != p24[ok]).mean()), "OA_2024_vs_ref": oa(p24)}
oof = {}
for tag, F in (("2025", F25), ("2024", F24)):
    o = np.zeros(idx.size, np.int8)
    for k, (tr, te) in enumerate(GroupKFold(5).split(idx, y, groups)):
        tr = tr[ok[tr]]; s = rng.choice(tr, min(30000, tr.size), replace=False)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                                   random_state=k).fit(F[s], y[s])
        te_ok = te[ok[te]]; o[te_ok] = m.predict(F[te_ok])
    oof[tag] = o
a25, a24 = oof["2025"][ok].sum() * A_HA, oof["2024"][ok].sum() * A_HA
out["oof_per_year"] = {"area_2025_ha": float(a25), "area_2024_ha": float(a24),
                       "area_2024_vs_2025_pct": float((a24 - a25) / a25 * 100),
                       "OA_2025": oa(oof["2025"]), "OA_2024": oa(oof["2024"]),
                       "cell_disagreement": float((oof["2025"][ok] != oof["2024"][ok]).mean())}
par = (O.cell.to_numpy()[ok] >> np.uint64(2 * (DEPTH - 16))).astype(np.int64)
g = pd.DataFrame({"p": par, "a": oof["2025"][ok], "b": oof["2024"][ok]}).groupby("p").mean()
out["oof_per_year"]["parent_depth16_area_ha"] = R["grid"]["cell_area_m2"] * 4 ** (DEPTH - 16) / 1e4
out["oof_per_year"]["share_1ha_cells_cover_diff_gt25pt"] = float(((g.a - g.b).abs() > 0.25).mean())
R["stability"] = out
json.dump(R, open(f"{ROOT}/results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
