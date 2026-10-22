"""Per-meadow genetic metrics recomputed from the open multilocus genotypes of Litsi-Mizan et al. 2024 (CC BY 4.0).

Usage: pixi run python pipeline/genetics/01_genetics.py <ROOT>

Input rows are the 334 multilocus genotypes (MLGs) the authors retained after clone removal (GenClone 2.0,
psex_FIS < 0.01), i.e. one row per genet, not per ramet. Allele size 0 = missing.
Loci: the 10 loci used by the authors for diversity (Pooc-153 monomorphic and Pooc-PC045G11F Bayescan outlier excluded).
Metrics per meadow:
  N        ramets genotyped (article Table 1, transcribed; not in the ESM)
  G        MLGs = rows in ESM1
  R        clonal (genotypic) richness = (G-1)/(N-1)  (Dorken & Eckert 2001)
  G_ident  distinct allele strings among the G rows (rows identical at all 12 loci collapsed). Some meadows keep
           identical MLGs as distinct genets because psex was not significant; R_ident = (G_ident-1)/(N-1) is a
           stricter lower bound used only as a sensitivity check.
  NA       mean number of alleles per locus
  Ar       allelic richness rarefied (Hurlbert 1971 / El Mousadik & Petit 1996) to the smallest number of gene copies
           at each locus across meadows
  Ho, Hs   observed and Nei (1987) unbiased within-meadow gene diversity, averaged over loci
  FIS      1 - mean(Ho)/mean(Hs)
No Ne: 10 SSR loci on 10-27 genets per meadow are too few for a usable LD-Ne estimate; none is published either.
Outputs: data/genetics_by_meadow.csv, tables/validation_vs_published.csv
"""
import sys, os
import numpy as np, pandas as pd
from scipy.special import gammaln

ROOT = sys.argv[1]
D = f"{ROOT}/data"
os.makedirs(f"{ROOT}/tables", exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
EXCL = {"Pooc-153", "Pooc-PC045G11F", "Pooc-045G11"}     # ESM header spells the outlier "Pooc-045G11"

raw = pd.read_excel(f"{D}/inputs/litsi_mizan_2024_esm1_genotypes.xlsx", header=None)
hdr = raw.iloc[0].tolist()
assert hdr[:2] == ["Site Name", "Sample ID"]
loci_all = hdr[2::2]
assert hdr[2::2] == hdr[3::2]
g = raw.iloc[1:].reset_index(drop=True)
g.columns = ["site", "id"] + [f"{l}|{k}" for l in loci_all for k in (0, 1)]
sites = pd.read_csv(f"{D}/inputs/sites.csv")
t1 = pd.read_csv(f"{HERE}/litsi_mizan_2024_table1.csv", comment="#")
name2code = dict(zip(sites.site, sites.site_code))
g["site_code"] = g.site.str.strip().map(name2code)
assert g.site_code.notna().all(), g.site[g.site_code.isna()].unique()
loci = [l for l in loci_all if l not in EXCL]
assert len(loci) == 10, loci
A = {l: g[[f"{l}|0", f"{l}|1"]].astype(int).replace(0, np.nan).values for l in loci}


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


# rarefaction size per locus: smallest number of non-missing gene copies in any meadow
gmin = {l: min(np.isfinite(A[l][g.site_code.values == s]).all(1).sum() * 2 for s in sites.site_code) for l in loci}
rows = []
for s in sites.site_code:
    m = g.site_code.values == s
    na, ar, ho, hs = [], [], [], []
    for l in loci:
        a = A[l][m]
        a = a[np.isfinite(a).all(1)]
        n = len(a)
        al, cnt = np.unique(a.ravel(), return_counts=True)
        p = cnt / cnt.sum()
        h_o = np.mean(a[:, 0] != a[:, 1])
        h_s = n / (n - 1) * (1 - np.sum(p ** 2) - h_o / (2 * n))           # Nei 1987, as hierfstat::basic.stats
        N2, k = cnt.sum(), gmin[l]
        r = np.sum(1 - np.exp(lchoose(N2 - cnt, k) - lchoose(N2, k)) * (N2 - cnt >= k))
        na.append(len(al)); ar.append(r); ho.append(h_o); hs.append(h_s)
    strings = g.loc[m, [c for c in g.columns if "|" in c]].astype(str).agg("-".join, axis=1)
    rows.append(dict(site_code=s, G=int(m.sum()), G_ident=int(strings.nunique()), NA=np.mean(na), Ar=np.mean(ar),
                     Ho=np.mean(ho), Hs=np.mean(hs), FIS=1 - np.mean(ho) / np.mean(hs)))
gen = sites.merge(pd.DataFrame(rows), on="site_code").merge(t1[["site_code", "N"]], on="site_code")
assert (gen.G <= gen.N).all()
gen["R"] = (gen.G - 1) / (gen.N - 1)
gen["R_ident"] = (gen.G_ident - 1) / (gen.N - 1)
gen["ar_rarefied_to"] = f"{min(gmin.values())}-{max(gmin.values())} gene copies"
gen.to_csv(f"{D}/genetics_by_meadow.csv", index=False)

# validation against the published Table 1
v = gen[["site_code", "N", "G", "R", "NA", "Hs", "Ho", "FIS"]].merge(
    t1.drop(columns="N").rename(columns={"MLG": "G"}), on="site_code", suffixes=("", "_pub"))
out = {}
for c in ("G", "R", "NA", "Hs", "Ho", "FIS"):
    d = v[c] - v[f"{c}_pub"]
    out[c] = dict(max_abs_diff=float(d.abs().max()), mean_abs_diff=float(d.abs().mean()),
                  r=float(np.corrcoef(v[c], v[f"{c}_pub"])[0, 1]) if v[c].std() > 0 else np.nan)
vt = pd.DataFrame(out).T
vt.to_csv(f"{ROOT}/tables/validation_vs_published.csv")
v.round(3).to_csv(f"{ROOT}/tables/genetics_recomputed_vs_published.csv", index=False)
print(gen[["site_code", "N", "G", "G_ident", "R", "R_ident", "NA", "Ar", "Ho", "Hs", "FIS"]].round(3).to_string())
print(vt.round(3).to_string())
