"""Figures: (1) site map: S2 true colour, HCMR reference, out-of-fold classification, agreement;
(2) confusion matrix (feature set A, out-of-fold); (3) accuracy by depth band.
Usage: pixi run python pipeline/accuracy/04_figures.py <ROOT = work/accuracy>
"""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = sys.argv[1]
M = np.load(f"{ROOT}/data/maps.npz"); R = json.load(open(f"{ROOT}/results.json"))
H, W, idx = int(M["H"]), int(M["W"]), M["idx"]
TXT, TXT2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
POS, OTH, NOD = "#1baf7a", "#e8e6df", "#ffffff"
plt.rcParams.update({"font.size": 9, "text.color": TXT, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT2, "figure.facecolor": SURF, "axes.facecolor": SURF})

def grid(v, fill=-1):
    g = np.full(H * W, fill, np.int16); g[idx] = v; return g.reshape(H, W)

rgb = np.moveaxis(M["rgb"], 0, -1); wm = np.zeros(H * W, bool); wm[idx] = True; wm = wm.reshape(H, W)  # stretch on seabed-domain pixels
for c in range(3):
    lo, hi = np.nanpercentile(rgb[..., c][wm], [2, 98]); rgb[..., c] = (rgb[..., c] - lo) / (hi - lo)
rgb = np.clip(rgb, 0, 1); rgb[~np.isfinite(rgb)] = 1
outline = M["insite"].astype(float)
fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
ax[0, 0].imshow(rgb); ax[0, 0].set_title("S2 L2A median Jun-Sep 2025, B4/B3/B2 (stretched)", loc="left")
cm2 = ListedColormap([NOD, OTH, POS])
ax[0, 1].imshow(grid(M["y"]) + 1, cmap=cm2, vmin=0, vmax=2, interpolation="nearest")
ax[0, 1].set_title("Reference: HCMR Natura habitat map (1999-2002), 1120 vs other", loc="left")
ax[1, 0].imshow(grid(M["oofA"]) + 1, cmap=cm2, vmin=0, vmax=2, interpolation="nearest")
ax[1, 0].set_title("Sentinel-2 RF map (out-of-fold, spectral features only)", loc="left")
y, p = M["y"], M["oofA"]
ag = np.where((y == 1) & (p == 1), 1, np.where((y == 0) & (p == 0), 2, np.where((y == 1) & (p == 0), 3, 4)))
cm4 = ListedColormap([NOD, POS, OTH, "#eb6834", "#2a78d6"])
ax[1, 1].imshow(grid(ag, 0), cmap=cm4, vmin=0, vmax=4, interpolation="nearest")
ax[1, 1].set_title("Agreement with reference", loc="left")
for a in ax.ravel():
    a.contour(outline, levels=[0.5], colors=TXT2, linewidths=0.6); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
ax[0, 1].legend(handles=[Patch(color=POS, label="Posidonia 1120"), Patch(color=OTH, label="Other seabed")],
                loc="lower left", frameon=False, fontsize=8)
ax[1, 1].legend(handles=[Patch(color=POS, label="Both Posidonia"), Patch(color=OTH, label="Both other"),
                         Patch(color="#eb6834", label="Missed (omission)"), Patch(color="#2a78d6", label="False Posidonia (commission)")],
                loc="lower left", frameon=False, fontsize=8)
fig.suptitle("GR2210002 Laganas Bay: Posidonia extent, 0-40 m, inside site boundary (grey line). Map grid 10 m, EPSG:32634",
             x=0.01, ha="left", fontsize=10, color=TXT)
fig.savefig(f"{ROOT}/figures/map_reference_vs_classified.png", dpi=150); plt.close(fig)

# confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for a, key, ttl in ((axs[0], "A_spectral", "A: spectral only"), (axs[1], "B_spectral+depth", "B: spectral + EMODnet depth")):
    cm = np.array(R[key]["confusion_rows_ref_cols_map[Pos,Other]"]) * 0.01  # ha
    rowp = cm / cm.sum(1, keepdims=True)
    a.imshow(rowp, cmap=ListedColormap(["#e8f6f0", "#a6e0c8", "#5fc6a0", "#1baf7a"]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,.0f} ha\n{rowp[i, j]:.0%} of row", ha="center", va="center", color=TXT, fontsize=9)
    a.set_xticks([0, 1], ["Map: Posidonia", "Map: other"]); a.set_yticks([0, 1], ["Ref: Posidonia", "Ref: other"])
    r = R[key]
    a.set_title(f"{ttl}\nOA {r['OA']:.1%}, kappa {r['kappa']:.2f}, Posidonia F1 {r['Posidonia']['F1']:.2f}", loc="left", fontsize=9)
    for s in a.spines.values(): s.set_visible(False)
fig.suptitle("Confusion matrix, spatial 1 km block 5-fold hold-out (all held-out pixels)", x=0.01, ha="left", fontsize=10)
fig.savefig(f"{ROOT}/figures/confusion_matrix.png", dpi=150); plt.close(fig)

# accuracy by depth band
bd = R["A_spectral"]["by_depth"]; names = list(bd)
fig, a = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
x = np.arange(len(names)); w = 0.26
for k, (m, c, lab) in enumerate((("OA", "#2a78d6", "Overall accuracy"), ("Pos_PA", "#1baf7a", "Posidonia producer's"),
                                  ("Pos_UA", "#eb6834", "Posidonia user's"))):
    a.bar(x + (k - 1) * w, [bd[n][m] for n in names], w - 0.03, color=c, label=lab)
a.set_xticks(x, [f"{n}\n({bd[n]['n_px'] * .01:,.0f} ha, {bd[n]['ref_Pos_frac']:.0%} Pos.)" for n in names])
a.set_ylim(0, 1); a.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"]); a.grid(axis="y", color="#e8e6df", lw=0.6)
a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
a.set_xlabel("EMODnet DTM depth band (feature set A, out-of-fold)")
fig.savefig(f"{ROOT}/figures/accuracy_by_depth.png", dpi=150); plt.close(fig)
print("ok")
