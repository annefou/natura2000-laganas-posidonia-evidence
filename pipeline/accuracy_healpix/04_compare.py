"""Step 5: HEALPix-native result vs the UTM-grid result (pipeline/accuracy, aggregated afterwards).

Reads <ROOT>/results.json and the UTM results in work/accuracy/results*.json; adds key "comparison_with_utm"
and prints the comparison as a Markdown table.
Usage: pixi run python pipeline/accuracy_healpix/04_compare.py <ROOT = work/accuracy_healpix> [UTM_ROOT = work/accuracy]
"""
import json, os, sys

ROOT = sys.argv[1]
U = sys.argv[2] if len(sys.argv) > 2 else "work/accuracy"
H = json.load(open(f"{ROOT}/results.json"))
if not all(os.path.exists(f"{U}/{f}") for f in ("results.json", "results_stability.json", "results_healpix.json")):
    sys.exit(f"UTM results not found in {U}: run `pixi run accuracy` first to get the comparison")
u, us, uh = (json.load(open(f"{U}/{f}")) for f in ("results.json", "results_stability.json", "results_healpix.json"))
hs, ua, ha = H["stability"]["oof_per_year"], u["A_spectral"], H["A_spectral"]
uo = us["oof_per_year"]
pct = lambda a, b: (a - b) / b * 100
rows = [
    ("Processing grid", "UTM 34N 10 m pixels (B1/B11 nearest to 10 m)", f"HEALPix d{H['grid']['depth']} cells, {H['grid']['cell_area_m2']:.2f} m2, conservative"),
    ("Domain units / area (ha)", f"{u['n_domain_px']:,} px / {u['areas_ha']['domain_ha']:,.0f}", f"{H['n_domain_cells']:,} cells / {H['areas_ha']['domain_ha']:,.0f}"),
    ("CV blocks", "1 km x 1 km (76 blocks)", f"HEALPix d{H['block_depth']} ({H['block_area_km2']:.2f} km2, {H['n_blocks']} blocks)"),
    ("OA, set A (%)", f"{ua['OA']*100:.1f}", f"{ha['OA']*100:.1f}"),
    ("kappa, set A", f"{ua['kappa']:.3f}", f"{ha['kappa']:.3f}"),
    ("Posidonia PA / UA (%)", f"{ua['Posidonia']['PA']*100:.1f} / {ua['Posidonia']['UA']*100:.1f}", f"{ha['Posidonia']['PA']*100:.1f} / {ha['Posidonia']['UA']*100:.1f}"),
    ("Other PA / UA (%)", f"{ua['Other']['PA']*100:.1f} / {ua['Other']['UA']*100:.1f}", f"{ha['Other']['PA']*100:.1f} / {ha['Other']['UA']*100:.1f}"),
    ("OA / kappa, set B (+depth)", f"{u['B_spectral+depth']['OA']*100:.1f} / {u['B_spectral+depth']['kappa']:.3f}", f"{H['B_spectral+depth']['OA']*100:.1f} / {H['B_spectral+depth']['kappa']:.3f}"),
    ("Posidonia mapped vs reference (ha)", f"{ua['mapped_Posidonia_ha_oof']:,.0f} vs {ua['ref_Posidonia_ha_in_domain']:,.0f} ({pct(ua['mapped_Posidonia_ha_oof'], ua['ref_Posidonia_ha_in_domain']):+.0f} %)",
     f"{ha['mapped_Posidonia_ha_oof']:,.0f} vs {ha['ref_Posidonia_ha_in_domain']:,.0f} ({pct(ha['mapped_Posidonia_ha_oof'], ha['ref_Posidonia_ha_in_domain']):+.0f} %)"),
    ("OA by depth 0-5/5-10/10-20/20-30/30-40 m (%)", "/".join(f"{v['OA']*100:.0f}" for v in ua["by_depth"].values()),
     "/".join(f"{v['OA']*100:.0f}" for v in ha["by_depth"].values())),
    ("2015 seagrass mapped as Posidonia (%)", f"{ua['vs_2015_survey']['share_mapped_Pos']*100:.1f}", f"{ha['vs_2015_survey']['share_mapped_Pos']*100:.1f}"),
    ("d17 cells: majority agreement (%)", f"{uh['cell_overall_accuracy_majority']*100:.1f}", f"{H['healpix_d17']['cell_overall_accuracy_majority']*100:.1f}"),
    ("d17 cells: share MAE / bias (points)", f"{uh['cell_share_mae']*100:.1f} / {uh['cell_share_bias']*100:+.1f}", f"{H['healpix_d17']['cell_share_mae']*100:.1f} / {H['healpix_d17']['cell_share_bias']*100:+.1f}"),
    ("2024 vs 2025 area, re-trained per year", f"{uo['area_2024_ha']:,.0f} vs {uo['area_2025_ha']:,.0f} ha ({pct(uo['area_2024_ha'], uo['area_2025_ha']):+.1f} %)",
     f"{hs['area_2024_ha']:,.0f} vs {hs['area_2025_ha']:,.0f} ha ({hs['area_2024_vs_2025_pct']:+.1f} %)"),
    ("2024 vs 2025 unit disagreement / ~1 ha units >25 pt", f"{uo['pixel_disagreement']*100:.1f} % / {uo['share_1ha_cells_cover_diff_gt25pt']*100:.1f} %",
     f"{hs['cell_disagreement']*100:.1f} % / {hs['share_1ha_cells_cover_diff_gt25pt']*100:.1f} %"),
    ("2025 model applied to 2024 (ha)", f"{us['transfer_2025model']['area_2024_ha']:,.0f}", f"{H['stability']['transfer_2025model']['area_2024_ha']:,.0f}"),
]
sens = f"{ROOT}/results_blocks_d12.json"
if os.path.exists(sens):
    s = json.load(open(sens))
    rows.append(("Sensitivity: d12 blocks (2.5 km2) OA / kappa A", "-", f"{s['A_spectral']['OA']*100:.1f} / {s['A_spectral']['kappa']:.3f}"))
H["comparison_with_utm"] = [{"metric": a, "utm_10m_then_aggregated": b, "healpix_native": c} for a, b, c in rows]
json.dump(H, open(f"{ROOT}/results.json", "w"), indent=1)
print("| Metric | UTM 10 m grid, then aggregated | HEALPix grid from the start |\n|---|---|---|")
for a, b, c in rows:
    print(f"| {a} | {b} | {c} |")
