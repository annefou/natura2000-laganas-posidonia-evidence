"""Rank meadow cells by observed thermal exposure at meadow depth (reanalysis T at the nearest model level).
Primary key: mean annual days with daily T > 28 C (2016-2025); then mean annual T_max (2016-2025); then MHW days.
Cells sharing a series tie; ties get the same (dense) rank.  Tier: A = any T>28 C day in 2016-2025 on average
>= 1 day/yr; B = T_max 2016-2025 >= 27 C; C = rest.
Usage: pixi run python pipeline/climate/06_rank.py <ROOT>
"""
import sys
import numpy as np, pandas as pd

ROOT = sys.argv[1]
c = pd.read_parquet(f"{ROOT}/data/cell_metrics.parquet")
m = c[c.meadow_ref].copy()
k = ["zt_days_gt_28_2016_2025", "zt_t_max_2016_2025", "zt_mhw_days_2016_2025"]
m = m.sort_values(k, ascending=False)
m["rank"] = m[k].round(3).apply(tuple, axis=1).rank(method="dense", ascending=False).astype(int)
m["tier"] = np.where(m.zt_days_gt_28_2016_2025 >= 1, "A", np.where(m.zt_t_max_2016_2025 >= 27, "B", "C"))
cols = ["cell", "lon", "lat", "depth_m", "band", "ref_share", "map_share", "rank", "tier"] + k + [
    "zt_days_gt_28.9_2016_2025", "zt_years_tmax_gt28_1987_2025", "sst_days_gt_28_2016_2025", "sst_mhw_days_2016_2025",
    "zt_key", "phy_level_m", "phy_fallback", "phy_fb_km", "below_model_bottom", "rep_key"]
m[cols].to_csv(f"{ROOT}/tables/meadow_cells_ranked.csv", index=False)
cell_ha = 4 * np.pi * 6371007.2 ** 2 / (12 * 4 ** 17) / 1e4
t = m.groupby(["tier", "band"], observed=True).size().unstack(fill_value=0)
t["total_cells"] = t.sum(1)
t["approx_ha"] = (t.total_cells * cell_ha).round(0)
t.to_csv(f"{ROOT}/tables/tiers_by_band.csv")
print("cell area ha", round(cell_ha, 3)); print(t)
g = m.groupby("band", observed=True)[k + ["zt_years_tmax_gt28_1987_2025", "zt_mhw_days_1987_2016", "zt_t_max_1987_2016"]].agg(["mean", "min", "max"]).round(2)
g.to_csv(f"{ROOT}/tables/band_cell_stats.csv")
print(g.T)
print("distinct ranks", m["rank"].nunique(), "fallback cells by tier", m.groupby("tier").phy_fallback.mean().round(2).to_dict())
