"""Shared helpers for the HEALPix-native accuracy pipeline (healpix-geo, WGS84 ellipsoid, NESTED).

Everything here is grid bookkeeping, no science choices:
  * the equal-area cell size on the WGS84 ellipsoid (authalic sphere, computed, not assumed);
  * the set of depth-19 cells inside the processing window and their polygons in a planar CRS;
  * exact area-overlap weights between those cells and any north-up regular raster grid
    (Sentinel-2 10/20/60 m in EPSG:32634, EMODnet bathymetry in EPSG:4326);
  * conservative (area-weighted) aggregation and area-weighted majority with those weights;
  * polygon area share per cell (healpix-geo polygon_coverage for candidates and the fully-covered flag,
    exact planar intersection for partially covered cells).
No point-spread / Gaussian kernel is used anywhere.
"""
from __future__ import annotations

import numpy as np
import shapely
import healpix_geo.nested as hn
from pyproj import Transformer

DEPTH = 19                 # ~12.4 m cells
ELLIPSOID = "WGS84"
UTM = 32634
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def authalic_radius(a: float = WGS84_A, f: float = WGS84_F) -> float:
    """Radius of the sphere with the same surface area as the ellipsoid (healpix-geo's equal-area basis)."""
    e2 = f * (2 - f)
    e = np.sqrt(e2)
    q_p = 1 + (1 - e2) / (2 * e) * np.log((1 + e) / (1 - e))   # authalic q at the pole / (1-e2) form
    return float(a * np.sqrt(q_p / 2))


def cell_area_m2(depth: int) -> float:
    """Area of one HEALPix cell on the WGS84 ellipsoid: ellipsoid area / (12 * 4**depth)."""
    return float(4 * np.pi * authalic_radius() ** 2 / (12 * 4 ** depth))


def to_utm():
    return Transformer.from_crs(4326, UTM, always_xy=True)


def to_ll():
    return Transformer.from_crs(UTM, 4326, always_xy=True)


def window_cells(bounds_utm, depth: int = DEPTH):
    """Depth-`depth` cells whose whole polygon lies inside the UTM window (so every cell is fully observed).

    Returns sorted cell ids (uint64), vertices lon/lat (N,4) and vertices in UTM (N,4,2).
    """
    x0, y0, x1, y1 = bounds_utm
    xs = np.r_[np.linspace(x0, x1, 50), np.full(50, x1), np.linspace(x1, x0, 50), np.full(50, x0)]
    ys = np.r_[np.full(50, y0), np.linspace(y0, y1, 50), np.full(50, y1), np.linspace(y1, y0, 50)]
    lon, lat = to_ll().transform(xs, ys)
    ids, _, _ = hn.polygon_coverage(np.c_[lon, lat], depth, ellipsoid=ELLIPSOID, flat=True)
    ids = np.unique(ids.astype(np.uint64))
    vlon, vlat = hn.vertices(ids, depth, ellipsoid=ELLIPSOID)
    vx, vy = to_utm().transform(vlon, vlat)
    inside = ((vx > x0) & (vx < x1) & (vy > y0) & (vy < y1)).all(1)
    return ids[inside], vlon[inside], vlat[inside], np.stack([vx[inside], vy[inside]], -1)


def grid_weights(quads: np.ndarray, x0: float, ytop: float, res_x: float, res_y: float, W: int, H: int):
    """Exact area-overlap weights between convex cell quads (N,4,2) and a north-up regular grid.

    Grid: column c spans [x0 + c*res_x, x0 + (c+1)*res_x], row r spans [ytop - (r+1)*res_y, ytop - r*res_y].
    Returns (cell_index, pixel_flat_index, w) with w = area(cell ∩ pixel) / area(cell), so sum_p w = 1 for a
    cell inside the grid. Planar areas in the grid's own coordinates: over a 12 m cell the scale factor of
    the projection is constant, so these area *fractions* equal ellipsoidal area fractions to < 1e-6.
    """
    N = len(quads)
    xmin, xmax = quads[..., 0].min(1), quads[..., 0].max(1)
    ymin, ymax = quads[..., 1].min(1), quads[..., 1].max(1)
    c0 = np.floor((xmin - x0) / res_x).astype(np.int64)
    c1 = np.floor((xmax - x0) / res_x).astype(np.int64)
    r0 = np.floor((ytop - ymax) / res_y).astype(np.int64)
    r1 = np.floor((ytop - ymin) / res_y).astype(np.int64)
    assert c0.min() >= 0 and r0.min() >= 0 and c1.max() < W and r1.max() < H, "cells outside the grid"
    nc, nr = c1 - c0 + 1, r1 - r0 + 1
    one = (nc == 1) & (nr == 1)
    out_i = [np.flatnonzero(one)]
    out_p = [(r0 * W + c0)[one]]
    out_w = [np.ones(one.sum())]
    multi = np.flatnonzero(~one)
    if multi.size:
        polys = shapely.polygons(quads[multi])
        area = shapely.area(polys)
        for dr in range(int(nr[multi].max())):
            for dc in range(int(nc[multi].max())):
                m = (dr < nr[multi]) & (dc < nc[multi])
                k = multi[m]
                rr, cc = r0[k] + dr, c0[k] + dc
                boxes = shapely.box(x0 + cc * res_x, ytop - (rr + 1) * res_y, x0 + (cc + 1) * res_x, ytop - rr * res_y)
                a = shapely.area(shapely.intersection(polys[m], boxes)) / area[m]
                keep = a > 0
                out_i.append(k[keep]); out_p.append((rr * W + cc)[keep]); out_w.append(a[keep])
    ci, pi, w = np.concatenate(out_i), np.concatenate(out_p), np.concatenate(out_w)
    tot = np.bincount(ci, w, minlength=N)
    assert np.allclose(tot, 1, atol=1e-9), f"weights do not sum to 1 (min {tot.min()}, max {tot.max()})"
    return ci.astype(np.int64), pi.astype(np.int64), w


def conservative_mean(ci, pi, w, values_flat, ncells, valid_flat=None):
    """Area-weighted mean of native pixels inside each cell, over valid pixels only.

    Returns (mean, valid_area_fraction). With every pixel valid, sum_c mean_c * A_c == sum_p v_p * A(p ∩ cells).
    """
    v = values_flat[pi].astype(np.float64)
    ok = np.isfinite(v) if valid_flat is None else (valid_flat[pi] & np.isfinite(v))
    ww = np.where(ok, w, 0.0)
    num = np.bincount(ci, ww * np.where(ok, v, 0.0), minlength=ncells)
    den = np.bincount(ci, ww, minlength=ncells)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    return mean, den


def area_majority(ci, pi, w, classes_flat, ncells, classes):
    """Area-weighted majority (mode) of a categorical raster per cell; never averaged. Returns (class, share)."""
    shares = np.stack([np.bincount(ci, w * (classes_flat[pi] == k), minlength=ncells) for k in classes], 1)
    j = shares.argmax(1)
    return np.asarray(classes)[j], shares[np.arange(ncells), j]


def polygon_share(cell_ids, quads_utm, geom_utm, depth: int = DEPTH):
    """Share of each cell's area covered by a (multi)polygon given in EPSG:32634.

    Candidates and the fully-covered flag come from healpix-geo polygon_coverage on the WGS84 ellipsoid,
    run on each exterior ring of the dissolved geometry; partially covered cells (and cells of parts with
    holes) get the exact planar intersection share. Returns (share, diagnostics).
    """
    geom_utm = shapely.make_valid(geom_utm)
    parts = [g for g in shapely.get_parts(geom_utm) if g.geom_type == "Polygon" and g.area > 0]
    tl = to_ll()
    share = np.zeros(len(cell_ids))
    exact_needed = np.zeros(len(cell_ids), bool)
    full_cells = np.zeros(len(cell_ids), bool)
    for p in parts:
        ex = np.asarray(p.exterior.coords)[:-1]
        lon, lat = tl.transform(ex[:, 0], ex[:, 1])
        try:
            ids, _, full = hn.polygon_coverage(np.c_[lon, lat], depth, ellipsoid=ELLIPSOID, flat=True)
        except Exception:                       # degenerate ring: fall back to exact test on bbox cells
            ids, full = np.array([], np.uint64), np.array([], bool)
            bx = shapely.bounds(p)
            cx, cy = quads_utm[..., 0].mean(1), quads_utm[..., 1].mean(1)
            m = (cx > bx[0] - 20) & (cx < bx[2] + 20) & (cy > bx[1] - 20) & (cy < bx[3] + 20)
            exact_needed |= m
        ids = ids.astype(np.uint64)
        pos = np.searchsorted(cell_ids, ids)
        pos = np.clip(pos, 0, len(cell_ids) - 1)
        inwin = cell_ids[pos] == ids
        pos, full = pos[inwin], np.asarray(full, bool)[inwin]
        if len(p.interiors) == 0:
            full_cells[pos[full]] = True
            exact_needed[pos[~full]] = True
        else:
            exact_needed[pos] = True
    exact_needed &= ~full_cells
    share[full_cells] = 1.0
    k = np.flatnonzero(exact_needed)
    if k.size:
        polys = shapely.polygons(quads_utm[k])
        shapely.prepare(geom_utm)
        share[k] = shapely.area(shapely.intersection(polys, geom_utm)) / shapely.area(polys)
    # check: polygon_coverage's fully-covered flag against the exact share on a sample
    rng = np.random.default_rng(0)
    fk = np.flatnonzero(full_cells)
    samp = rng.choice(fk, min(2000, fk.size), replace=False) if fk.size else fk
    chk = (shapely.area(shapely.intersection(shapely.polygons(quads_utm[samp]), geom_utm))
           / shapely.area(shapely.polygons(quads_utm[samp]))) if samp.size else np.array([1.0])
    diag = {"n_fully_covered": int(full_cells.sum()), "n_partial_exact": int(k.size),
            "fully_covered_flag_check_min_exact_share": float(chk.min())}
    return np.clip(share, 0, 1), diag
