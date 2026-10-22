"""Conservative (area-weighted) aggregation onto HEALPix cells preserves totals and means.

Synthetic check of pipeline/accuracy_healpix/common.py on a small window of a real UTM 34N grid
(10 m and 60 m), depth-19 healpix-geo cells on the WGS84 ellipsoid, NESTED.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpix_geo")
shapely = pytest.importorskip("shapely")

spec = importlib.util.spec_from_file_location(
    "hpx_common", Path(__file__).resolve().parents[1] / "pipeline" / "accuracy_healpix" / "common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

X0, Y0, X1, Y1 = 490020.0, 4170000.0, 490620.0, 4170600.0   # 600 m x 600 m, 60 m aligned


@pytest.fixture(scope="module")
def cells():
    return common.window_cells((X0, Y0, X1, Y1))


@pytest.mark.parametrize("res", [10, 20, 60])
def test_weights_and_integral(cells, res):
    ids, _, _, quads = cells
    W = H = int(600 / res)
    ci, pi, w = common.grid_weights(quads, X0, Y1, res, res, W, H)
    n = len(ids)
    np.testing.assert_allclose(np.bincount(ci, w, minlength=n), 1, atol=1e-12)
    area = shapely.area(shapely.polygons(quads))
    v = np.random.default_rng(1).random(W * H)
    mean, _ = common.conservative_mean(ci, pi, w, v, n)
    total_cells = (mean * area).sum()
    total_pixels = (v[pi] * w * area[ci]).sum()          # sum_p v_p * area(p ∩ cells)
    assert abs(total_cells - total_pixels) / total_pixels < 1e-12
    const, _ = common.conservative_mean(ci, pi, w, np.full(W * H, 0.25), n)
    np.testing.assert_allclose(const, 0.25, atol=1e-12)
    # area of pixels overlapped by cells never exceeds the pixel area
    assert np.bincount(pi, w * area[ci]).max() <= res * res + 1e-6


def test_mask_gives_mean_of_valid_pixels(cells):
    ids, _, _, quads = cells
    ci, pi, w = common.grid_weights(quads, X0, Y1, 10, 10, 60, 60)
    v = np.where(np.arange(3600) % 2, 1.0, 3.0)
    valid = v == 1.0
    mean, frac = common.conservative_mean(ci, pi, w, v, len(ids), valid)
    np.testing.assert_allclose(mean[frac > 0], 1.0)
    assert ((frac > 0) & (frac < 1)).all()


def test_majority_is_categorical(cells):
    ids, _, _, quads = cells
    ci, pi, w = common.grid_weights(quads, X0, Y1, 20, 20, 30, 30)
    scl = np.random.default_rng(2).choice([4, 6, 9], 900)
    cls, share = common.area_majority(ci, pi, w, scl, len(ids), [4, 6, 9])
    assert set(np.unique(cls)) <= {4, 6, 9} and (share > 1 / 3 - 1e-9).all()


def test_cell_area_equal_area_wgs84():
    from pyproj import Geod
    import healpix_geo.nested as hn
    a = common.cell_area_m2(19)
    assert abs(common.authalic_radius() - 6371007.1809) < 1e-3
    ids = hn.lonlat_to_healpix(np.array([20.9, 20.95]), np.array([37.7, 37.75]), np.uint8(19), ellipsoid="WGS84")
    lo, la = hn.vertices(ids, 19, ellipsoid="WGS84", step=8)
    g = Geod(ellps="WGS84")
    for x, y in zip(lo, la):
        assert abs(abs(g.polygon_area_perimeter(x, y)[0]) - a) / a < 1e-5
