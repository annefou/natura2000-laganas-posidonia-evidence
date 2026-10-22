"""Unit tests for the marine heatwave extensions in pipeline/climate/mhw.py (synthetic series, no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "climate"))
import mhw  # noqa: E402


def _series(years=40, trend=0.0, seed=0):
    t = pd.date_range("1980-01-01", f"{1980 + years - 1}-12-31", freq="D")   # whole years, so the seasonal cycle does not bias the slope
    yr = t.year + (t.dayofyear - 1) / 365.25
    rng = np.random.default_rng(seed)
    x = 20 + 5 * np.sin(2 * np.pi * (t.dayofyear / 365.25)) + trend * (yr - yr[0]) + rng.normal(0, 0.5, len(t))
    return t, np.asarray(x, float)


def test_detrend_removes_linear_trend():
    t, x = _series(trend=0.05)
    xd, slope = mhw.detrend(t, x)
    assert abs(slope - 0.05) < 0.005
    _, s2 = mhw.detrend(t, xd)
    assert abs(s2) < 1e-3
    assert abs(np.mean(xd) - np.mean(x)) < 1e-9


def test_warming_inflates_fixed_but_not_detrended_mhw_days():
    t, x = _series(trend=0.05)
    seas, thr = mhw.climatology(t, x, (1980, 1999))
    _, fixed, _ = mhw.detect(t, x, seas, thr)
    xd, _ = mhw.detrend(t, x)
    seas_d, thr_d = mhw.climatology(t, xd, (1980, 1999))
    _, det, _ = mhw.detect(t, xd, seas_d, thr_d)
    late = t.year >= 2010
    assert fixed[late].mean() > 3 * det[late].mean()


def test_categories_follow_hobday_2018():
    seas, thr = np.zeros(366), np.ones(366)
    d = np.arange(5)
    x = np.array([0.5, 1.5, 2.5, 3.5, 9.0])
    inm = np.array([False, True, True, True, True])
    assert mhw.categories(x, seas, thr, d, inm).tolist() == [0, 1, 2, 3, 4]


def test_degree_days():
    assert mhw.degree_days([27.0, 28.5, 29.0, np.nan], 28.0) == 1.5
