"""Marine heatwaves after Hobday et al. 2016 (Prog. Oceanogr. 141:227-238, doi:10.1016/j.pocean.2015.12.014).

Climatology: for each day of a 366-day year, pool baseline-period values within +/-5 days (11-day window),
take the mean and the 90th percentile, then smooth both with a 31-day circular moving average.
Event: T > threshold for >= 5 consecutive days; events separated by <= 2 below-threshold days are joined
(the gap days become part of the event).  Intensity = T - climatological mean.
"""
import numpy as np, pandas as pd


def doy366(t):
    t = pd.DatetimeIndex(t)
    d = t.dayofyear.values.copy()
    nonleap = ~t.is_leap_year
    d[nonleap & (d >= 60)] += 1          # Mar 1 is always index 61 -> Feb 29 = 60
    return d - 1                          # 0..365


def climatology(t, x, base, win=5, smooth=31, pct=90):
    t = pd.DatetimeIndex(t)
    d = doy366(t)
    inb = (t.year >= base[0]) & (t.year <= base[1]) & np.isfinite(x)
    seas, thr = np.full(366, np.nan), np.full(366, np.nan)
    for k in range(366):
        dd = np.abs(((d - k + 183) % 366) - 183) <= win
        v = x[inb & dd]
        seas[k], thr[k] = v.mean(), np.percentile(v, pct)
    ker = np.ones(smooth) / smooth
    circ = lambda a: np.convolve(np.r_[a[-smooth:], a, a[:smooth]], ker, "same")[smooth:-smooth]
    return circ(seas), circ(thr)


def detect(t, x, seas, thr, min_dur=5, max_gap=2):
    t = pd.DatetimeIndex(t)
    d = doy366(t)
    s, h = seas[d], thr[d]
    exc = np.nan_to_num(x > h, nan=False).astype(bool)
    # runs of exceedance
    runs, i, n = [], 0, len(exc)
    while i < n:
        if exc[i]:
            j = i
            while j + 1 < n and exc[j + 1]:
                j += 1
            runs.append([i, j]); i = j + 1
        else:
            i += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_dur]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    inmhw = np.zeros(n, bool)
    ev = []
    for a, b in merged:
        inmhw[a:b + 1] = True
        it = x[a:b + 1] - s[a:b + 1]
        ev.append(dict(start=t[a], end=t[b], duration=b - a + 1, i_max=np.nanmax(it), i_mean=np.nanmean(it),
                       i_cum=np.nansum(it), peak=t[a + int(np.nanargmax(it))]))
    return pd.DataFrame(ev), inmhw, x - s


def annual(t, x, inmhw, anom, ev, thresholds=(28.0, 28.9, 30.0)):
    """Calendar-year indicators.  MHW days and intensities counted on the days inside events in that year."""
    df = pd.DataFrame({"x": x, "m": inmhw, "a": np.where(inmhw, anom, np.nan)}, index=pd.DatetimeIndex(t))
    g = df.groupby(df.index.year)
    out = pd.DataFrame({"mhw_days": g.m.sum(), "mhw_i_mean": g.a.mean(), "mhw_i_max": g.a.max(),
                        "mhw_i_cum": g.a.sum(min_count=0), "t_max": g.x.max(), "t_mean": g.x.mean(),
                        "n_days": g.x.count()})
    out["n_events"] = (ev.start.dt.year.value_counts().reindex(out.index).fillna(0).astype(int)
                       if len(ev) else 0)
    for th in thresholds:
        out[f"days_gt_{th:g}"] = g.x.apply(lambda v: int((v > th).sum()))
        out[f"run_gt_{th:g}"] = g.x.apply(lambda v: longest_run(v.values > th))
    return out


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def trend(years, y):
    """OLS slope with Newey-West (HAC, 2 lags) 95% CI, and Theil-Sen slope with 95% CI.  Units per decade."""
    import statsmodels.api as sm
    from scipy.stats import theilslopes
    years, y = np.asarray(years, float), np.asarray(y, float)
    ok = np.isfinite(y)
    X = sm.add_constant(years[ok])
    r = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    lo, hi = r.conf_int(0.05)[1]
    ts = theilslopes(y[ok], years[ok], 0.95)
    return dict(ols=10 * r.params[1], ols_lo=10 * lo, ols_hi=10 * hi, ols_p=float(r.pvalues[1]),
                sen=10 * ts.slope, sen_lo=10 * ts.low_slope, sen_hi=10 * ts.high_slope, n=int(ok.sum()))
