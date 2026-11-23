"""Evaluation metrics: Poisson deviance, Gini, decile lift (issue #4).

Everything is exposure-weighted. Frequency = claims per policy-year and
the weight is exposure, the same set-up as the GLM fit in glm.py.
"""
import numpy as np
import pandas as pd


def _arrays(claims, exposure, pred):
    claims = np.asarray(claims, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if not (claims.shape == exposure.shape == pred.shape):
        raise ValueError('claims, exposure and pred must have the same length')
    if len(claims) == 0:
        raise ValueError('empty input')
    if (exposure <= 0).any() or (claims < 0).any():
        raise ValueError('exposure must be > 0 and claims must be >= 0')
    if not np.isfinite(pred).all():
        raise ValueError('predictions contain NaN or inf')
    return claims, exposure, pred


def poisson_deviance(y, pred, weight):
    """Weighted mean Poisson deviance. y = frequency, weight = exposure."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if (pred <= 0).any():
        raise ValueError('predictions must be > 0 for a Poisson deviance')
    # y*log(y/mu) is 0 when y = 0 (limit), so the unit deviance is 2*mu there.
    safe_y = np.where(y > 0, y, 1.0)
    unit = 2.0 * (np.where(y > 0, y * np.log(safe_y / pred), 0.0) - (y - pred))
    return float(np.sum(weight * unit) / np.sum(weight))


def gini(claims, pred, exposure):
    """Gini on the exposure-weighted Lorenz curve.

    Sort by predicted frequency (lowest risk first), x = cumulative share of
    exposure, y = cumulative share of claims. Gini = 1 - 2 * area under curve.
    0 = no ranking power, positive = good, negative = ranks backwards.
    Policies with an identical prediction are pooled into one step, so the
    curve is a straight line across a tie and row order cannot change the answer.
    Returns NaN if there are no claims (the curve is undefined).
    """
    claims, exposure, pred = _arrays(claims, exposure, pred)
    if claims.sum() == 0:
        return float('nan')
    g = pd.DataFrame({'p': pred, 'c': claims, 'e': exposure}).groupby('p', sort=True).sum()
    x = np.concatenate([[0.0], np.cumsum(g['e'].to_numpy()) / g['e'].sum()])
    y = np.concatenate([[0.0], np.cumsum(g['c'].to_numpy()) / g['c'].sum()])
    area = np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0)  # trapezoids
    return float(1.0 - 2.0 * area)


def lift_table(claims, pred, exposure, n=10):
    """Decile table with (approximately) equal EXPOSURE per bucket.

    Sorted by predicted frequency, each policy goes to the bucket holding
    the midpoint of its exposure. Policies are not split, so buckets are
    equal to within one policy's exposure (tiny here).
    """
    claims, exposure, pred = _arrays(claims, exposure, pred)
    df = pd.DataFrame({'claims': claims, 'pred': pred, 'exposure': exposure})
    df = df.sort_values('pred', kind='mergesort').reset_index(drop=True)
    mid = df['exposure'].cumsum() - df['exposure'] / 2.0
    df['decile'] = np.minimum((mid / df['exposure'].sum() * n).astype(int), n - 1) + 1
    df['pred_claims'] = df['pred'] * df['exposure']
    t = df.groupby('decile').agg(
        policies=('claims', 'size'),
        exposure=('exposure', 'sum'),
        claims=('claims', 'sum'),
        pred_claims=('pred_claims', 'sum'),
    )
    t['actual_freq'] = t['claims'] / t['exposure']
    t['pred_freq'] = t['pred_claims'] / t['exposure']
    t['ae'] = t['claims'] / t['pred_claims']
    return t.drop(columns='pred_claims')


def evaluate(claims, exposure, pred, null_rate=None):
    """Headline metrics. null_rate: pass the TRAIN rate when scoring valid/test."""
    claims, exposure, pred = _arrays(claims, exposure, pred)
    if null_rate is None:
        null_rate = claims.sum() / exposure.sum()
    y = claims / exposure
    dev = poisson_deviance(y, pred, exposure)
    dev0 = poisson_deviance(y, np.full(len(y), null_rate), exposure)
    return {
        'dev': dev,
        'null_dev': dev0,
        'gain': 1.0 - dev / dev0,
        'ae': float(claims.sum() / (pred * exposure).sum()),
        'gini': gini(claims, pred, exposure),
    }
