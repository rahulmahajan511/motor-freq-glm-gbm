"""Feature engineering and the GLM pipeline (issue #3)."""
import itertools

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from motorfreq.data import clean_features

# Interior cut points. Bands are [lo, hi), open at both ends, so an
# out-of-range input still lands in a band instead of becoming NaN.
DRIVAGE_CUTS = [21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
VEHAGE_CUTS = [1, 3, 6, 10, 15]

GLM_CATS = ['DrivAgeBand', 'VehAgeBand', 'VehPowerCat', 'VehBrand', 'VehGas', 'Region']
GLM_NUMS = ['LogDensity', 'LogBM']


def _labels(cuts):
    mid = [f'{lo}-{hi - 1}' for lo, hi in itertools.pairwise(cuts)]
    return [f'<{cuts[0]}'] + mid + [f'{cuts[-1]}+']


def _band(s, cuts):
    edges = [-np.inf] + list(cuts) + [np.inf]
    return pd.cut(s, bins=edges, labels=_labels(cuts), right=False).astype(str)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Banding and log transforms. Keeps other columns (e.g. Exposure)."""
    df = df.copy()
    df['DrivAgeBand'] = _band(df['DrivAge'], DRIVAGE_CUTS)
    df['VehAgeBand'] = _band(df['VehAge'], VEHAGE_CUTS)
    df['VehPowerCat'] = 'P' + df['VehPower'].astype(int).astype(str)
    # Floors are the mirror image of the caps: they stop log() blowing up.
    df['LogDensity'] = np.log(df['Density'].clip(lower=1))
    df['LogBM'] = np.log(df['BonusMalus'].clip(lower=50))
    for c in ('VehBrand', 'VehGas', 'Region'):
        df[c] = df[c].astype(str)
    return df


def choose_bases(df: pd.DataFrame) -> dict:
    """Base level = largest exposure, as you would set it in Emblem."""
    f = add_features(clean_features(df))
    return {c: f.groupby(c)['Exposure'].sum().idxmax() for c in GLM_CATS}


def build_glm(bases: dict) -> Pipeline:
    onehot = OneHotEncoder(
        drop=[bases[c] for c in GLM_CATS],
        handle_unknown='ignore',  # unseen level -> all zeros -> base level
    )
    pre = ColumnTransformer([('cat', onehot, GLM_CATS), ('num', 'passthrough', GLM_NUMS)])
    glm = GeneralizedLinearRegressor(family='poisson', link='log', alpha=0)
    return Pipeline([
        ('clean', FunctionTransformer(clean_features)),
        ('feat', FunctionTransformer(add_features)),
        ('pre', pre),
        ('glm', glm),
    ])
