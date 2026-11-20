"""Poisson GLM baseline: fit on train, check on valid (issue #3)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from motorfreq.data import FEATURES, get_splits
from motorfreq.features import build_glm, choose_bases

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'glm.joblib'


def fit_glm(train: pd.DataFrame):
    model = build_glm(choose_bases(train))
    # y = claims per year, weight = exposure. Same fit as y = claims with a
    # log(exposure) offset, which is what you would do in Radar/Emblem.
    model.fit(train[FEATURES], train['Freq'], glm__sample_weight=train['Exposure'])
    return model


def _summary() -> None:
    train, valid, _ = get_splits()  # test stays untouched until Day 9
    bases = choose_bases(train)
    print('base levels:', bases)
    model = fit_glm(train)
    joblib.dump(model, MODEL_PATH)

    null = train['ClaimNb'].sum() / train['Exposure'].sum()
    for name, part in (('train', train), ('valid', valid)):
        pred = model.predict(part[FEATURES])
        w = part['Exposure']
        dev = mean_poisson_deviance(part['Freq'], pred, sample_weight=w)
        dev0 = mean_poisson_deviance(part['Freq'], np.full(len(part), null), sample_weight=w)
        ae = part['ClaimNb'].sum() / (pred * w).sum()
        print(f'{name:<6} null dev {dev0:.4f}  glm dev {dev:.4f}  gain {100 * (1 - dev / dev0):.2f}%  A/E {ae:.4f}')

    g = model.named_steps['glm']
    names = model.named_steps['pre'].get_feature_names_out()
    print('n coefficients:', len(g.coef_), ' iterations:', g.n_iter_)
    print('intercept:', round(float(g.intercept_), 4))
    for n, c in zip(names, g.coef_):
        if n == 'num__LogBM':
            print(f'LogBM coef: {c:.3f}')
        if 'DrivAgeBand_' in n:
            print(f'  DrivAge {n.split("DrivAgeBand_")[1]:<7} relativity {np.exp(c):.3f}')

    # API-style checks: a plain one-row frame, and an unseen category.
    row = pd.DataFrame([valid[FEATURES].iloc[0].to_dict()])
    p_batch = model.predict(valid[FEATURES].iloc[:5])[0]
    print('single-row == batch row:', bool(np.isclose(model.predict(row)[0], p_batch)))
    odd, base = row.copy(), row.copy()
    odd['VehBrand'] = 'ZZZ'
    base['VehBrand'] = bases['VehBrand']
    print('unseen brand == base brand:', bool(np.isclose(model.predict(odd)[0], model.predict(base)[0])))


if __name__ == '__main__':
    _summary()
