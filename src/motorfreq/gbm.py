"""LightGBM Poisson frequency model: fit on train, early-stop on valid (issue #5)."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from motorfreq.data import FEATURES, clean_features, get_splits
from motorfreq.metrics import evaluate, lift_table

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'lgbm.joblib'
GLM_PATH = MODEL_PATH.with_name('glm.joblib')

# Area stays here (the GLM drops it). Trees take the raw numerics as they are.
GBM_CATS = ['Area', 'VehBrand', 'VehGas', 'Region']

PARAMS = {
    'objective': 'poisson',
    'n_estimators': 5000,       # ceiling only, early stopping decides
    'learning_rate': 0.03,
    'num_leaves': 16,
    'min_child_samples': 300,   # ~5% of rows have a claim, so leaves need volume
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_lambda': 10.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def levels_from(train: pd.DataFrame) -> dict:
    """Category levels fixed from TRAIN. LightGBM stores this order, so a
    one-row frame must be encoded with the same levels (matters for the API)."""
    clean = clean_features(train[FEATURES])
    return {c: sorted(clean[c].astype(str).unique()) for c in GBM_CATS}


def prepare(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Same cleaning/capping as the GLM, then fixed-level categoricals.
    Unseen levels become NaN (= missing), NOT the base level as in the GLM."""
    x = clean_features(df[FEATURES]).copy()
    for c in GBM_CATS:
        s = x[c].astype(str)
        x[c] = pd.Categorical(s.where(s.isin(levels[c])), categories=levels[c])
    return x[FEATURES]


def fit_gbm(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    levels = levels_from(train)
    model = lgb.LGBMRegressor(**PARAMS)
    # y = claims per year, weight = exposure: same likelihood as the GLM fit.
    model.fit(
        prepare(train, levels),
        train['Freq'],
        sample_weight=train['Exposure'],
        eval_set=[(prepare(valid, levels), valid['Freq'])],
        eval_sample_weight=[valid['Exposure']],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return {'model': model, 'levels': levels}


def predict_gbm(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return bundle['model'].predict(prepare(df, bundle['levels']))


def _summary() -> None:
    train, valid, _ = get_splits()  # test stays untouched until Day 9
    missing = [c for c in GBM_CATS if c not in FEATURES]
    assert not missing, f'not in FEATURES: {missing}'

    bundle = fit_gbm(train, valid)
    joblib.dump(bundle, MODEL_PATH)
    model = bundle['model']
    print('best iteration:', model.best_iteration_)

    null = train['ClaimNb'].sum() / train['Exposure'].sum()  # train rate for both
    res = {}
    for name, part in (('train', train), ('valid', valid)):
        pred = predict_gbm(bundle, part)
        m = evaluate(part['ClaimNb'], part['Exposure'], pred, null_rate=null)
        res[name] = m
        print(
            f"{name:<6} null dev {m['null_dev']:.4f}  gbm dev {m['dev']:.4f}  "
            f"gain {100 * m['gain']:.2f}%  A/E {m['ae']:.4f}  Gini {m['gini']:.4f}"
        )
        if name == 'valid':
            vpred = pred

    if GLM_PATH.exists():
        glm = joblib.load(GLM_PATH)
        g = evaluate(valid['ClaimNb'], valid['Exposure'],
                     glm.predict(valid[FEATURES]), null_rate=null)
        b = res['valid']
        print('valid, GLM vs LightGBM (diff < 0.01 in dev or Gini = noise):')
        for k in ('dev', 'gain', 'ae', 'gini'):
            print(f'  {k:<5} glm {g[k]:.4f}  gbm {b[k]:.4f}  diff {b[k] - g[k]:+.4f}')

    print('valid decile lift (equal exposure, sorted by predicted frequency):')
    print(lift_table(valid['ClaimNb'], vpred, valid['Exposure']).to_string(
        float_format=lambda v: f'{v:,.4f}'))

    imp = pd.Series(model.booster_.feature_importance('gain'),
                    index=model.booster_.feature_name())
    print('gain importance %:')
    print((100 * imp / imp.sum()).sort_values(ascending=False).round(1).to_string())

    # API-style checks: one row must equal the same row in a batch.
    row = valid[FEATURES].iloc[[0]]
    p_one = predict_gbm(bundle, row)[0]
    p_batch = predict_gbm(bundle, valid[FEATURES].iloc[:5])[0]
    print('single-row == batch row:', bool(np.isclose(p_one, p_batch)))
    odd = row.copy()
    odd['VehBrand'] = 'ZZZ'
    print('unseen brand still predicts (as missing, not base):',
          round(float(predict_gbm(bundle, odd)[0]), 4))


if __name__ == '__main__':
    _summary()

