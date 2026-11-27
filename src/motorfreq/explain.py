"""TreeSHAP explanations for the LightGBM model (issue #6).

contributions() imports neither matplotlib nor shap, so the API (#8) can reuse it.
Values are in LOG space: sum(contributions) + bias = log(predicted frequency).
So exp(contribution) is a multiplicative relativity, same scale as GLM coefficients.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from motorfreq.data import FEATURES, get_splits
from motorfreq.gbm import GBM_CATS, MODEL_PATH, predict_gbm, prepare

REPORTS = Path(__file__).resolve().parents[2] / 'reports'


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'{MODEL_PATH} not found. explain.py never refits on its own.')
    return joblib.load(MODEL_PATH)


def contributions(bundle: dict, df: pd.DataFrame):
    """Return (per-feature log contributions as a DataFrame, bias as an array)."""
    x = prepare(df, bundle['levels'])
    raw = bundle['model'].predict(x, pred_contrib=True)
    contrib = pd.DataFrame(raw[:, :-1], columns=list(x.columns), index=df.index)
    return contrib, raw[:, -1]


def importance_table(contrib: pd.DataFrame, exposure: pd.Series, model) -> pd.DataFrame:
    """Exposure-weighted mean |contribution| (%) next to LightGBM gain (%)."""
    w = np.average(np.abs(contrib.to_numpy()), axis=0, weights=exposure.to_numpy())
    shap_pct = pd.Series(100 * w / w.sum(), index=contrib.columns)
    gain = pd.Series(model.booster_.feature_importance('gain'),
                     index=model.booster_.feature_name())
    out = pd.DataFrame({'shap_pct': shap_pct, 'gain_pct': 100 * gain / gain.sum()})
    return out.loc[contrib.columns].sort_values('shap_pct', ascending=False)


def summary_plot(bundle: dict, sample: pd.DataFrame, path: Path) -> np.ndarray:
    """Beeswarm of TreeSHAP values. Returns the shap values for cross-checking."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap

    x = prepare(sample, bundle['levels'])
    sv = np.asarray(shap.TreeExplainer(bundle['model']).shap_values(x))
    assert sv.shape == x.shape, f'unexpected shap shape {sv.shape}'
    x_plot = x.copy()
    for c in GBM_CATS:
        x_plot[c] = x_plot[c].cat.codes  # colour is meaningless for these 4
    shap.summary_plot(sv, x_plot, show=False)
    plt.title('SHAP, LightGBM frequency model (log scale, valid sample)')
    plt.gcf().savefig(path, dpi=120, bbox_inches='tight')
    plt.close('all')
    return sv


def _summary() -> None:
    _, valid, _ = get_splits()  # test is never touched here
    bundle = load_bundle()
    model = bundle['model']

    contrib, bias = contributions(bundle, valid)
    pred = predict_gbm(bundle, valid)
    recon = np.exp(contrib.sum(axis=1).to_numpy() + bias)
    print('contrib shape:', contrib.shape, ' columns == FEATURES:',
          list(contrib.columns) == FEATURES)
    print(f'max |exp(sum + bias) - predict|: {np.abs(recon - pred).max():.2e}')
    print('bias constant:', bool(bias.max() - bias.min() < 1e-9),
          ' exp(bias) =', round(float(np.exp(bias[0])), 4))

    one, _ = contributions(bundle, valid.iloc[[0]])
    print('single-row == batch row:',
          bool(np.allclose(one.to_numpy()[0], contrib.to_numpy()[0])))

    imp = importance_table(contrib, valid['Exposure'], model)
    print('importance % (exposure-weighted mean |SHAP| vs gain):')
    print(imp.round(1).to_string())

    bm, cb, w = valid['BonusMalus'], contrib['BonusMalus'], valid['Exposure']
    for name, m in (('BM <= 55', bm <= 55), ('BM >= 100', bm >= 100)):
        print(f'mean BonusMalus contribution, {name}: '
              f'{np.average(cb[m], weights=w[m]):+.3f}  (n={int(m.sum()):,})')

    REPORTS.mkdir(exist_ok=True)
    sample = valid.sample(n=5000, random_state=42)
    sv = summary_plot(bundle, sample, REPORTS / 'shap_summary.png')
    c_s, _ = contributions(bundle, sample)
    print(f'shap vs pred_contrib, max abs diff: {np.abs(sv - c_s.to_numpy()).max():.2e}')
    print('wrote reports\\shap_summary.png')


if __name__ == '__main__':
    _summary()
