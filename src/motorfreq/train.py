"""Final comparison: Null vs GLM vs LightGBM on train, valid and TEST (issue #5).

Test is scored here for the first and only time. Run: python -m motorfreq.train
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from motorfreq.data import FEATURES, get_splits
from motorfreq.gbm import GLM_PATH, fit_gbm, predict_gbm
from motorfreq.gbm import MODEL_PATH as GBM_PATH
from motorfreq.glm import fit_glm
from motorfreq.metrics import evaluate, gini, lift_table, poisson_deviance

REPORTS = Path(__file__).resolve().parents[2] / 'reports'


def load_models(train, valid, refit):
    if not refit and GLM_PATH.exists() and GBM_PATH.exists():
        return joblib.load(GLM_PATH), joblib.load(GBM_PATH)
    glm = fit_glm(train)
    bundle = fit_gbm(train, valid)
    joblib.dump(glm, GLM_PATH)
    joblib.dump(bundle, GBM_PATH)
    return glm, bundle


def boot_diff(part, p_glm, p_gbm, n_boot, seed=42):
    """Paired bootstrap over policies. Differences are GBM minus GLM."""
    claims = part['ClaimNb'].to_numpy(float)
    expo = part['Exposure'].to_numpy(float)
    y = claims / expo
    rng = np.random.default_rng(seed)
    n = len(claims)
    d_dev, d_gini = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d_dev.append(poisson_deviance(y[i], p_gbm[i], expo[i])
                     - poisson_deviance(y[i], p_glm[i], expo[i]))
        d_gini.append(gini(claims[i], p_gbm[i], expo[i])
                      - gini(claims[i], p_glm[i], expo[i]))

    def summ(a):
        a = np.asarray(a)
        return {'mean': float(a.mean()),
                'lo95': float(np.percentile(a, 2.5)),
                'hi95': float(np.percentile(a, 97.5))}

    return {'n_boot': n_boot, 'dev_diff': summ(d_dev), 'gini_diff': summ(d_gini)}


def lift_chart(part, preds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ('glm', 'gbm')):
        t = lift_table(part['ClaimNb'], preds[name], part['Exposure'])
        ax.bar(t.index, t['actual_freq'], color='#9ecae1', label='actual')
        ax.plot(t.index, t['pred_freq'], 'o-', color='#d94801', label='predicted')
        ax.set_title(f'{name.upper()} - test deciles (equal exposure)')
        ax.set_xlabel('decile of predicted frequency (1 = lowest risk)')
        ax.set_xticks(range(1, 11))
    axes[0].set_ylabel('claims per policy-year')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--refit', action='store_true', help='refit instead of loading models')
    ap.add_argument('--boot', type=int, default=200, help='bootstrap resamples, 0 = skip')
    args = ap.parse_args()

    train, valid, test = get_splits()
    null = float(train['ClaimNb'].sum() / train['Exposure'].sum())
    glm, bundle = load_models(train, valid, args.refit)

    results, test_preds = {}, {}
    for name, part in (('train', train), ('valid', valid), ('test', test)):
        preds = {
            'null': np.full(len(part), null),
            'glm': glm.predict(part[FEATURES]),
            'gbm': predict_gbm(bundle, part),
        }
        results[name] = {
            m: evaluate(part['ClaimNb'], part['Exposure'], p, null_rate=null)
            for m, p in preds.items()
        }
        if name == 'test':
            test_preds = preds
        print(f'--- {name} ({len(part):,} policies) ---')
        for m, r in results[name].items():
            print(f"  {m:<5} dev {r['dev']:.4f}  gain {100 * r['gain']:.2f}%  "
                  f"A/E {r['ae']:.4f}  Gini {r['gini']:.4f}")
        g, b = results[name]['glm'], results[name]['gbm']
        print(f"  GBM-GLM dev {b['dev'] - g['dev']:+.4f}  Gini {b['gini'] - g['gini']:+.4f}")

    out = {
        'null_rate_train': null,
        'sizes': {'train': len(train), 'valid': len(valid), 'test': len(test)},
        'gbm_best_iteration': int(bundle['model'].best_iteration_),
        'metrics': results,
    }
    if args.boot > 0:
        boot = boot_diff(test, test_preds['glm'], test_preds['gbm'], args.boot)
        out['bootstrap_test_gbm_minus_glm'] = boot
        print(f"bootstrap test ({args.boot} resamples), GBM minus GLM, mean [95% CI]:")
        for k in ('dev_diff', 'gini_diff'):
            s = boot[k]
            print(f"  {k:<9} {s['mean']:+.4f} [{s['lo95']:+.4f}, {s['hi95']:+.4f}]")

    out['lift_test'] = {
        m: json.loads(
            lift_table(test['ClaimNb'], test_preds[m], test['Exposure'])
            .reset_index().to_json(orient='records'))
        for m in ('glm', 'gbm')
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / 'metrics.json').write_text(
        json.dumps(out, indent=2, allow_nan=False), encoding='ascii')
    lift_chart(test, test_preds, REPORTS / 'lift_test.png')
    print('wrote reports\\metrics.json and reports\\lift_test.png')


if __name__ == '__main__':
    main()
