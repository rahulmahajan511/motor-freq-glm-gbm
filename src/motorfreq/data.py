"""Data loading, cleaning and splitting for freMTPL2freq (issue #2)."""
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CAT_COLS = ['Area', 'VehBrand', 'VehGas', 'Region']
NUM_COLS = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
FEATURES = CAT_COLS + NUM_COLS  # order matters: LightGBM stores category order

# Capping table: top of the last open-ended band for each variable.
CAPS = {'VehPower': 9, 'VehAge': 20, 'DrivAge': 90, 'BonusMalus': 150}


def load_raw() -> pd.DataFrame:
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['IDpol'] = df['IDpol'].astype(int)
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature cleaning only. Reused by the API, so no target columns here."""
    df = df.copy()
    df[NUM_COLS] = df[NUM_COLS].astype(float)
    for col, cap in CAPS.items():
        df[col] = df[col].clip(upper=cap)
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip().str.strip("'").astype('category')
    return df


def load_clean() -> pd.DataFrame:
    df = clean_features(load_raw())
    df['ClaimNb'] = df['ClaimNb'].astype(float).clip(upper=4)
    df['Exposure'] = df['Exposure'].astype(float).clip(upper=1.0)
    df['Freq'] = df['ClaimNb'] / df['Exposure']
    return df


def get_splits(seed: int = 42):
    df = load_clean()
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, valid = train_test_split(train, test_size=0.125, random_state=seed)
    return train, valid, test  # 70 / 10 / 20


def _summary() -> None:
    raw = load_raw()
    d = load_clean()
    claims, expo = d['ClaimNb'].sum(), d['Exposure'].sum()
    print('raw shape:  ', raw.shape)
    print('clean shape:', d.shape)
    print(f'claims: {claims:,.0f}   exposure: {expo:,.0f} policy-years')
    print(f'mean ClaimNb: {d["ClaimNb"].mean():.4f}   mean Exposure: {d["Exposure"].mean():.4f}')
    print(f'claim frequency (sum claims / sum exposure): {claims / expo:.4f}')
    print(f'mean of per-policy Freq (the WRONG way):     {d["Freq"].mean():.4f}')
    print(f'policies with 0 claims: {100 * (d["ClaimNb"] == 0).mean():.1f}%')
    print('rows hit by each cap:')
    hits = {
        'ClaimNb > 4': raw['ClaimNb'].astype(float) > 4,
        'Exposure > 1': raw['Exposure'].astype(float) > 1,
        'VehPower > 9': raw['VehPower'].astype(float) > 9,
        'VehAge > 20': raw['VehAge'].astype(float) > 20,
        'DrivAge > 90': raw['DrivAge'].astype(float) > 90,
        'BonusMalus > 150': raw['BonusMalus'].astype(float) > 150,
    }
    for name, mask in hits.items():
        n = int(mask.sum())
        print(f'  {name:<18}{n:>8,}  ({100 * n / len(raw):.2f}%)')
    print('levels:', {c: d[c].nunique() for c in CAT_COLS})
    print('Area:', d['Area'].cat.categories.tolist(), ' VehGas:', d['VehGas'].cat.categories.tolist())
    train, valid, test = get_splits()
    for name, part in (('train', train), ('valid', valid), ('test', test)):
        f = part['ClaimNb'].sum() / part['Exposure'].sum()
        print(f'{name:<6}{len(part):>9,}  freq {f:.4f}')
    print('first train IDpol:', train['IDpol'].head(3).tolist())


if __name__ == '__main__':
    _summary()
