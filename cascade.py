"""Inference-only routing for practice 11."""
import pandas as pd

GROUPS = {'basic': 'Качество 1–5', 'standard': 'Качество 6–7', 'premium': 'Качество 8–10'}
NAMES = {'base': 'Общий Gradient Boosting', 'ridge': 'Ridge', 'gbr': 'Gradient Boosting', 'hist': 'HistGradientBoosting'}


def segments(quality):
    return pd.cut(quality, bins=[0, 5, 7, 10], labels=list(GROUPS)).astype(str)


def predict(bundle, X):
    base = bundle['base'].predict(X)
    result = pd.DataFrame({'group': segments(X.OverallQual), 'base': base, 'specialist': float('nan'), 'prediction': base.copy()}, index=X.index)
    for name, rule in bundle['routes'].items():
        mask = result.group.eq(name)
        if rule['weight'] and mask.any():
            expert = bundle['experts'][name].predict(X.loc[mask])
            result.loc[mask, 'specialist'] = expert
            result.loc[mask, 'prediction'] += rule['weight'] * (expert - result.loc[mask, 'base'])
    return result
