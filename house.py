"""Общие признаки, проверка ручного ввода и виджеты. Не обучает модель."""
from pathlib import Path
import json
import math
import os
import joblib
import pandas as pd

NUMERIC = ['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageCars', 'TotalBsmtSF', 'FullBath']
FEATURES = NUMERIC + ['Neighborhood']
LABELS = dict(zip(FEATURES, ['Общее качество (1–10)', 'Жилая площадь над землёй, ft²', 'Год постройки', 'Мест в гараже', 'Площадь подвала, ft²', 'Полных санузлов', 'Район Эймса']))
ROOT = Path(__file__).resolve().parent


def input_frame(values, schema):
    if set(values) != set(FEATURES):
        raise ValueError('Нужны ровно семь характеристик дома.')
    valid = {}
    for name in NUMERIC:
        raw = values[name]
        try:
            num = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{LABELS[name]}: требуется целое число.') from exc
        if isinstance(raw, bool) or not math.isfinite(num) or int(num) != num:
            raise ValueError(f'{LABELS[name]}: требуется целое число.')
        bounds = schema['numeric'][name]
        if not bounds['min'] <= num <= bounds['max']:
            raise ValueError(f"{LABELS[name]}: от {bounds['min']} до {bounds['max']}.")
        valid[name] = int(num)
    area = values['Neighborhood']
    if not isinstance(area, str) or area not in schema['neighborhoods']:
        raise ValueError('Выберите район из списка.')
    valid['Neighborhood'] = area
    return pd.DataFrame([valid], columns=FEATURES)


def make_schema(fit):
    names = sorted(fit['Neighborhood'].dropna().unique().tolist())
    return {
        'features': FEATURES, 'neighborhoods': names,
        'default_neighborhood': str(fit['Neighborhood'].mode().iloc[0]),
        'numeric': {c: {'min': int(fit[c].min()), 'max': int(fit[c].max()), 'default': int(round(fit[c].median()))} for c in NUMERIC},
    }


def load_bundle():
    # Set only by the application operator. Never load a joblib uploaded by a user.
    root = Path(os.environ.get('HOUSING_ARTIFACTS', ROOT / 'artifacts'))
    return joblib.load(root / 'models.joblib'), json.loads((root / 'meta.json').read_text(encoding='utf-8'))


def widgets(st, schema):
    values = {}
    columns = st.columns(2)
    for i, name in enumerate(NUMERIC):
        bounds = schema['numeric'][name]
        values[name] = columns[i % 2].number_input(LABELS[name], min_value=bounds['min'], max_value=bounds['max'], value=bounds['default'], step=1)
    names = schema['neighborhoods']
    values['Neighborhood'] = st.selectbox(LABELS['Neighborhood'], names, index=names.index(schema['default_neighborhood']))
    return values


def input_table(frame):
    return pd.DataFrame({'Параметр': [LABELS[c] for c in FEATURES], 'Значение': [str(frame.iloc[0][c]) for c in FEATURES]})


def money(value):
    return f'{value:,.0f}'.replace(',', ' ') + ' USD'
