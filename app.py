"""Интерфейс каскада: streamlit run app.py."""
import pandas as pd
import streamlit as st
from cascade import GROUPS, NAMES, predict
from house import input_frame, input_table, load_bundle, money, widgets

st.set_page_config(page_title='Оценка жилья: каскад', layout='centered')
st.title('Оценка жилья: каскад моделей')
st.caption('Практическая работа 11 · Плющик Вячеслав · Ames, Iowa · USD')
st.write('Введите характеристики дома. Сначала общая модель оценивает цену, потом каскад уточняет её моделью для группы качества.')


@st.cache_resource
def saved():
    return load_bundle()


try:
    bundle, meta = saved()
except (OSError, ValueError):
    st.error('Не удалось загрузить сохранённые модели. Проверьте папку artifacts.')
    st.stop()

with st.form('housing'):
    values = widgets(st, meta['schema'])
    submit = st.form_submit_button('Рассчитать цену', type='primary')

if submit:
    try:
        frame = input_frame(values, meta['schema'])
    except ValueError as exc:
        st.error(str(exc))
    else:
        result = predict(bundle, frame).iloc[0]
        route = bundle['routes'][result.group]
        st.metric('Итоговая прогнозная цена', money(result.prediction))
        st.write(f"Группа: {GROUPS[result.group]}. Специалист: {NAMES[route['model']]}; вес {route['weight']:.0%}.")
        prices = [{'Этап': 'Первый: общая модель', 'Цена': money(result.base)}]
        if route['weight']:
            prices.append({'Этап': 'Прогноз специалиста', 'Цена': money(result.specialist)})
        prices.append({'Этап': 'Итог каскада', 'Цена': money(result.prediction)})
        st.table(pd.DataFrame(prices))
        with st.expander('Введённые характеристики'):
            st.table(input_table(frame))

with st.expander('Как проверялась модель'):
    st.write('Маршруты выбраны на validation по MAE. Ниже MAE и RMSE на независимых 292 домах, в USD. Ошибки не являются интервалом для отдельного дома.')
    rows = []
    for group, result in meta['test'].items():
        rows.append({'Группа': GROUPS.get(group, 'Все дома'), 'Домов': result['count'], 'MAE общей': round(result['base']['mae']), 'MAE каскада': round(result['cascade']['mae']), 'RMSE общей': round(result['base']['rmse']), 'RMSE каскада': round(result['cascade']['rmse'])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.write('На этом тесте каскад уменьшил RMSE, но не улучшил общую MAE. Это разные способы учитывать ошибки.')
    st.markdown(f"[Источник данных: Kaggle House Prices]({meta['source']})")

st.caption('Учебный прогноз для старых данных по Эймсу, не рыночная оценка жилья сегодня.')
