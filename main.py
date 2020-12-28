import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib

vectorizer = joblib.load('dumped/vectorizer')
model = joblib.load('dumped/model')

def get_prediction(text):
    my_tf = vectorizer.transform(pd.DataFrame([text], columns=['text'])['text'])

    return model.predict_proba(my_tf)[0][1]

st.set_page_config(layout='wide', page_title='Demo-stand: toxic comments')

st.markdown(
    '<h1 style="text-align: center;">Модель для определения токсичных комментариев</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h2 style="text-align: center;">Как модель на основе ИИ может помочь бизнесу в определении токсичных комментариев</h2>',
    unsafe_allow_html=True,
)

with st.beta_expander(label='Попробовать online:', expanded=False):
    text_col, result_col = st.beta_columns(2)
    proba = None
    comment = ''
    
    with text_col:
        comment = st.text_area(label='Введите свой комментарий (на английском):')

        if st.button(label='Анализ') or comment != '':
            proba = get_prediction(comment)

    with result_col:
        threshold_str = st.slider(label='Задайте порог токсичности:', min_value=0.0, max_value=0.5, step=0.01, value=0.31)

        if proba == None:
            st.write('Результат: <введите комментарий в поле>')
        elif proba < float(threshold_str):
            st.write('Результат: нетоксичный.')
        else:
            st.write('Результат: токсичный.')
        
        if proba != None:
            st.text('(вероятность токсичности - {}%)'.format(int(proba * 100)))

with st.beta_expander(label='Бизнес задача:', expanded=False):
    st.write('На нашем сервисе пользователи могут оставлять комментарии под описаниями товаров. ' +\
        'Мы не публикуем токсичные записи, поэтому вручную проверяем их перед публикацией. ' +\
        'Сервис за последний месяц сильно вырос и мы перестали справляться с возросшим объёмом ' +\
        'комментариев.')

    st.write('Необходимо разгрузить отдел модераторов:\n' +\
        '  - создайте ИИ, которые будет классифицировать комментарии;\n' +\
        '  - токсичные комментарии будут отправляться в отдел модерации на окончательное заключение;\n' +\
        '  - нетоксичные будут публиковаться без доп.вмешательства человека;\n' +\
        '  - доля токсичных, которые не были идентифицированы ИИ и были опубликованы должна ' +\
        '    составлять не более 1% от всех токсичных.')

with st.beta_expander(label='Решение и результаты исследования:', expanded=True):
    graph_col, description_col = st.beta_columns(2)

    with graph_col:
        st.subheader('Возможности модели на текущий момент:')

        # dumped data for plot
        graph = joblib.load('dumped/graph')

        fig, ax_1 = plt.subplots(figsize=(12, 9))

        ax_1.plot(graph['x_axe'], graph['y_axe_left'], color='blue')
        ax_1.set_xlabel('порог токсичности (можно отрегулировать до запуска модели в работу)', fontsize=14)
        ax_1.set_yticks(np.arange(0, 101, 10))
        ax_1.set_ylabel('доля\nкомментариев для\nручной модерации, %', color='blue', fontsize=16, rotation='horizontal', ha='right')
        ax_1.set_xlim([0, 0.5])
        ax_1.set_ylim([0, 100])

        ax_1.plot([0, 0.31125], [18.47, 18.47], color='blue', linestyle='dotted')
        ax_1.plot([0.31125, 0.31125], [0, 100], color='black', linestyle='dotted') #hack with 25 (has to be on ax_2)

        ax_2 = ax_1.twinx()
        ax_2.plot(graph['x_axe'], graph['y_axe_right'], color='red')

        ax_2.plot([0.31125, 0.5], [1, 1], color='red', linestyle='dotted')

        ax_2.set_xticks(np.arange(0, 0.51, 0.05))
        ax_2.set_yticks(np.arange(0, 4.6, 0.5))
        ax_2.set_ylabel('пропущенные\nтоксичные\nкомментарии, %', color='red', fontsize=16, rotation='horizontal', ha='left')

        ax_1.text(0.04, 20, '18.47%', color='blue')
        ax_2.text(0.45, 1.1, '1%', color='red')

        st.pyplot(fig)

    with description_col:
        st.subheader('Описание:')
        
        st.write('Модель анализирует комментарий и выдает вероятность его токсичности. ' +\
            'До завершения работ над моделью можно установить порог, выше которого комментарий ' +\
            'будет считаться токсичным.')
        
        st.write('Например, установив порог токсичности 0.31 (вертикальная линия на графике), ' +\
            'модель правильно находит 99% токсичных комментариев, пропуская 1% (красная ' +\
            'горизонтальная линия). При этом отправляет на модерацию 18.47% комментариев от их общего ' +\
            'числа, т.е. снижает нагрузку на отдел модерации на 81.53%.')

        st.write('P.S. Дальнейшее улучшение модели:\n' +\
            '  - может позволить снизить % отправляемых на модерацию комментариев ' +\
            '    при сохранении 1% ошибок;\n' +\
            '  - может снизить % ошибок при сохранении % отправляемых на модерацию ' +\
            '    комментариев;\n' +\
            '  - и то, и другое при компромиссном регулировании порога токсичности.')

        st.write('P.S.S. До завершения работ над моделью возможно отрегулировать порог ' +\
            'токсичности. Например, если существует возможность просматривать ' +\
            'не 18.47%, а 50% комментариев от их общего количества, то % пропущенных ' +\
            'токсичных комментариев снизится с 1% до 0.07%')
