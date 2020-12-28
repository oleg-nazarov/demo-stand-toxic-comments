import joblib
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

def draw():
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