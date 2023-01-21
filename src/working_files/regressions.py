import pandas as pd
import streamlit as st
import numpy as np
from working_files import data_loading
from working_files import etl

import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs='cdn', 
                                separator=None, auto_open=False):
    with open(html_fname, 'w') as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            if separator:
                f.write(separator)
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    if auto_open:
        import pathlib, webbrowser
        uri = pathlib.Path(html_fname).absolute().as_uri()
        webbrowser.open(uri)


html_fname = 'plots.html'

def linear_regression():
    """Run linear regression on opus data."""
    opus = etl.clean_data()[0]
    data = ['Opus', 'Netflix', 'Disney+', 'Prime', 'Hulu']
    choices = st.selectbox("Pick Dataset: ", data)
    if choices == 'Opus':
        col = opus.columns
        columns = {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
            4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
            7: opus.columns[6], 8: opus.columns[7]}
        category = st.multiselect("Select a category to analyze", list(columns.keys()), format_func=lambda x: columns[x])
        if len(category) == 0:
            return opus.columns[0]
        for i in category:
            ind = columns[i]
        x_val = st.selectbox("X-value: ", col[:])
        y_val = st.selectbox("Y-val: ", col[:])
        if x_val == y_val:
            x_val = opus.columns[0]
            y_val = opus.columns[1]

        fig = px.scatter(opus, x = x_val, y = y_val, color = ind, title = f"Comparison of {y_val} on {x_val}")
        st.plotly_chart(fig)

        opus_binary = opus[[x_val, y_val]]
        opus_binary300 = opus_binary[:][:300]
        # Selecting the 1st 500 rows of the data
        fig1 = (px.scatter(opus_binary300, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of 300 {y_val} on {x_val}'))
        fig1.data[1].line.color = 'red'
        st.plotly_chart(fig1)
        fig2 = (px.scatter(opus_binary, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of {y_val} on {x_val}'))
        fig2.data[1].line.color = 'red'
        st.plotly_chart(fig2)


linear_regression()
#os.remove(html_fname)