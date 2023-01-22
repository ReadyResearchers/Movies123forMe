import pandas as pd
import streamlit as st
import numpy as np
from src.working_files import data_loading
from src.working_files import etl

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
    data = ['Opus', 'Netflix', 'Disney+', 'Prime']
    choices = st.selectbox("Pick Dataset: ", data)
    if choices == 'Opus':
        opus = etl.clean_data()[0]
        st.write(opus)
        col = opus.columns
        columns = {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
            4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
            7: opus.columns[6], 8: opus.columns[7], 9: opus.columns[8],
            10: opus.columns[9], 11: opus.columns[10], 12: opus.columns[11],
            13: opus.columns[12], 14: opus.columns[13], 15: opus.columns[14],
            16: opus.columns[15], 17: opus.columns[16], 18: opus.columns[17],
            19: opus.columns[18]}
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
    if choices == 'Netflix':
        netflix = etl.clean_data()[1]
        st.write(netflix)
        col = netflix.columns
        columns = {5: netflix.columns[4], 6: netflix.columns[5],
            7: netflix.columns[6], 8: netflix.columns[7],
            10: netflix.columns[9], 11: netflix.columns[10], 12: netflix.columns[11],
            13: netflix.columns[12], 14: netflix.columns[13], 15: netflix.columns[14],
            16: netflix.columns[15], 17: netflix.columns[16], 18: netflix.columns[17],
            19: netflix.columns[18]}
        category = st.multiselect("Select a category to analyze", list(columns.keys()), format_func=lambda x: columns[x])
        if len(category) == 0:
            return netflix.columns[0]
        for i in category:
            ind = columns[i]
        x_val = st.selectbox("X-value: ", col[5:])
        y_val = st.selectbox("Y-val: ", col[5:])
        if x_val == y_val:
            x_val = netflix.columns[5]
            y_val = netflix.columns[6]

        fig = px.scatter(netflix, x = x_val, y = y_val, color = ind, title = f"Comparison of {y_val} on {x_val}")
        st.plotly_chart(fig)

        netflix_binary = netflix[[x_val, y_val]]
        netflix_binary300 = netflix_binary[:][:300]
        # Selecting the 1st 500 rows of the data
        fig1 = (px.scatter(netflix_binary300, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of 300 {y_val} on {x_val}'))
        fig1.data[1].line.color = 'red'
        st.plotly_chart(fig1)
        fig2 = (px.scatter(netflix_binary, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of {y_val} on {x_val}'))
        fig2.data[1].line.color = 'red'
        st.plotly_chart(fig2)

    if choices == 'Disney+':
        disney = etl.clean_data()[3]
        st.write(disney)
        col = disney.columns
        columns = {5: disney.columns[4], 6: disney.columns[5],
            7: disney.columns[6], 8: disney.columns[7],
            10: disney.columns[9], 11: disney.columns[10], 12: disney.columns[11],
            13: disney.columns[12], 14: disney.columns[13], 15: disney.columns[14],
            16: disney.columns[15], 17: disney.columns[16], 18: disney.columns[17],
            19: disney.columns[18]}
        category = st.multiselect("Select a category to analyze", list(columns.keys()), format_func=lambda x: columns[x])
        if len(category) == 0:
            return disney.columns[0]
        for i in category:
            ind = columns[i]
        x_val = st.selectbox("X-value: ", col[5:])
        y_val = st.selectbox("Y-val: ", col[5:])
        if x_val == y_val:
            x_val = disney.columns[5]
            y_val = disney.columns[6]

        fig = px.scatter(disney, x = x_val, y = y_val, color = ind, title = f"Comparison of {y_val} on {x_val}")
        st.plotly_chart(fig)

        disney_binary = disney[[x_val, y_val]]
        disney_binary300 = disney_binary[:][:300]
        # Selecting the 1st 500 rows of the data
        fig1 = (px.scatter(disney_binary300, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of 300 {y_val} on {x_val}'))
        fig1.data[1].line.color = 'red'
        st.plotly_chart(fig1)
        fig2 = (px.scatter(disney_binary, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of {y_val} on {x_val}'))
        fig2.data[1].line.color = 'red'
        st.plotly_chart(fig2)
    if choices == 'Prime':
        prime = etl.clean_data()[2]
        st.write(prime)
        col = prime.columns
        columns = {5: prime.columns[4], 6: prime.columns[5],
            7: prime.columns[6], 8: prime.columns[7],
            10: prime.columns[9], 11: prime.columns[10], 12: prime.columns[11],
            13: prime.columns[12], 14: prime.columns[13], 15: prime.columns[14]}
        category = st.multiselect("Select a category to analyze", list(columns.keys()), format_func=lambda x: columns[x])
        if len(category) == 0:
            return prime.columns[0]
        for i in category:
            ind = columns[i]
        x_val = st.selectbox("X-value: ", col[5:])
        y_val = st.selectbox("Y-val: ", col[5:])
        if x_val == y_val:
            x_val = prime.columns[5]
            y_val = prime.columns[6]

        fig = px.scatter(prime, x = x_val, y = y_val, color = ind, title = f"Comparison of {y_val} on {x_val}")
        st.plotly_chart(fig)

        prime_binary = prime[[x_val, y_val]]
        prime_binary300 = prime_binary[:][:300]
        # Selecting the 1st 500 rows of the data
        fig1 = (px.scatter(prime_binary300, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of 300 {y_val} on {x_val}'))
        fig1.data[1].line.color = 'red'
        st.plotly_chart(fig1)
        fig2 = (px.scatter(prime_binary, x =x_val, y =y_val, color = y_val,
                        trendline = 'lowess', title=f'Comparison of {y_val} on {x_val}'))
        fig2.data[1].line.color = 'red'
        st.plotly_chart(fig2)

linear_regression()
#os.remove(html_fname)