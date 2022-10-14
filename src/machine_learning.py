import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import data_loading as dl

st.set_option('deprecation.showPyplotGlobalUse', False)

if dl.display(choice='Home') == True:
        # loading the data into a dataset
        data_basic_load = dl.load_data_basic(100)
        data_full_load = dl.load_data_full(100)
        x1 = pd.DataFrame(data_basic_load.data, columns=data_basic_load.feature_names)
        x2 = pd.DataFrame(data_full_load.data, columns=data_full_load.feature_names)
        y1 = pd.DataFrame(data_basic_load.target, column=["MEDV"])
        y2 = pd.DataFrame(data_full_load.target, column=["MEDV"])

        df_basic = pd.DataFrame(data_basic_load.data, columns=data_basic_load.feature_names)
        df_full = pd.DataFrame(data_full_load.data, columns=data_full_load.feature_names)
        df_basic['target'] = pd.Series(data_basic_load.target)
        df_full['target'] = pd.Series(data_full_load.target)
        st.write(df_basic)
        st.write(df_full)
        # visualizing the dataset
        chart = st.sidebar.selectbox(
            label = "Select the type of chart",
            options = ["Scatterplot", "Lineplots", "Histogram", "Boxplot"]
        )
        numeric_column1 = list(df_basic.select_dtypes(['float', 'int']).columns)
        numeric_column2 = list(df_full.select_dtypes(['float', 'int']).columns)

        if chart == 'Scatterplot':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_vals1 = st.sidebar.selectbox('X Axis', options=numeric_column1)
                y_vals1 = st.sidebar.selectbox('Y Axis', options=numeric_column1)
                x_vals2 = st.sidebar.selectbox('X Axis', options=numeric_column2)
                y_vals2 = st.sidebar.selectbox('Y Axis', options=numeric_column2)
                plot1 = px.scatter(data_frame=df_basic, x=x_vals1, y=y_vals1)
                plot2 = px.scatter(data_frame=df_full, x=x_vals2, y=y_vals2)
                st.write(plot1)
                st.write(plot2)
            except Exception as e:
                print(e)
        
        if chart == 'Histogram':
            st.sidebar.subheader('Histogram Settings')
            try:
                x_vals1 = st.sidebar.selectbox('X Axis', options=numeric_column1)
                x_vals2 = st.sidebar.selectbox('X Axis', options=numeric_column2)
                plot1 = px.histogram(data_frame=df_basic, x=x_vals1)
                plot2 = px.histogram(data_frame=df_full, x=x_vals2)
                st.write(plot1)
                st.write(plot2)
            except Exception as e:
                print(e)
        
        if chart == 'Lineplot':
            st.sidebar.subheader('Lineplot Settings')
            try:
                x_vals1 = st.sidebar.selectbox('X Axis', options=numeric_column1)
                y_vals1 = st.sidebar.selectbox('Y Axis', options=numeric_column1)
                x_vals2 = st.sidebar.selectbox('X Axis', options=numeric_column2)
                y_vals2 = st.sidebar.selectbox('Y Axis', options=numeric_column2)
                plot1 = px.line(data_frame=df_basic, x=x_vals1, y=y_vals1)
                plot2 = px.line(data_frame=df_full, x=x_vals2, y=y_vals2)
                st.write(plot1)
                st.write(plot2)
            except Exception as e:
                print(e)
        
        if chart == 'Boxplot':
            st.sidebar.subheader('Boxplot Settings')
            try:
                x_vals1 = st.sidebar.selectbox('X Axis', options=numeric_column1)
                y_vals1 = st.sidebar.selectbox('Y Axis', options=numeric_column1)
                x_vals2 = st.sidebar.selectbox('X Axis', options=numeric_column2)
                y_vals2 = st.sidebar.selectbox('Y Axis', options=numeric_column2)
                plot1 = px.box(data_frame=df_basic, x=x_vals1, y=y_vals1)
                plot2 = px.box(data_frame=df_full, x=x_vals2, y=y_vals2)
                st.write(plot1)
                st.write(plot2)
            except Exception as e:
                print(e)