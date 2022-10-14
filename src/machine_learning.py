import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import data_loading as dl

st.set_option('deprecation.showPyplotGlobalUse', False)

if dl.display('Home'):
        st.write("Movies123ForMe")
        # loading the data into a dataset
        data_basic_load = dl.load_data_basic()
        data_full_load = dl.load_data_full()
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
        # numeric_column = list(df_basic.select_dtypes(['float', 'int']).colu)