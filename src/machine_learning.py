import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import data_loading as dl

st.set_option('deprecation.showPyplotGlobalUse', False)


dl.display()