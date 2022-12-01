import pandas as pd
import streamlit as st
import numpy as np

import data_loading

acting = data_loading.load_data_basic(10000)[0]
inter = data_loading.load_data_basic(10000)[1]
ident = data_loading.load_data_basic(10000)[2]
key = data_loading.load_data_basic(10000)[3]
lang = data_loading.load_data_basic(10000)[4]
rati = data_loading.load_data_basic(10000)[5]
releas = data_loading.load_data_basic(10000)[6]
summ = data_loading.load_data_basic(10000)[7]
vid = data_loading.load_data_basic(10000)[8]
peop = data_loading.load_data_basic(10000)[9]
prod = data_loading.load_data_basic(10000)[10]
countr = data_loading.load_data_basic(10000)[11]
cred = data_loading.load_data_basic(10000)[12]

type(acting)

# show class type
st.write(type(acting))
# show first 5 entries
st.write(acting.head())
pd.set_option("display.max.columns", None)
# display all columns and data types
st.write(acting.info())
# describe the dataset
st.write(acting.describe())
# describe with include parameter
st.write(acting.describe(include=object))
# show how often specific values occur in a column
st.write(rati.query("rating == 'PG'"))