import pandas as pd
import streamlit as st
import numpy as np
from functools import reduce
import data_loading

acting = data_loading.load_data_basic(100)[0]
inter = data_loading.load_data_basic(100)[1]
ident = data_loading.load_data_basic(100)[2]
key = data_loading.load_data_basic(100)[3]
lang = data_loading.load_data_basic(100)[4]
rati = data_loading.load_data_basic(100)[5]
releas = data_loading.load_data_basic(100)[6]
summ = data_loading.load_data_basic(100)[7]
vid = data_loading.load_data_basic(100)[8]
peop = data_loading.load_data_basic(100)[9]
prod = data_loading.load_data_basic(100)[10]
countr = data_loading.load_data_basic(100)[11]
cred = data_loading.load_data_basic(100)[12]

# compile the list of dataframes you want to merge
data_frames = [acting, inter, ident, key, lang, rati, releas, summ, vid, peop, prod, countr, cred]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['movie_odid'], how='outer'), data_frames)

st.write(df_merged)