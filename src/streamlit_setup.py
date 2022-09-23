"""This will be the initial implementation of the webscraping tool for my prototype."""

import streamlit as st
import pandas as pd
import numpy as np

st.title('Movies123forMe - A Personalized Movie Selector')

DATA_BASIC_PATH = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\movie_data\\Basic Extract\\movie_ratings.csv')
DATA_FULL_PATH = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\movie_data\\Full Extract\\movie_ratings.csv')

@st.cache
def load_data(nrows):
    data_basic = pd.read_csv(DATA_BASIC_PATH, nrows=nrows)
    data_full = pd.read_csv(DATA_FULL_PATH, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data_basic.rename(lowercase, axis='columns', inplace=True)
    data_full.rename(lowercase, axis='columns', inplace=True)
    return data_basic, data_full

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data1 = load_data(10000)[0]
data2 = load_data(10000)[1]
# Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Click to show raw data for Basic Extract:'):
    st.subheader('Raw data:')
    st.write(data1)

print("\n-------------------------------\n")

if st.checkbox('Click to show raw data for Full Extract:'):
    st.subheader('Raw data:')
    st.write(data2)
