import streamlit as st
import pandas as pd

# st.title('Movies123forMe - A Personalized Movie Selector')

DATA_OPUS = ('pages/movie_data/movie_data/')
DATA_NETFLIX = ('pages/movie_data/netflix/')
DATA_HULU = ('pages/movie_data/hulu/')
DATA_PRIME = ('pages/movie_data/prime/')
DATA_DISNEY = ('pages/movie_data/disney+/')
DATA_IMDB = ('pages/movie_data/imdb/')

# OPUS PART OF THE CODE
# creating text element to show the loading of the data in the app

@st.cache_data
def load_data_opus(nrows):
    data = pd.read_csv('pages/movie_data/movie_data/movie_data.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data.rename(lowercase, axis='columns', inplace=True)
    return data

@st.cache_data
# NETFLIX PART OF THE CODE
def load_data_netflix(nrows):
    data1 = pd.read_csv(DATA_NETFLIX + 'archive/netflix_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data1.rename(lowercase, axis='columns', inplace=True)
    return data1

@st.cache_data
# HULU PART OF THE CODE
def load_data_hulu(nrows):
    data2 = pd.read_csv(DATA_HULU + 'archive/hulu_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data2.rename(lowercase, axis='columns', inplace=True)
    return data2

@st.cache_data
# PRIME PART OF THE CODE
def load_data_prime(nrows):
    data3 = pd.read_csv(DATA_PRIME + 'archive/amazon_prime_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data3.rename(lowercase, axis='columns', inplace=True)
    return data3

@st.cache_data
# DISNEY+ PART OF THE CODE
def load_data_disney(nrows):
    data4 = pd.read_csv(DATA_DISNEY + 'archive/disney_plus_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data4.rename(lowercase, axis='columns', inplace=True)
    return data4
# IMDB PACKAGE PART OF THE CODE

# creating text element to show the loading of the data in the app
@st.cache_data
def load_data_imdb(nrows):
    title_basics = pd.read_table(DATA_IMDB + 'data.tsv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    title_basics.rename(lowercase, axis='columns', inplace=True)
    return title_basics