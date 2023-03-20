import streamlit as st
import pandas as pd

from pathlib import Path
import plotly.express as px

# st.title('Movies123forMe - A Personalized Movie Selector')

DATA_OPUS = ('pages\\movie_data\\movie_data\\')
DATA_NETFLIX = ('pages\\movie_data\\netflix\\')
DATA_HULU = ('pages\\movie_data\\hulu\\')
DATA_PRIME = ('pages\\movie_data\\prime\\')
DATA_DISNEY = ('pages\\movie_data\\disney+\\')
DATA_IMDB = ('pages\\movie_data\\imdb\\')

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
    data1 = pd.read_csv(DATA_NETFLIX + 'archive\\netflix_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data1.rename(lowercase, axis='columns', inplace=True)
    return data1

@st.cache_data
# HULU PART OF THE CODE
def load_data_hulu(nrows):
    data2 = pd.read_csv(DATA_HULU + 'archive\\hulu_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data2.rename(lowercase, axis='columns', inplace=True)
    return data2

@st.cache_data
# PRIME PART OF THE CODE
def load_data_prime(nrows):
    data3 = pd.read_csv(DATA_PRIME + 'archive\\amazon_prime_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data3.rename(lowercase, axis='columns', inplace=True)
    return data3

@st.cache_data
# DISNEY+ PART OF THE CODE
def load_data_disney(nrows):
    data4 = pd.read_csv(DATA_DISNEY + 'archive\\disney_plus_titles.csv', nrows=nrows)
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


# basic package display
def display():
    menu = ["Home", "Opus", "Netflix", "Hulu", "Disney+", "Prime", "IMDB", "Plots"]
    choice = st.selectbox("Menu", menu, key=menu)
    # imdb display
    if choice == 'Opus':
        st.subheader("Raw data for Opus Data package:")
        st.write(load_data_opus(10000))
        st.write("\n---\n")
    if choice == 'Netflix':
        st.subheader("Raw data for Netflix Data package:")
        st.write(load_data_netflix(10000))
        st.write("\n---\n")
    if choice == 'Hulu':
        st.subheader("Raw data for Hulu Data package:")
        st.write(load_data_hulu(10000))
        st.write("\n---\n")
    if choice == 'Disney+':
        st.subheader("Raw data for Disney+ Data package:")
        st.write(load_data_disney(10000))
        st.write("\n---\n")
    if choice == 'Prime':
        st.subheader("Raw data for Prime Data package:")
        st.write(load_data_prime(10000))
        st.write("\n---\n")
    if choice == 'IMDB':
        st.subheader(f"Raw data for IMDB Data Package:")
        st.write(load_data_imdb(10000))
        st.write("\n---\n")
    if choice == 'Plots':
        st.subheader("Demo for Movies123ForMe Analysis")
        # loading the data into a dataset
        data = load_data_opus(10000)
        df = pd.DataFrame(data.values, columns=data.columns)
        st.write(df)
        # visualizing the dataset
        chart = st.sidebar.selectbox(
            label = "Select the type of chart",
            options = ["Scatterplot", "Lineplots", "Histogram", "Boxplot"]
        )
        if chart == 'Scatterplot':
            # st.sidebar.subheader('Scatterplot Settings')
            scatter_fig1 = (px.scatter(x = data['production_budget'], y = data['rating'], title = "Movie Revenue by Rating"))
            scatter_fig1.update_layout(xaxis_title = "Movie Budget", yaxis_title = "Movie Rating")
            st.write(scatter_fig1)
        if chart == 'Lineplots':
            line_fig1 = (px.line(x = data['production_budget'], y = data['rating'], title = "Movie Revenue by Rating"))
            line_fig1.update_layout(xaxis_title = "Movie Budget", yaxis_title = "Movie Rating")
            st.write(line_fig1)
        if chart == 'Histogram':
            hist_fig1 = (px.histogram(x = data['production_budget'], y = data['rating'], title = "Movie Revenue by Rating"))
            hist_fig1.update_layout(xaxis_title = "Movie Budget", yaxis_title = "Movie Rating")
            st.write(hist_fig1)
        if chart == 'Boxplot':
            box_fig1 = (px.box(x = data['production_budget'], y = data['rating'], title = "Movie Revenue by Rating"))
            box_fig1.update_layout(xaxis_title = "Movie Budget", yaxis_title = "Movie Rating")
            st.write(box_fig1)

display()