"""This will be the initial implementation of the webscraping tool for my prototype."""

from symbol import test_nocond
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import plotly.express as px

# st.title('Movies123forMe - A Personalized Movie Selector')

DATA_OPUS = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\movie_data\\')
DATA_NETFLIX = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\netflix\\')
DATA_HULU = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\hulu\\')
DATA_PRIME = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\prime\\')
DATA_DISNEY = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\disney+\\')
DATA_IMDB = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\imdb\\')

# OPUS PART OF THE CODE
# creating text element to show the loading of the data in the app

@st.cache(allow_output_mutation=True)
def load_data_opus(nrows):
    data = pd.read_csv(DATA_OPUS + 'movie_data.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# NETFLIX PART OF THE CODE
@st.cache(allow_output_mutation=True)
def load_data_netflix(nrows):
    data1 = pd.read_csv(DATA_NETFLIX + 'archive\\netflix_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data1.rename(lowercase, axis='columns', inplace=True)
    return data1
# HULU PART OF THE CODE
@st.cache(allow_output_mutation=True)
def load_data_hulu(nrows):
    data2 = pd.read_csv(DATA_HULU + 'archive\\hulu_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data2.rename(lowercase, axis='columns', inplace=True)
    return data2
# PRIME PART OF THE CODE
@st.cache(allow_output_mutation=True)
def load_data_prime(nrows):
    data3 = pd.read_csv(DATA_PRIME + 'archive\\amazon_prime_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data3.rename(lowercase, axis='columns', inplace=True)
    return data3
# DISNEY+ PART OF THE CODE
@st.cache(allow_output_mutation=True)
def load_data_disney(nrows):
    data4 = pd.read_csv(DATA_DISNEY + 'archive\\disney_plus_titles.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    data4.rename(lowercase, axis='columns', inplace=True)
    return data4
# IMDB PACKAGE PART OF THE CODE
list_names_imdb = ['title.akas', 'title.basics', 'title.crew', 'title.episode', 'title.principals',
'title.ratings', 'name.basics']

# creating text element to show the loading of the data in the app
@st.cache(allow_output_mutation=True)
def load_data_imdb(nrows):
    akas = pd.read_table(DATA_IMDB + list_names_imdb[0] + '.tsv\\data.tsv', nrows=nrows)
    title_basics = pd.read_table(DATA_IMDB + list_names_imdb[1] + '.tsv\\data.tsv', nrows=nrows)
    crew = pd.read_table(DATA_IMDB + list_names_imdb[2] + '.tsv\\data.tsv', nrows=nrows)
    episode = pd.read_table(DATA_IMDB + list_names_imdb[3] + '.tsv\\data.tsv', nrows=nrows)
    principals = pd.read_table(DATA_IMDB + list_names_imdb[4] + '.tsv\\data.tsv', nrows=nrows)
    ratings = pd.read_table(DATA_IMDB + list_names_imdb[5] + '.tsv\\data.tsv', nrows=nrows)
    name_basics = pd.read_table(DATA_IMDB + list_names_imdb[6] + '.tsv\\data.tsv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    akas.rename(lowercase, axis='columns', inplace=True)
    title_basics.rename(lowercase, axis='columns', inplace=True)
    crew.rename(lowercase, axis='columns', inplace=True)
    episode.rename(lowercase, axis='columns', inplace=True)
    principals.rename(lowercase, axis='columns', inplace=True)
    ratings.rename(lowercase, axis='columns', inplace=True)
    name_basics.rename(lowercase, axis='columns', inplace=True)
    return akas, title_basics, crew, episode, principals, ratings, name_basics

# notify the reader that the data was successfully loaded

# basic package display
def display():
    menu = ["Home", "Opus", "Netflix", "Hulu", "Disney+", "Prime", "IMDB", "Demo"]
    choice = st.sidebar.selectbox("Menu", menu)
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
        st.subheader(f"Raw data for {list_names_imdb[0]}:")
        st.write(load_data_imdb(10000)[0])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[1]}:")
        st.write(load_data_imdb(10000)[1])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[2]}:")
        st.write(load_data_imdb(10000)[2])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[3]}:")
        st.write(load_data_imdb(10000)[3])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[4]}:")
        st.write(load_data_imdb(10000)[4])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[5]}:")
        st.write(load_data_imdb(10000)[5])
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_imdb[6]}:")
        st.write(load_data_imdb(10000)[6])
        st.write("\n---\n")
        st.subheader("Explanation of the variables listed in each of the different tables:")
        path = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_data\\imdb\\imdb_exp.md'
        def markdown_file(file):
            return Path(file).read_text()
        intro_markdown = markdown_file(path)
        st.markdown(intro_markdown, unsafe_allow_html=True)
    if choice == 'Demo':
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
            scatter_fig1 = (px.scatter(x = data['total_revenue'], y = data['rating'], title = "Movie Revenue by Rating"))
            scatter_fig1.update_layout(xaxis_title = "Movie Revenue", yaxis_title = "Movie Rating")
            st.write(scatter_fig1)
        if chart == 'Lineplots':
            line_fig1 = (px.line(x = data['total_revenue'], y = data['rating'], title = "Movie Revenue by Rating"))
            line_fig1.update_layout(xaxis_title = "Movie Revenue", yaxis_title = "Movie Rating")
            st.write(line_fig1)
        if chart == 'Histogram':
            hist_fig1 = (px.histogram(x = data['total_revenue'], y = data['rating'], title = "Movie Revenue by Rating"))
            hist_fig1.update_layout(xaxis_title = "Movie Revenue", yaxis_title = "Movie Rating")
            st.write(hist_fig1)
        if chart == 'Boxplot':
            box_fig1 = (px.box(x = data['total_revenue'], y = data['rating'], title = "Movie Revenue by Rating"))
            box_fig1.update_layout(xaxis_title = "Movie Revenue", yaxis_title = "Movie Rating")
            st.write(box_fig1)
