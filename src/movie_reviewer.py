"""This will be the initial implementation of the webscraping tool for my prototype."""

import streamlit as st
import pandas as pd
import numpy as np

st.title('Movies123forMe - A Personalized Movie Selector')

DATA_FULL_PATH = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\movie_data\\Full Extract\\')
DATA_BASIC_PATH = ('C:\\Users\\solis\\OneDrive\\Documents\\comp\\movie_data\\Basic Extract\\')

# BASIC PACKAGE PART OF THE CODE

list_names_basic = ['acting_credits', 'international_financials', 'movie_identifiers', 'movie_keywords',
'movie_languages', 'movie_ratings', 'movie_releases', 'movie_summary', 'movie_video', 'people',
'production_companies', 'production_countries', 'technical_credits']

# creating text element to show the loading of the data in the app
data_load_state = st.text("Loading the movie data for the Basic Extract package... ")

@st.cache
def load_data_basic(nrows):
    # acting credits loading
    acting = pd.read_csv(DATA_BASIC_PATH + 'acting_credits.csv', nrows=nrows)
    # international financials loading
    international = pd.read_csv(DATA_BASIC_PATH + 'international_financials.csv', nrows=nrows)
    # movie identifiers loading
    identifiers = pd.read_csv(DATA_BASIC_PATH + 'movie_identifiers.csv', nrows=nrows)
    # movie keywords loading
    keywords = pd.read_csv(DATA_BASIC_PATH + 'movie_keywords.csv', nrows=nrows)
    # movie languages loading
    languages = pd.read_csv(DATA_BASIC_PATH + 'movie_languages.csv', nrows=nrows)
    # movie ratings loading
    ratings = pd.read_csv(DATA_BASIC_PATH + 'movie_ratings.csv', nrows=nrows)
    # movie releases loading
    releases = pd.read_csv(DATA_BASIC_PATH + 'movie_releases.csv', nrows=nrows)
    # movie summary loading
    summary = pd.read_csv(DATA_BASIC_PATH + 'movie_summary.csv', nrows=nrows)
    # movie video releases loading
    video = pd.read_csv(DATA_BASIC_PATH + 'movie_video_releases.csv', nrows=nrows)
    # people loading
    people = pd.read_csv(DATA_BASIC_PATH + 'people.csv', nrows=nrows)
    # production companies loading
    production = pd.read_csv(DATA_BASIC_PATH + 'production_companies.csv', nrows=nrows)
    # production countries loading
    countries = pd.read_csv(DATA_BASIC_PATH + 'production_countries.csv', nrows=nrows)
    # technical credits loading
    credits = pd.read_csv(DATA_BASIC_PATH + 'technical_credits.csv', nrows=nrows)
    # changing everything in text file to lowercase
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    acting.rename(lowercase, axis='columns', inplace=True)
    international.rename(lowercase, axis='columns', inplace=True)
    identifiers.rename(lowercase, axis='columns', inplace=True)
    keywords.rename(lowercase, axis='columns', inplace=True)
    languages.rename(lowercase, axis='columns', inplace=True)
    ratings.rename(lowercase, axis='columns', inplace=True)
    releases.rename(lowercase, axis='columns', inplace=True)
    summary.rename(lowercase, axis='columns', inplace=True)
    video.rename(lowercase, axis='columns', inplace=True)
    people.rename(lowercase, axis='columns', inplace=True)
    production.rename(lowercase, axis='columns', inplace=True)
    countries.rename(lowercase, axis='columns', inplace=True)
    credits.rename(lowercase, axis='columns', inplace=True)
    # returning all of the files
    return acting, international, identifiers, keywords, languages, ratings, releases, summary, video, people, production, countries, credits

# creating text element to show the loading of the data in the app
data_load_state = st.text("Loading the movie data for the Full Extract package... ")
# loading the first 10,000 rows of data into the pandas dataframe
act = load_data_basic(10000)[0]
inter = load_data_basic(10000)[1]
ident = load_data_basic(10000)[2]
key = load_data_basic(10000)[3]
lang = load_data_basic(10000)[4]
rati = load_data_basic(10000)[5]
releas = load_data_basic(10000)[6]
summ = load_data_basic(10000)[7]
vid = load_data_basic(10000)[8]
peop = load_data_basic(10000)[9]
prod = load_data_basic(10000)[10]
countr = load_data_basic(10000)[11]
cred = load_data_basic(10000)[12]

# FULL PACKAGE PART OF THE CODE
list_names_basic = ['acting_credits', 'international_financials', 'movie_identifiers', 'movie_keywords',
'movie_languages', 'movie_ratings', 'movie_releases', 'movie_summary', 'movie_video', 'people',
'production_companies', 'production_countries', 'technical_credits']

# creating text element to show the loading of the data in the app
data_load_state = st.text("Loading the movie data for the Basic Extract package... ")

@st.cache
def load_data_basic(nrows):
    # acting credits loading
    acting = pd.read_csv(DATA_BASIC_PATH + 'acting_credits.csv', nrows=nrows)
    # international financials loading
    international = pd.read_csv(DATA_BASIC_PATH + 'international_financials.csv', nrows=nrows)
    # movie identifiers loading
    identifiers = pd.read_csv(DATA_BASIC_PATH + 'movie_identifiers.csv', nrows=nrows)
    # movie keywords loading
    keywords = pd.read_csv(DATA_BASIC_PATH + 'movie_keywords.csv', nrows=nrows)
    # movie languages loading
    languages = pd.read_csv(DATA_BASIC_PATH + 'movie_languages.csv', nrows=nrows)
    # movie ratings loading
    ratings = pd.read_csv(DATA_BASIC_PATH + 'movie_ratings.csv', nrows=nrows)
    # movie releases loading
    releases = pd.read_csv(DATA_BASIC_PATH + 'movie_releases.csv', nrows=nrows)
    # movie summary loading
    summary = pd.read_csv(DATA_BASIC_PATH + 'movie_summary.csv', nrows=nrows)
    # movie video releases loading
    video = pd.read_csv(DATA_BASIC_PATH + 'movie_video_releases.csv', nrows=nrows)
    # people loading
    people = pd.read_csv(DATA_BASIC_PATH + 'people.csv', nrows=nrows)
    # production companies loading
    production = pd.read_csv(DATA_BASIC_PATH + 'production_companies.csv', nrows=nrows)
    # production countries loading
    countries = pd.read_csv(DATA_BASIC_PATH + 'production_countries.csv', nrows=nrows)
    # technical credits loading
    credits = pd.read_csv(DATA_BASIC_PATH + 'technical_credits.csv', nrows=nrows)
    # changing everything in text file to lowercase
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    acting.rename(lowercase, axis='columns', inplace=True)
    international.rename(lowercase, axis='columns', inplace=True)
    identifiers.rename(lowercase, axis='columns', inplace=True)
    keywords.rename(lowercase, axis='columns', inplace=True)
    languages.rename(lowercase, axis='columns', inplace=True)
    ratings.rename(lowercase, axis='columns', inplace=True)
    releases.rename(lowercase, axis='columns', inplace=True)
    summary.rename(lowercase, axis='columns', inplace=True)
    video.rename(lowercase, axis='columns', inplace=True)
    people.rename(lowercase, axis='columns', inplace=True)
    production.rename(lowercase, axis='columns', inplace=True)
    countries.rename(lowercase, axis='columns', inplace=True)
    credits.rename(lowercase, axis='columns', inplace=True)
    # returning all of the files
    return acting, international, identifiers, keywords, languages, ratings, releases, summary, video, people, production, countries, credits

# loading the first 10,000 rows of data into the pandas dataframe
act = load_data_basic(10000)[0]
inter = load_data_basic(10000)[1]
ident = load_data_basic(10000)[2]
key = load_data_basic(10000)[3]
lang = load_data_basic(10000)[4]
rati = load_data_basic(10000)[5]
releas = load_data_basic(10000)[6]
summ = load_data_basic(10000)[7]
vid = load_data_basic(10000)[8]
peop = load_data_basic(10000)[9]
prod = load_data_basic(10000)[10]
countr = load_data_basic(10000)[11]
cred = load_data_basic(10000)[12]

# notify the reader that the data was successfully loaded
data_load_state.text("Done!")

# basic package display
if st.checkbox("Click to see all of the raw data for the Basic Extract:"):
    st.subheader(f"Raw data for {list_names_basic[0]}:")
    st.write(act)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[1]}:")
    st.write(inter)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[2]}:")
    st.write(ident)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[3]}:")
    st.write(key)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[4]}:")
    st.write(lang)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[5]}:")
    st.write(rati)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[6]}:")
    st.write(releas)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[7]}:")
    st.write(summ)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[8]}:")
    st.write(vid)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[9]}:")
    st.write(peop)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[10]}:")
    st.write(prod)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[11]}:")
    st.write(countr)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[12]}:")
    st.write(cred)
    st.write("\n---\n")

# full package display
if st.checkbox("Click to see all of the raw data for the Full Extract:"):
    st.subheader(f"Raw data for {list_names_basic[0]}:")
    st.write(act)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[1]}:")
    st.write(inter)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[2]}:")
    st.write(ident)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[3]}:")
    st.write(key)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[4]}:")
    st.write(lang)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[5]}:")
    st.write(rati)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[6]}:")
    st.write(releas)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[7]}:")
    st.write(summ)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[8]}:")
    st.write(vid)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[9]}:")
    st.write(peop)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[10]}:")
    st.write(prod)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[11]}:")
    st.write(countr)
    st.write("\n---\n")
    st.subheader(f"Raw data for {list_names_basic[12]}:")
    st.write(cred)
    st.write("\n---\n")