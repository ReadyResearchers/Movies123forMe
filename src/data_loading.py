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
list_names_full = ['acting_credits', 'daily_boxoffice', 'international_financials', 'movie_identifiers',
'movie_keywords', 'movie_languages', 'movie_ratings', 'movie_releases', 'movie_summary', 
'movie_video_releases', 'movie_video_summary', 'movie_weekly_bluray', 'movie_weekly_dvd',
'movie_weekly_est', 'movie_weekly_physical_disc_rental', 'movie_weekly_vod', 'people', 'production_companies',
'production_countries', 'technical_credits', 'weekend_boxoffice', 'weekend_international', 'weekly_boxoffice']

# creating text element to show the loading of the data in the app
data_load_state = st.text("Loading the movie data for the Full Extract package... ")

@st.cache
def load_data_full(nrows):
    acting_cred = pd.read_csv(DATA_FULL_PATH + list_names_full[0]+ '.csv', nrows=nrows)
    daily_box = pd.read_csv(DATA_FULL_PATH + list_names_full[1]+ '.csv', nrows=nrows)
    int_finance = pd.read_csv(DATA_FULL_PATH + list_names_full[2]+ '.csv', nrows=nrows)
    mov_ident = pd.read_csv(DATA_FULL_PATH + list_names_full[3]+ '.csv', nrows=nrows)
    mov_keyw = pd.read_csv(DATA_FULL_PATH + list_names_full[4]+ '.csv', nrows=nrows)
    mov_lang = pd.read_csv(DATA_FULL_PATH + list_names_full[5]+ '.csv', nrows=nrows)
    mov_rat = pd.read_csv(DATA_FULL_PATH + list_names_full[6]+ '.csv', nrows=nrows)
    mov_rel = pd.read_csv(DATA_FULL_PATH + list_names_full[7]+ '.csv', nrows=nrows)
    mov_summ = pd.read_csv(DATA_FULL_PATH + list_names_full[8]+ '.csv', nrows=nrows)
    mov_vid_rel = pd.read_csv(DATA_FULL_PATH + list_names_full[9]+ '.csv', nrows=nrows)
    mov_vid_summ = pd.read_csv(DATA_FULL_PATH + list_names_full[10]+ '.csv', nrows=nrows)
    mov_week_blu = pd.read_csv(DATA_FULL_PATH + list_names_full[11]+ '.csv', nrows=nrows)
    mov_week_dvd = pd.read_csv(DATA_FULL_PATH + list_names_full[12]+ '.csv', nrows=nrows)
    mov_week_est = pd.read_csv(DATA_FULL_PATH + list_names_full[13]+ '.csv', nrows=nrows)
    mov_week_phys = pd.read_csv(DATA_FULL_PATH + list_names_full[14]+ '.csv', nrows=nrows)
    mov_week_vod = pd.read_csv(DATA_FULL_PATH + list_names_full[15]+ '.csv', nrows=nrows)
    people = pd.read_csv(DATA_FULL_PATH + list_names_full[16]+ '.csv', nrows=nrows)
    prod_comp = pd.read_csv(DATA_FULL_PATH + list_names_full[17]+ '.csv', nrows=nrows)
    prod_countr = pd.read_csv(DATA_FULL_PATH + list_names_full[18]+ '.csv', nrows=nrows)
    tech_cred = pd.read_csv(DATA_FULL_PATH + list_names_full[19]+ '.csv', nrows=nrows)
    week_boxo = pd.read_csv(DATA_FULL_PATH + list_names_full[20]+ '.csv', nrows=nrows)
    week_inter = pd.read_csv(DATA_FULL_PATH + list_names_full[21]+ '.csv', nrows=nrows)
    weekly_boxo = pd.read_csv(DATA_FULL_PATH + list_names_full[22]+ '.csv', nrows=nrows)
    # changing everything in text file to lowercase
    lowercase = lambda x: str(x).lower()
    # setting up pandas dataframe for all of the files
    acting_cred.rename(lowercase, axis='columns', inplace=True)
    daily_box.rename(lowercase, axis='columns', inplace=True)
    int_finance.rename(lowercase, axis='columns', inplace=True)
    mov_ident.rename(lowercase, axis='columns', inplace=True)
    mov_keyw.rename(lowercase, axis='columns', inplace=True)
    mov_lang.rename(lowercase, axis='columns', inplace=True)
    mov_rat.rename(lowercase, axis='columns', inplace=True)
    mov_rel.rename(lowercase, axis='columns', inplace=True)
    mov_summ.rename(lowercase, axis='columns', inplace=True)
    mov_vid_rel.rename(lowercase, axis='columns', inplace=True)
    mov_vid_summ.rename(lowercase, axis='columns', inplace=True)
    mov_week_blu.rename(lowercase, axis='columns', inplace=True)
    mov_week_dvd.rename(lowercase, axis='columns', inplace=True)
    mov_week_est.rename(lowercase, axis='columns', inplace=True)
    mov_week_phys.rename(lowercase, axis='columns', inplace=True)
    mov_week_vod.rename(lowercase, axis='columns', inplace=True)
    people.rename(lowercase, axis='columns', inplace=True)
    prod_comp.rename(lowercase, axis='columns', inplace=True)
    prod_countr.rename(lowercase, axis='columns', inplace=True)
    tech_cred.rename(lowercase, axis='columns', inplace=True)
    week_boxo.rename(lowercase, axis='columns', inplace=True)
    week_inter.rename(lowercase, axis='columns', inplace=True)
    weekly_boxo.rename(lowercase, axis='columns', inplace=True)
    return acting_cred, daily_box, int_finance, mov_ident, mov_keyw, mov_lang, mov_rat, mov_rel, mov_summ, mov_vid_rel, mov_vid_summ, mov_week_blu, mov_week_dvd, mov_week_est, mov_week_phys, mov_week_vod, people, prod_comp, prod_countr, tech_cred, week_boxo, week_inter, weekly_boxo

# loading the first 10,000 rows of data into the pandas dataframe
movie1 = load_data_full(10000)[0]
movie2 = load_data_full(10000)[1]
movie3 = load_data_full(10000)[2]
movie4 = load_data_full(10000)[3]
movie5 = load_data_full(10000)[4]
movie6 = load_data_full(10000)[5]
movie7 = load_data_full(10000)[6]
movie8 = load_data_full(10000)[7]
movie9 = load_data_full(10000)[8]
movie10 = load_data_full(10000)[9]
movie11 = load_data_full(10000)[10]
movie12 = load_data_full(10000)[11]
movie13 = load_data_full(10000)[12]
movie14 = load_data_full(10000)[13]
movie15 = load_data_full(10000)[14]
movie16 = load_data_full(10000)[15]
movie17 = load_data_full(10000)[16]
movie18 = load_data_full(10000)[17]
movie19 = load_data_full(10000)[18]
movie20 = load_data_full(10000)[19]
movie21 = load_data_full(10000)[20]
movie22 = load_data_full(10000)[21]
movie23 = load_data_full(10000)[22]


# notify the reader that the data was successfully loaded
data_load_state.text("Done!")

# basic package display
def display():
    menu = ["Home", "Basic", "Full"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Basic':
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
    if choice == 'Full':
        st.subheader(f"Raw data for {list_names_full[0]}:")
        st.write(movie1)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[1]}:")
        st.write(movie2)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[2]}:")
        st.write(movie3)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[3]}:")
        st.write(movie4)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[4]}:")
        st.write(movie5)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[5]}:")
        st.write(movie6)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[6]}:")
        st.write(movie7)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[7]}:")
        st.write(movie8)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[8]}:")
        st.write(movie9)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[9]}:")
        st.write(movie10)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[10]}:")
        st.write(movie11)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[11]}:")
        st.write(movie12)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[12]}:")
        st.write(movie13)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[13]}:")
        st.write(movie14)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[14]}:")
        st.write(movie15)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[15]}:")
        st.write(movie16)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[16]}:")
        st.write(movie17)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[17]}:")
        st.write(movie18)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[18]}:")
        st.write(movie19)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[19]}:")
        st.write(movie20)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[20]}:")
        st.write(movie21)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[21]}:")
        st.write(movie22)
        st.write("\n---\n")
        st.subheader(f"Raw data for {list_names_full[22]}:")
        st.write(movie23)
        st.write("\n---\n")
    else:
        st.subheader("Home")