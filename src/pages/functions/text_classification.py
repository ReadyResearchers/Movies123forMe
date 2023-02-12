import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from pages import B_etl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


## importing necessary files
duplicates = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_search.csv'
inFile = open('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_search.csv', 'r')
outFile = open('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_clean.csv', 'w')
netflix = B_etl.clean_data()[0]

# remove any \n characters in file
dups = []
for line in inFile:
    if line in dups:
        continue
    else:
        outFile.write(line)
        dups.append(line)

outFile.close()
inFile.close()

data = pd.read_csv('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_clean.csv',encoding='ISO-8859-1')

# dropping an nan values

# creating movie success column based on mean production budget of opus data
if data['earnings'].astype(float).any() >= 55507312.604108825:
    data['movie_success'] = 1
else:
    data['movie_success'] = 0

data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
'Production','Website','Response', 'movie_success','earnings']

data = data.drop_duplicates(subset = ['Title'], keep='first')

def by_plot():
    # checking for duplicate data
    movies = ['Title', 'Plot']

    movie_data = data[movies].copy()

    tfidf = TfidfVectorizer(stop_words='english')

    movie_data['Plot'] = movie_data['Plot'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['Plot'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movie_data.index, index=movie_data['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movie_data[['Title', 'Plot']].iloc[movie_indices]


    movie_list = movie_data['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list[:-1])

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        for i in movie_list:
            if i == recommended_movie_names.values.any():
                url = f'http://www.omdbapi.com/?t={i}&apikey=4482116e'
                re = requests.get(url)
                re = re.json()
                try:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(re['Poster'])
                    with col2:
                        st.subheader(re['Title'])
                        st.caption(f"Genre: {re['Genre']} | Year: {re['Year']} | Rated: {re['Rated']} | Released: {re['Released']}")
                        st.write(re['Plot'])
                        st.text(f"Rating: {re['imdbRating']}")
                        st.progress(float(re['imdbRating']) / 10)
                except:
                    st.error("No movie with that title found in the API Database YET -- add it to our database using the Movie Search feature!")


def by_actor():
    # checking for duplicate data
    movies = ['Title', 'Actors']
    movie_data = data[movies].copy()

    tfidf = TfidfVectorizer(stop_words='english')

    movie_data['Actors'] = movie_data['Actors'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['Actors'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movie_data.index, index=movie_data['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movie_data[['Title', 'Actors']].iloc[movie_indices]


    movie_list = movie_data['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list )

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        for i in movie_list:
            if i == recommended_movie_names.values.any():
                url = f'http://www.omdbapi.com/?t={i}&apikey=4482116e'
                re = requests.get(url)
                re = re.json()
                try:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(re['Poster'])
                    with col2:
                        st.subheader(re['Title'])
                        st.caption(f"Genre: {re['Genre']} | Year: {re['Year']} | Rated: {re['Rated']} | Released: {re['Released']}")
                        st.write(re['Plot'])
                        st.text(f"Rating: {re['imdbRating']}")
                        st.progress(float(re['imdbRating']) / 10)
                except:
                    st.error("No movie with that title found in the API Database YET -- add it to our database using the Movie Search feature!")


def by_director():
    movies = ['Title', 'Director']
    movie_data = data[movies].copy()

    tfidf = TfidfVectorizer(stop_words='english')

    movie_data['Director'] = movie_data['Director'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['Director'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movie_data.index, index=movie_data['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movie_data[['Title', 'Director']].iloc[movie_indices]


    movie_list = data['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list )

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        for i in movie_list:
            if i == recommended_movie_names.values.any():
                url = f'http://www.omdbapi.com/?t={i}&apikey=4482116e'
                re = requests.get(url)
                re = re.json()
                try:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(re['Poster'])
                    with col2:
                        st.subheader(re['Title'])
                        st.caption(f"Genre: {re['Genre']} | Year: {re['Year']} | Rated: {re['Rated']} | Released: {re['Released']}")
                        st.write(re['Plot'])
                        st.text(f"Rating: {re['imdbRating']}")
                        st.progress(float(re['imdbRating']) / 10)
                except:
                    st.error("No movie with that title found in the API Database YET -- add it to our database using the Movie Search feature!")


choices = ['Plot', 'Director', 'Actors', 'Genre', 'Movie Success']
choices = st.selectbox("Choose which type of movie grouping for recommendation: ", choices)

def category():
    st.header("Movie Recommendation System")

    if choices == 'Plot':
        by_plot()
    if choices == 'Director':
        by_director()
    if choices == 'Actors':
        by_actor()
    if choices == 'Genre':
        st.write("Coming soon")
    if choices == 'Movie Success':
        st.write("Coming soon")

category()