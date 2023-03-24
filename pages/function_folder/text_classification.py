import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from pages import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


# remove any \n characters in file
# inFile = open('movie_search.csv', 'r')
# dups = []
# for line in inFile:
#     if line in dups:
#         continue
#     else:
#         outFile.write(line)
#         dups.append(line)
# outFile.close()
# inFile.close()
## importing necessary files
chunks = pd.read_csv('movie_clean.csv', chunksize=100)
data = pd.concat(chunks)

# creating column names
data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
                    'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
                    'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
                    'Production','Website','Response', 'movie_success','earnings']

data = data.drop_duplicates(subset = ['Title'], keep='first').reset_index()


def by_all():
    # doing data preprocessing
    movies = data[['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated']]

    #st.write(movies.isnull().sum())

    movies.dropna(inplace=True)

    movies = movies.drop_duplicates().reset_index()

    movies['tags'] = movies['Genre'] + " " +  movies['Plot'] + " " + movies['Actors'] + " " + movies['Director'] + " " + movies['Writer'] + " " + movies['Rated']

    new_df = movies[['imdbID', 'Title', 'tags']]
    # checking for duplicate data

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    ps = PorterStemmer()

    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    new_df['tags'] = new_df['tags'].apply(stem)
    similarity = cosine_similarity(vectors)
    indices = pd.Series(new_df.index, index=new_df['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=similarity):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return pd.DataFrame(new_df[['Title', 'tags']].iloc[movie_indices])

    movie_list = new_df['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list, key=movie_list)

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        a = recommended_movie_names.values[0][0]
        b = recommended_movie_names.values[1][0]
        c = recommended_movie_names.values[2][0]
        d = recommended_movie_names.values[3][0]
        if a:
            url = f'http://www.omdbapi.com/?t={a}&apikey=a98f1e4b'
            re = requests.get(url)
            re = re.json()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(re['Poster'])
            with col2:
                st.subheader(re['Title'])
                st.caption(f"Genre: {re['Genre']} | Year: {re['Year']} | Rated: {re['Rated']} | Released: {re['Released']}")
                st.write(re['Plot'])
                st.text(f"Rating: {re['imdbRating']}")
                st.progress(float(re['imdbRating']) / 10)
        if b:
            url = f'http://www.omdbapi.com/?t={b}&apikey=a98f1e4b'
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
        if c:
            url = f'http://www.omdbapi.com/?t={c}&apikey=a98f1e4b'
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
        if d:
            url = f'http://www.omdbapi.com/?t={d}&apikey=a98f1e4b'
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


def by_plot():
    # checking for duplicate data
    movies = ['Title', 'Plot']

    movie_data = data[movies].copy()

    tfidf = TfidfVectorizer(stop_words='english')

    movie_data['Plot'] = movie_data['Plot'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['Plot'])

    cosine_sim = cosine_similarity(tfidf_matrix)

    indices = pd.Series(movie_data.index, index=movie_data['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return pd.DataFrame(movie_data[['Title', 'Plot']].iloc[movie_indices])

    movie_data = movie_data.drop_duplicates(subset = ['Title'], keep='first')
    movie_list = movie_data['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list, key=movie_list)

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        a = recommended_movie_names.values[0][0]
        b = recommended_movie_names.values[1][0]
        c = recommended_movie_names.values[2][0]
        d = recommended_movie_names.values[3][0]
        if a:
            url = f'http://www.omdbapi.com/?t={a}&apikey=a98f1e4b'
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
        if b:
            url = f'http://www.omdbapi.com/?t={b}&apikey=a98f1e4b'
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
        if c:
            url = f'http://www.omdbapi.com/?t={c}&apikey=a98f1e4b'
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
        if d:
            url = f'http://www.omdbapi.com/?t={d}&apikey=a98f1e4b'
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

    cosine_sim = cosine_similarity(tfidf_matrix)

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
        a = recommended_movie_names.values[0][0]
        b = recommended_movie_names.values[1][0]
        c = recommended_movie_names.values[2][0]
        d = recommended_movie_names.values[3][0]
        if a:
            url = f'http://www.omdbapi.com/?t={a}&apikey=a98f1e4b'
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
        if b:
            url = f'http://www.omdbapi.com/?t={b}&apikey=a98f1e4b'
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
        if c:
            url = f'http://www.omdbapi.com/?t={c}&apikey=a98f1e4b'
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
        if d:
            url = f'http://www.omdbapi.com/?t={d}&apikey=a98f1e4b'
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

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

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
        a = recommended_movie_names.values[0][0]
        b = recommended_movie_names.values[1][0]
        c = recommended_movie_names.values[2][0]
        d = recommended_movie_names.values[3][0]
        if a:
            url = f'http://www.omdbapi.com/?t={a}&apikey=a98f1e4b'
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
        if b:
            url = f'http://www.omdbapi.com/?t={b}&apikey=a98f1e4b'
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
        if c:
            url = f'http://www.omdbapi.com/?t={c}&apikey=a98f1e4b'
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
        if d:
            url = f'http://www.omdbapi.com/?t={d}&apikey=a98f1e4b'
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


def by_genre():
    movies = ['Title', 'Genre']
    movie_data = data[movies].copy()

    tfidf = TfidfVectorizer(stop_words='english')

    movie_data['Genre'] = movie_data['Genre'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['Genre'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movie_data.index, index=movie_data['Title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movie_data[['Title', 'Genre']].iloc[movie_indices]


    movie_list = data['Title'].values

    selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list )

    if st.button('Show Recommendation'):
        recommended_movie_names = get_recommendations(selected_movie)
        a = recommended_movie_names.values[0][0]
        b = recommended_movie_names.values[1][0]
        c = recommended_movie_names.values[2][0]
        d = recommended_movie_names.values[3][0]
        if a:
            url = f'http://www.omdbapi.com/?t={a}&apikey=a98f1e4b'
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
        if b:
            url = f'http://www.omdbapi.com/?t={b}&apikey=a98f1e4b'
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
        if c:
            url = f'http://www.omdbapi.com/?t={c}&apikey=a98f1e4b'
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
        if d:
            url = f'http://www.omdbapi.com/?t={d}&apikey=a98f1e4b'
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


def category():
    st.header("Movie Recommendation System")
    choose = ['Plot', 'Director', 'Actors', 'Genre', 'All']
    choices = st.selectbox("Choose which type of movie grouping for recommendation: ", choose)

    if choices == 'Plot':
        by_plot()
    if choices == 'Director':
        by_director()
    if choices == 'Actors':
        by_actor()
    if choices == 'Genre':
        by_genre()
    if choices == 'All':
        by_all()

category()