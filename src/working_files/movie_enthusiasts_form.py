import streamlit as st
import pandas as pd
from src.working_files import etl
from src.working_files import data_loading
from sklearn.model_selection import train_test_split
from src.working_files import machine_learning

import tensorflow as tf

import numpy as np
import joblib
import requests
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
from streamlit import session_state as session
from functools import reduce
import json
import csv
import pathlib



def predict(data):
    logreg_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/logreg_model.sav"
    lr_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/lr_model.sav"
    rf_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/rf_model.sav"
    et_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/et_model.sav"
    dtc_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/dtc_model.sav"
    svm_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/svm_model.sav"
    logreg = joblib.load(logreg_filename)
    lr = joblib.load(lr_filename)
    rf = joblib.load(rf_filename)
    et = joblib.load(et_filename)
    dtc = joblib.load(dtc_filename)
    svm = joblib.load(svm_filename)
    return logreg.predict(data), lr.predict(data), rf.predict(data), et.predict(data), dtc.predict(data), svm.predict(data)

def submit_form():
    # first section
    prod_budget = st.slider('What kind of production budget do you want for your movie?', min_value=0, max_value=525000000, value=1000000, step=100000)

    rating_choices = {0: 'G', 1: 'PG', 2: 'PG-13', 3: 'R', 4: 'NC-17', 5: 'Not Rated'}
    rating = st.selectbox("Choose which rating(s) for the movie: ", list(rating_choices.keys()), format_func=lambda x: rating_choices[x])

    # second section
    genre_choices = {0: 'No', 1: 'Yes'}
    genre_Action = st.selectbox("Do you want your movie to be an Action movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Comedy = st.selectbox("Do you want your movie to be a Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Drama = st.selectbox("Do you want your movie to be a Drama movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Adventure = st.selectbox("Do you want your movie to be an Adventure movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_BlackComedy = st.selectbox("Do you want your movie to be a Black Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Concert = st.selectbox("Do you want your movie to be a Concert/Performance movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Documentary = st.selectbox("Do you want your movie to be a Documentary movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Horror = st.selectbox("Do you want your movie to be a Horror movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Musical = st.selectbox("Do you want your movie to be a Musical movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_RomanticComedy = st.selectbox("Do you want your movie to be a Romantic Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Thriller = st.selectbox("Do you want your movie to be a Thriller/Suspense movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])
    genre_Western = st.selectbox("Do you want your movie to be a Western movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])


    sequel_choices = {0: 'No', 1: 'Yes'}
    sequel = st.selectbox("Do you want your movie to be a sequel?", list(sequel_choices.keys()), format_func=lambda x: sequel_choices[x])
    # submit button
    clickSubmit = st.button('Predict success of your movie!')

    if clickSubmit:
        df = np.array([[prod_budget, rating, sequel, genre_Action, genre_Comedy, genre_Drama, genre_Adventure, genre_BlackComedy, genre_Concert, genre_Documentary,
        genre_Horror, genre_Musical, genre_RomanticComedy, genre_Thriller, genre_Western]], dtype=int)

        result_logreg = predict(df)[0]
        result_lr = predict(df)[1]
        result_rf = predict(df)[2]
        result_et = predict(df)[3]
        result_dtc = predict(df)[4]
        result_svm = predict(df)[5]

        result_logreg = result_logreg.reshape(1, -1)
        result_lr = result_lr.reshape(1, -1)
        result_rf = result_rf.reshape(1, -1)
        result_et = result_et.reshape(1, -1)
        result_dtc = result_dtc.reshape(1, -1)
        result_svm = result_svm.reshape(1, -1)

        score_logreg = machine_learning.logreg()[1]
        score_lr = machine_learning.lr()[1]
        score_rf = machine_learning.rf()[1]
        score_et = machine_learning.et()[1]
        score_dtc = machine_learning.dtc()[1]
        score_svm = machine_learning.svm()[1]

        st.subheader("Movie Success Results:")
        for i in result_logreg:
            if i[0].round() == 0:
                st.write(f"The Logistic Regression Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_logreg) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Logistic Regression Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_logreg) }% confidence! :)")
            else:
                st.write("Please try again later.")
        for i in result_lr:
            if i[0].round() == 0:
                st.write(f"The Linear Regression Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_lr) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Linear Regression Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_lr) }% confidence! :)")
            else:
                st.write("Please try again later.")
        for i in result_rf:
            if i[0].round() == 0:
                st.write(f"The Random Forest Regressor Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_rf) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Random Forest Regressor Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_rf) }% confidence! :)")
            else:
                st.write("Please try again later.")
        for i in result_et:
            if i[0].round() == 0:
                st.write(f"The Extra Tree Regressor Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_et) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Extra Tree Regressor Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_et) }% confidence! :)")
            else:
                st.write("Please try again later.")
        for i in result_dtc:
            if i[0].round() == 0:
                st.write(f"The Decision Tree Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_dtc) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Decision Tree Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_dtc) }% confidence! :)")
            else:
                st.write("Please try again later.")
        for i in result_svm:
            if i[0].round() == 0:
                st.write(f"The Support Vector Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_svm) }% confidence! :(")
            elif i[0].round() == 1:
                st.write(f"The Support Vector Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_svm) }% confidence! :)")
            else:
                st.write("Please try again later.")

def search_movies():
    title = st.text_input("Type the title of the desired Movie/TV Show:")
    def running():
        #creating a baseline list of movies in csv file for analysis
        netflix = data_loading.load_data_netflix(10)
        for _, row in netflix.iterrows():
            if row[1] == 'Movie':
                i = str(row[2])
                web = f'http://www.omdbapi.com/?t={i}&apikey=4482116e'
                res = requests.get(web)
                res = res.json()
                if res['Response'] == 'False':
                    continue
                try:
                    with open('response.json', 'w') as json_file:
                        json.dump(res, json_file)
                    with open('response.json', 'r') as file:
                        json.loads(file.read())
                    with open('response.json', encoding='utf-8') as inputfile:
                        df = pd.read_json(inputfile)
                    open('movie_search.csv', 'a').write(df.to_csv(header = False))
                except:
                    st.write("Try again tomorrow!")
            else:
                continue
    running()
    if title:
        url = f'http://www.omdbapi.com/?t={title}&apikey=4482116e'
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
            st.error("No movie with that title found in the API Database OR try again tomorrow!")
        
        with open('response.json', 'w') as json_file:
            json.dump(re, json_file)
        with open('response.json') as file:
            json.load(file)
        with open('response.json', encoding='utf-8') as inputfile:
            df = pd.read_json(inputfile)
        open('movie_search.csv', 'a').write(df.to_csv(header = False, index=False))


    if len(title) == 0:
        url = f'http://www.omdbapi.com/?t=clueless&apikey=4482116e'
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


def what_movie():
    netflix = etl.clean_data()[1]
    prime = etl.clean_data()[2]
    disney = etl.clean_data()[3]

    df = [netflix, prime, disney]
    df_merged = pd.concat(df)

    # tokenize the words for lemmatization and removing stopwords
    df_merged['description'] = df_merged['description'].apply(word_tokenize)
    df_merged['description'] = df_merged['description'].apply(
    lambda x:[word for word in x if word not in set(stopwords.words('english'))]
    )

    # joining the words after lemmatization and stopword removal
    df_merged['description'] = df_merged['description'].apply(lambda x: ' '.join(x))
        
    # making an object of TfidfVectorizer in which words contains only in 1 document and word repeated in 70% of documents are ignored.
    tfidf = TfidfVectorizer(min_df = 2, max_df = 0.7)


    # fitting the cleaned text in TfidfVectorizer
    X = tfidf.fit_transform(df_merged['description'])


    # making a suitable dataframe for calculating the cosine similarity and save it
    tfidf_df = pd.DataFrame(X.toarray(), columns = tfidf.get_feature_names())
    tfidf_df.index = df_merged['title']
    tfidf_df.to_csv("data/tfidf_data.csv")

    def recommend_table(list_of_movie_enjoyed, tfidf_data, movie_count=20):
        """
        function for recommending movies
        :param list_of_movie_enjoyed: list of movies
        :param tfidf_data: self-explanatory
        :param movie_count: no of movies to suggest
        :return: dataframe containing suggested movie
        """
        movie_enjoyed_df = tfidf_data.reindex(list_of_movie_enjoyed)
        user_prof = movie_enjoyed_df.mean()
        tfidf_subset_df = tfidf_data.drop(list_of_movie_enjoyed)
        similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).head(movie_count)

        return sorted_similarity_df
    
    @st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
    def load_data():
        """
        load and cache data
        :return: tfidf data
        """
        tfidf_data = pd.read_csv("data/tfidf_data.csv", index_col=0)
        return tfidf_data


    tfidf = load_data()

    with open("data/movie_list.pickle", "rb") as f:
        movies = pickle.load(f)
    
    dataframe = None

    st.title("""
    Netflix, Amazon Prime, Disney+, and Hulu Recommendation System
    This is an Content Based Recommender System made on implicit ratings :smile:.
    """)

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    session.options = st.multiselect(label="Select Movies", options=movies)

    st.text("")
    st.text("")

    session.slider_count = st.slider(label="movie_count", min_value=5, max_value=50)

    st.text("")
    st.text("")

    buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

    is_clicked = col1.button(label="Recommend")

    if is_clicked:
        dataframe = recommend_table(session.options, movie_count=session.slider_count, tfidf_data=tfidf)

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    if dataframe is not None:
        st.table(dataframe)


def interface():
    choices = ['Movie Search', 'Predict Success of a Movie!', 'What Movie Should You Watch?']
    success = st.sidebar.selectbox("Select a Movie Experience :)", choices)
    if success == 'Movie Search':
        st.subheader("Welcome to Movies123ForMe Movie Enthusiasts!")
        search_movies()
    if success == 'Predict Success of a Movie!':
        submit_form()
    elif success == 'What Movie Should You Watch?':
        what_movie()

interface()