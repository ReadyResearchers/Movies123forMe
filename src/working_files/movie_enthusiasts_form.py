import streamlit as st
import pandas as pd
from src.working_files import etl
from src.working_files import machine_learning

import numpy as np
import joblib

def predict(data):
    filename = "Movies123forMe/finalized_movie_model.sav"
    logreg = joblib.load(filename)
    return logreg.predict(data)

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
        result = predict(df)
        result = result.reshape(1, -1)
        st.subheader("Movie Success Results:")
        for i in result:
            if i[0] == 0:
                st.write("The Logistic Regression Machine Learning model predicted your movie would NOT BE A SUCCESS! :(")
            elif i[0] == 1:
                st.write("The Logistic Regression Machine Learning model predicted your movie would BE A SUCCESS! :)")
            else:
                st.write("Please try again later.")

submit_form()
