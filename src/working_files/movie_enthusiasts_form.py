import streamlit as st
import pandas as pd
from src.working_files import etl
from src.working_files import machine_learning

import numpy as np
import joblib

def predict(data):
    filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/finalized_movie_model.sav"
    logreg = joblib.load(filename)
    return logreg.predict(data)

def submit_form():
    # first option
    rating_choices = {0: 'G', 1: 'PG', 2: 'PG-13', 3: 'R', 4: 'NC-17', 5: 'Not Rated'}
    rating = st.selectbox("Choose which rating(s) for the movie: ", list(rating_choices.keys()), format_func=lambda x: rating_choices[x])

    # second option
    genre_choices = {0: 'genre_Action', 1: 'genre_Adventure', 2: 'genre_Black Comedy', 3: 'genre_Comedy', 
    4: 'genre_Concert/Performance', 5: 'genre_Documentary', 6: 'genre_Drama', 7: 'genre_Horror', 8: 'genre_Musical',
    9: 'genre_Romantic Comedy', 10: 'genre_Thriller/Suspense', 11: 'genre_Western'}
    genre = st.selectbox("Choose at least 2 genre(s) for the movie", list(genre_choices.keys()), format_func=lambda x: genre_choices[x])

    # third option
    sequel_choices = {0: 'No', 1: 'Yes'}
    sequel = st.selectbox("Do you want your movie to be a sequel?", list(sequel_choices.keys()), format_func=lambda x: sequel_choices[x])

    # submit button
    clickSubmit = st.button('Predict success of your movie!')

    if clickSubmit:
        df = np.array([[rating_choices[rating], sequel_choices[sequel], genre_choices[genre]]], dtype=float)
        result = predict(df)
        result = result.reshape(1, -1)
        st.text(result[0])

submit_form()