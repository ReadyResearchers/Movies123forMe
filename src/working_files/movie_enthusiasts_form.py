import streamlit as st
import pandas as pd
import joblib

def form():
    # first option
    st.write("MPAA Rating Section:")
    rating_choices = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated']
    rating = st.multiselect("Choose which rating(s) for the movie: ", rating_choices)

    # second option
    st.write("Genre Section:")
    genre_Action = st.selectbox("Do you want your movie to be an Action movie?", ['Yes', 'No'])
    genre_Adventure = st.selectbox("Do you want your movie to be an Adventure movie?", ['Yes', 'No'])
    genre_BlackComedy = st.selectbox("Do you want your movie to be a Black Comedy movie?", ['Yes', 'No'])
    genre_Comedy = st.selectbox("Do you want your movie to be a Comedy movie?", ['Yes', 'No'])
    genre_ConcertPerformance = st.selectbox("Do you want your movie to be a Concert/Performance movie?", ['Yes', 'No'])
    genre_Documentary = st.selectbox("Do you want your movie to be a Documentary movie?", ['Yes', 'No'])
    genre_Drama = st.selectbox("Do you want your movie to be a Drama movie?", ['Yes', 'No'])
    genre_Horror = st.selectbox("Do you want your movie to be a Horror movie?", ['Yes', 'No'])
    genre_Musical = st.selectbox("Do you want your movie to be a Musical movie?", ['Yes', 'No'])
    genre_RomanticComedy = st.selectbox("Do you want your movie to be a Romantic Comedy movie?", ['Yes', 'No'])
    genre_ThrillerSuspense = st.selectbox("Do you want your movie to be a Thriller/Suspense movie?", ['Yes', 'No'])
    genre_Western = st.selectbox("Do you want your movie to be a Western movie?", ['Yes', 'No'])


    # third option
    st.write("Sequel Section:")
    sequel_choices = ['Yes', 'No']
    sequel = st.selectbox("Do you want your movie to be a sequel?", sequel_choices)

    if st.button("Submit"):
        # unpickle model
        logreg = joblib.load("logreg.pkl")

        # store inputs in dataframe
        x = pd.DataFrame([[rating, sequel, genre_Action, genre_Comedy, genre_Drama, 
        genre_Adventure, genre_BlackComedy, genre_ConcertPerformance, genre_Documentary, genre_Horror,
        genre_Musical, genre_RomanticComedy, genre_ThrillerSuspense, genre_Western]])
        x = x.replace(['Yes', 'No'], [1, 0])

        # get predicition
        prediction = logreg.predict(x)[0]


        # output prediction
        st.text(f"This movie will be {prediction}")

form()