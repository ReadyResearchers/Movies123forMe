import streamlit as st
import pandas as pd
from src.working_files import etl
from sklearn.model_selection import train_test_split
from src.working_files import machine_learning

import tensorflow as tf

import numpy as np
import joblib


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

def interface():
    choices = ['Home', 'Predict Success of a Movie!', 'List of Movies', 'What Movie Should You Watch?']
    success = st.sidebar.selectbox("Select a Movie Experience :)", choices)
    if success == 'Home':
        st.subheader("Please choose one of the options on the sidebar to get started!")
    if success == 'Predict Success of a Movie!':
        submit_form()
    elif success == 'List of Movies':
        st.write("Coming soon...")
    elif success == 'What Movie Should You Watch?':
        st.write("Coming soon...")

interface()
