"""Main Method for the Interactive Movie Recommendation app."""

import streamlit as st # pylint: disable=E0401, C0413, C0411
import pandas as pd # pylint: disable=E0401, C0413, C0411
# from pages.function_folder import merge
from pages.function_folder import text_classification # pylint: disable=E0401, C0413, C0411
from pages import G_machine_learning # pylint: disable=E0401, C0413, C0411, C0412
from pages.function_folder import F_machine_texting

import numpy as np # pylint: disable=E0401, C0413, C0411
import joblib # pylint: disable=E0401, C0413, C0411
import requests # pylint: disable=E0401, C0413, C0411
import json # pylint: disable=E0401, C0413, C0411
import subprocess # pylint: disable=C0411

st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Main Page 🎈")

data = pd.read_csv("merged_data.csv")

@st.cache_data
def predict(data): # pylint: disable=W0621
    """Initial set up function to import the machine learning models."""
    logreg_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/logreg_model.sav"
    lr_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/lr_model.sav"
    rf_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/rf_model.sav"
    et_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/et_model.sav"
    dtc_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/dtc_model.sav"
    svm_filename = "C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/svm_model.sav"
    logreg = joblib.load(logreg_filename)
    lr = joblib.load(lr_filename) # pylint: disable=C0103
    rf = joblib.load(rf_filename) # pylint: disable=C0103
    et = joblib.load(et_filename) # pylint: disable=C0103
    dtc = joblib.load(dtc_filename) # pylint: disable=C0103
    svm = joblib.load(svm_filename) # pylint: disable=C0103
    return logreg.predict(data), lr.predict(data), rf.predict(data), et.predict(data), dtc.predict(data), svm.predict(data) # pylint: disable=C0301

def submit_form(): # pylint: disable=R0914, R0912, R0915
    """Function to create the movie prediction form."""
    # first section
    prod_budget = st.slider('What kind of production budget do you want for your movie?', min_value=0, max_value=525000000, value=1000000, step=100000) # pylint: disable=C0301

    rating_choices = {0: 'G', 1: 'PG', 2: 'PG-13', 3: 'R', 4: 'NC-17', 5: 'Not Rated'}
    rating = st.selectbox("Choose which rating(s) for the movie: ", list(rating_choices.keys()), format_func=lambda x: rating_choices[x]) # pylint: disable=C0301

    # second section
    genre_choices = {0: 'No', 1: 'Yes'}
    genre_Action = st.selectbox("Do you want your movie to be an Action movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Comedy = st.selectbox("Do you want your movie to be a Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Drama = st.selectbox("Do you want your movie to be a Drama movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Adventure = st.selectbox("Do you want your movie to be an Adventure movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_BlackComedy = st.selectbox("Do you want your movie to be a Black Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Concert = st.selectbox("Do you want your movie to be a Concert/Performance movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Documentary = st.selectbox("Do you want your movie to be a Documentary movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Horror = st.selectbox("Do you want your movie to be a Horror movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Musical = st.selectbox("Do you want your movie to be a Musical movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_RomanticComedy = st.selectbox("Do you want your movie to be a Romantic Comedy movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Thriller = st.selectbox("Do you want your movie to be a Thriller/Suspense movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103
    genre_Western = st.selectbox("Do you want your movie to be a Western movie?", list(genre_choices.keys()), format_func=lambda x: genre_choices[x]) # pylint: disable=C0301, C0103


    sequel_choices = {0: 'No', 1: 'Yes'}
    sequel = st.selectbox("Do you want your movie to be a sequel?", list(sequel_choices.keys()), format_func=lambda x: sequel_choices[x]) # pylint: disable=C0301, C0103
    # submit button
    clickSubmit = st.button('Predict success of your movie!') # pylint: disable=C0103

    if clickSubmit:
        df = np.array([[prod_budget, rating, sequel, genre_Action, # pylint: disable=C0103
        genre_Comedy, genre_Drama, genre_Adventure,
        genre_BlackComedy, genre_Concert, genre_Documentary,
        genre_Horror, genre_Musical, genre_RomanticComedy,
        genre_Thriller, genre_Western]], dtype=int)

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

        score_logreg = G_machine_learning.logreg()[1]
        score_lr = G_machine_learning.lr()[1]
        score_rf = G_machine_learning.rf()[1]
        score_et = G_machine_learning.et()[1]
        score_dtc = G_machine_learning.dtc()[1]
        score_svm = G_machine_learning.svm()[1]

        st.subheader("Movie Success Results:")
        for i in result_logreg:
            if i[0].round() == 0:
                st.write(f"The Logistic Regression Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_logreg) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Logistic Regression Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_logreg) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")
        for i in result_lr:
            if i[0].round() == 0:
                st.write(f"The Linear Regression Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_lr) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Linear Regression Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_lr) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")
        for i in result_rf:
            if i[0].round() == 0:
                st.write(f"The Random Forest Regressor Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_rf) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Random Forest Regressor Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_rf) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")
        for i in result_et:
            if i[0].round() == 0:
                st.write(f"The Extra Tree Regressor Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_et) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Extra Tree Regressor Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_et) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")
        for i in result_dtc:
            if i[0].round() == 0:
                st.write(f"The Decision Tree Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_dtc) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Decision Tree Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_dtc) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")
        for i in result_svm:
            if i[0].round() == 0:
                st.write(f"The Support Vector Machine Learning model predicted your movie would NOT BE A SUCCESS with a { (100 * score_svm) }% confidence! :(") # pylint: disable=C0301
            elif i[0].round() == 1:
                st.write(f"The Support Vector Machine Learning model predicted your movie would BE A SUCCESS with a { (100 * score_svm) }% confidence! :)") # pylint: disable=C0301
            else:
                st.write("Please try again later.")

def search_movies():
    """Function to search for the movies using the OMDB API."""
    title = st.text_input("Type the title of the desired Movie/TV Show:")
    if title:
        url = f'http://www.omdbapi.com/?t={title}&apikey=a98f1e4b'
        req = requests.get(url)
        req = req.json()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(req['Poster'])
        with col2:
            st.subheader(req['Title'])
            st.caption(f"Genre: {req['Genre']} | Year: {req['Year']} | Rated: {req['Rated']} | Released: {req['Released']}") # pylint: disable=C0301
            st.write(req['Plot'])
            st.text(f"Rating: {req['imdbRating']}")
            st.progress(float(req['imdbRating']) / 10)
        with open('response.json', 'w', encoding='utf-8') as json_file:
            json.dump(req, json_file)
        with open('response.json', 'r', encoding='utf-8') as file:
            json.loads(file.read())
        with open('response.json', encoding='utf-8') as inputfile:
            df = pd.read_json(inputfile) # pylint: disable=C0103
        with open('movie_search.csv', 'a', encoding='utf-8') as inFile: # pylint: disable=C0103
            inFile.write(df.to_csv(header = False, index=False))
        with open('movie_clean.csv', 'a', newline='', encoding='utf-8') as outFile: # pylint: disable=C0103
            tail = df.tail(1)
            outFile.write(tail.to_csv(header=False, index=False, mode='a' ))
            df = pd.read_csv('movie_clean.csv') # pylint: disable=C0103
            df['earnings'] = df["BoxOffice"].replace(np.nan,"0")
            df['earnings'] = df['earnings'].str.replace(r'[^\w\s]+', "", regex=True)
            df = df[df['earnings'].str.contains("TRUE") == False] # pylint: disable=C0103, C0121
            df['movie_success'] = np.where(
                df['earnings'].astype(int) > 55507312, 1, 0)
            outFile.close()
        subprocess.run("commit.sh", check=True) # pylint: disable=W1510

    if len(title) == 0:
        url = 'http://www.omdbapi.com/?t=clueless&apikey=a98f1e4b'
        req = requests.get(url)
        req = req.json()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(req['Poster'])
        with col2:
            st.subheader(req['Title'])
            st.caption(f"Genre: {req['Genre']} | Year: {req['Year']} | Rated: {req['Rated']} | Released: {req['Released']}") # pylint: disable=C0301
            st.write(req['Plot'])
            st.text(f"Rating: {req['imdbRating']}")
            st.progress(float(req['imdbRating']) / 10)


def interface():
    """Main interface for the app."""
    choices = ['Movie Search', 'Predict Success of a Movie!',
    'What Movie Should You Watch?', 'Predict Movie Success with Text']
    success = st.sidebar.selectbox("Select a Movie Experience :)", choices)
    if success == 'Movie Search':
        search_movies()
    elif success == 'Predict Success of a Movie!':
        submit_form()
    elif success == 'What Movie Should You Watch?':
        text_classification.category()
    elif success == 'Predict Movie Success with Text':
        F_machine_texting.predict_text()

interface()
