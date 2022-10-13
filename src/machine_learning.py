import streamlit as st
import data_loading
import pandas as pd
import numpy as np
import joblib
import os

def load_model(model_file):
    model = joblib.load(open(os.path.join(model_file), "rb"))
    return model

data_classifier = load_model('src/movie_data/basic/movie_ratings.csv')

def plot_prediction(term):
    pred_prob_df = pd.DataFrame({'Probabilities':data_classifier.predict_prob_df([term])[0], 'Classes':data_classifier})
    c = alt.Chart(pred_prob_df).mark_bar().encode(
        x = 'Classes',
        y = 'Probabilities',
        color = 'Classes'
    )
    st.altair_chart(c)

def main():
    display = data_loading.display()
    with st.form(key='myform'):
        predict = st.text_area("Enter movie data here: ")
        submit = st.form_submit_button(label='predict')
    
    if submit:
        col1, col2 = st.columns(2)
        prediction = data_classifier.predict([predict])
        pred_proba = data_classifier.predict_proba([predict])
        probabilities = dict(zip(data_classifier.classes_,pred_proba[0]))

        # display results
        with col1:
            st.info("Original Data")
            st.text(predict)
            st.write(prediction)
            st.write(probabilities)

        # Display Plots
        with col2:
            st.info("Probability Plots")
            

if __name__ == '__main__':
    main()