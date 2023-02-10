import streamlit as st
from working_files import data_loading
from working_files import etl
from working_files import regressions
from working_files import machine_learning
from working_files import movie_enthusiasts_form

st.title("Welcome to the Movie Analysis Experience :)")

def main():
    mode = st.radio("Pick an experience", ('Movie Industry Professional', 'Professional Movie Enthusiast'))
    
    if mode == 'Movie Industry Professional':
        options = ['Home','Machine Learning','Regressions']
        choice = st.sidebar.selectbox("Choose an analysis option", options, key = "298")

        if choice == 'Home':
            st.write("")
        if choice == 'Regressions':
            regressions.regression()
        if choice == 'Machine Learning':
            machine_learning.main_dashboard()
    if mode == 'Professional Movie Enthusiast':
        st.write("Coming soon...")
        movie_enthusiasts_form.submit_form()

main()