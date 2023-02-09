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
        options = ['Home', 'Data Loading', 'ETL', 'Machine Learning',
        'Linear Regressions']
        choice = st.sidebar.selectbox("Choose an analysis option", options, key = "298")

        if choice == 'Home':
            st.write("")
        if choice == 'Data Loading':
            data_loading.display()
        if choice == 'ETL':
            st.write("------------------------")
            st.subheader("Summary of process:")
            st.write("- Dropped non-essential columns") 
            st.write("- Set the unique identifier as the index")
            st.write("- Drop nan values")
            st.write("- Change dtype of columns")
            st.write("- split up rows with more than one essential element")
            st.write("- create dummy variables for essential string columns")
            st.write("-------------------------")
            st.write("Cleaned Opus Data:", etl.clean_data()[0])
            st.write("-------------------------")
            st.write("Cleaned Netflix Data:", etl.clean_data()[1])
            st.write("-------------------------")
            st.write("Cleaned Prime Data:", etl.clean_data()[2])
            st.write("-------------------------")
            st.write("Cleaned Disney+ Data:", etl.clean_data()[3])
            st.write("-------------------------")
            st.write("Cleaned Hulu Data:", etl.clean_data()[4])
        if choice == 'Linear Regressions':
            regressions.regression()
        if choice == 'Machine Learning':
            machine_learning.machine_learning()
    if mode == 'Professional Movie Enthusiast':
        st.write("Coming soon...")
        movie_enthusiasts_form.submit_form()

main()