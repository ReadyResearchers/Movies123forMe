import streamlit as st
from working_files import data_loading
from working_files import etl
from working_files import ml_model
from working_files import machine_learning

st.title("Welcome to the Movie Analysis Experience :)")

def main():
    options = ['Home', 'Data Loading', 'ETL', 'ML Model', 'Machine Learning']
    choice = st.sidebar.selectbox("Choose an analysis option", options, key = "298")

    if choice == 'Home':
        st.write("")
    if choice == 'Data Loading':
        data_loading.display()
    if choice == 'ETL':
        etl.clean_data()
    if choice == 'ML Model':
        ml_model.machine_model()
    if choice == 'Machine Learning':
        machine_learning.m_learning()

main()