import streamlit as st
from working_files import data_loading

st.title("Welcome to the Movie Analysis Experience :)")

def main():
    options = ['Home', 'Data Loading', ]
    choice = st.sidebar.selectbox("Choose an analysis option", options, key = "298")

    if choice == 'Home':
        st.write("")
    if choice == 'Data Loading':
        data_loading.display()

main()