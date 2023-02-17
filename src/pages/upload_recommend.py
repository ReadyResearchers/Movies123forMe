import streamlit as st
import pandas as pd


uploaded_file = st.file_uploader("Choose a file (CSV or XSL)")
if uploaded_file is not None:
    # reading file as a pandas csv file
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    except:
        st.write("Please upload a valid file to continue!")
    # inferring the schema of the data
    df = df.infer_objects()
    