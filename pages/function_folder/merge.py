import streamlit as st
import pandas as pd
import numpy as np
from pages.function_folder import A_data_loading

def merge_data():
    opus = A_data_loading.load_data_opus(10000)
    netflix = A_data_loading.load_data_netflix(10000)
    prime = A_data_loading.load_data_prime(10000)
    disney = A_data_loading.load_data_disney(10000)
    hulu = A_data_loading.load_data_hulu(10000)
    imdb = A_data_loading.load_data_imdb(10000)

    # renaming the column to title
    opus = opus.rename(columns = {'movie_name': 'title'})
    st.write(imdb.head())
    # creating dataframes with only the title to merge on
    data = opus[['title']]
    data1 = netflix[['title']]
    data2 = prime[['title']]
    data3 = disney[['title']]
    data4 = hulu[['title']]
    data5 = imdb[['primarytitle']]

    data5 = data5.rename(columns = {'primarytitle': 'title'})

    # shape of data for comparison
    # data.shape
    # data1.shape
    # data2.shape
    # data3.shape
    # data4.shape
    # data5.shape

    merged1 = pd.merge(data, data1, on =['title'], how="outer")
    merged2 = pd.merge(merged1, data2, on=['title'], how="outer")
    merged3 = pd.merge(merged2, data3, on=['title'], how="outer")
    merged4 = pd.merge(merged3, data4, on=['title'], how="outer")
    merged_data = pd.merge(merged4, data5, on=['title'], how="outer")
    # st.write(merged5.shape)
    merged_data = merged_data.sort_values('title').reset_index(drop=True)
    # st.download_button(
    #      label="Download",
    #      data=merged_data.to_csv().encode("utf-8"),
    #      file_name='merged_data.csv',
    #      mime='text/csv',
    #      )
    return merged_data


merge_data()