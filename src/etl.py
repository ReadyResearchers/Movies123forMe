import pandas as pd
import streamlit as st
import numpy as np
import data_loading


def clean_data():
    opus = data_loading.load_data_opus(10000)
    netflix = data_loading.load_data_netflix(10000)
    disney = data_loading.load_data_disney(10000)
    hulu = data_loading.load_data_hulu(10000)
    prime = data_loading.load_data_prime(10000)

    # compile the list of columns to be dropped for analysis

    opus_drop = ['movie_name', 'creative_type', 'source', 'production_method']
    netflix_drop = ['type', 'title', 'date_added', 'duration', 'description']
    disney_drop = ['type', 'title', 'date_added', 'duration', 'description']
    hulu_drop = ['type', 'title', 'date_added', 'duration', 'description']
    prime_drop = ['type', 'title', 'date_added', 'duration', 'description']


    # opus = opus.drop(opus_drop, inplace=True, axis=1)
    # netflix = netflix.drop(netflix_drop, inplace=True, axis=1)
    # disney = disney.drop(disney_drop, inplace=True, axis=1)
    # hulu = hulu.drop(hulu_drop, inplace=True, axis=1)
    # prime = prime.drop(prime_drop, inplace=True, axis=1)

    # set the unique identifier as the index

    opus = opus.set_index('movie_odid')
    netflix = netflix.set_index('show_id')
    disney = disney.set_index('show_id')
    hulu = hulu.set_index('show_id')
    prime = prime.set_index('show_id')
    # st.write(opus.head())
    # st.write(netflix.head())
    # st.write(disney.head())
    # st.write(hulu.head())
    # st.write(prime.head())

    # testing how to locate using the new index

    # st.write(opus.loc[8220100])
    # st.write(opus.iloc[0])

    # changing dtype of columns

    opus['production_year'] = pd.to_numeric(opus['production_year'])
    opus['production_budget'] = pd.to_numeric(opus['production_budget'])
    opus['domestic_box_office'] = pd.to_numeric(opus['domestic_box_office'])
    opus['international_box_office'] = pd.to_numeric(opus['international_box_office'])
    opus['running_time'] = pd.to_numeric(opus['running_time'])

    #opus['production_year'].dtype
    # opus['production_budget'].dtype
    # opus['domestic_box_office'].dtype
    # opus['international_box_office'].dtype
    # opus['running_time'].dtype

    netflix['release_year'] = pd.to_numeric(netflix['release_year'])
    disney['release_year'] = pd.to_numeric(disney['release_year'])
    prime['release_year'] = pd.to_numeric(prime['release_year'])
    hulu['release_year'] = pd.to_numeric(hulu['release_year'])

    # drop nan values

    opus = opus.dropna()
    prime = prime.dropna()
    netflix = netflix.dropna()
    disney = disney.dropna()
    hulu = hulu.dropna()
    # opus
    # prime
    # netflix
    # disney
    # hulu

    # splitting up rows that have more than one element

    netflix = netflix.assign(director=netflix['director'].str.split(",")).explode('director')
    netflix = netflix.assign(cast=netflix['cast'].str.split(",")).explode('cast')
    netflix = netflix.assign(country=netflix['country'].str.split(",")).explode('country')
    netflix = netflix.assign(listed_in=netflix['listed_in'].str.split(",")).explode('listed_in')


    prime = prime.assign(director=prime['director'].str.split(",")).explode('director')
    prime = prime.assign(cast=prime['cast'].str.split(",")).explode('cast')
    prime = prime.assign(country=prime['country'].str.split(",")).explode('country')
    prime = prime.assign(listed_in=prime['listed_in'].str.split(",")).explode('listed_in')


    disney = disney.assign(director=disney['director'].str.split(",")).explode('director')
    disney = disney.assign(cast=disney['cast'].str.split(",")).explode('cast')
    disney = disney.assign(country=disney['country'].str.split(",")).explode('country')
    disney = disney.assign(listed_in=disney['listed_in'].str.split(",")).explode('listed_in')
    return opus, netflix, prime, disney, hulu

def transform_data():
    opus = clean_data()[0]
    netflix = clean_data()[1]
    prime = clean_data()[2]
    disney = clean_data()[3]
    hulu = clean_data()[4]

    # calculating the mean
    opus['mean_budget'] = opus.groupby('genre')['production_budget'].transform('mean')
    opus['mean_budget']
    return opus, netflix, prime, disney, hulu
    

"""Opus Data"""
st.write(transform_data()[0])
"""Netflix Data"""
st.write(transform_data()[1])
"""Amazon Prime Data"""
st.write(transform_data()[2])
"""Disney+ Data"""
st.write(transform_data()[3])
"""Hulu Data"""
st.write(transform_data()[4])