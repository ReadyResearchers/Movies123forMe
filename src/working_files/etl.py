import pandas as pd
import streamlit as st
from src.working_files import data_loading


def clean_data():
    # st.subheader("Performing ETL functions for the imported movie data")

    opus = data_loading.load_data_opus(10000)
    netflix = data_loading.load_data_netflix(10000)
    disney = data_loading.load_data_disney(10000)
    hulu = data_loading.load_data_hulu(10000)
    prime = data_loading.load_data_prime(10000)

    # compile the list of columns to be dropped for analysis

    opus_drop = ['movie_name', 'creative_type', 'source', 'production_method']
    netflix_drop = ['title', 'date_added', 'duration']
    disney_drop = ['title', 'date_added', 'duration']
    hulu_drop = ['title', 'date_added', 'duration']
    prime_drop = ['title', 'date_added', 'duration']

    if opus.columns.any() in opus_drop:
        opus.drop(opus_drop, inplace=True, axis=1)
    if netflix.columns.any() in netflix_drop:
        netflix.drop(netflix_drop, inplace=True, axis=1)
    if disney.columns.any() in disney_drop:
        disney.drop(disney_drop, inplace=True, axis=1)
    if hulu.columns.any() in hulu_drop:
        hulu.drop(hulu_drop, inplace=True, axis=1)
    if prime.columns.any() in prime_drop:
        prime.drop(prime_drop, inplace=True, axis=1)

    # set the unique identifier as the index

    # opus = opus.set_index('movie_odid')
    # netflix = netflix.set_index('show_id')
    # disney = disney.set_index('show_id')
    # hulu = hulu.set_index('show_id')
    # prime = prime.set_index('show_id')
    # st.write(opus.head())
    # st.write(netflix.head())
    # st.write(disney.head())
    # st.write(hulu.head())
    # st.write(prime.head())

    # testing how to locate using the new index

    # st.write(opus.loc[8220100])
    # st.write(opus.iloc[0])

    # drop nan values

    opus = opus.dropna()
    prime = prime.dropna()
    netflix = netflix.dropna()
    disney = disney.dropna()
    hulu = hulu.dropna()

    # changing dtype of columns

    opus['production_year'] = pd.to_numeric(opus['production_year'])
    opus['production_budget'] = pd.to_numeric(opus['production_budget'])
    opus['domestic_box_office'] = pd.to_numeric(opus['domestic_box_office'])
    opus['international_box_office'] = pd.to_numeric(opus['international_box_office'])
    opus['running_time'] = pd.to_numeric(opus['running_time'])
    opus['sequel'] = (opus['sequel']).astype(int)

    #opus['production_year'].dtype
    # opus['production_budget'].dtype
    # opus['domestic_box_office'].dtype
    # opus['international_box_office'].dtype
    # opus['running_time'].dtype

    netflix['release_year'] = pd.to_numeric(netflix['release_year'])
    disney['release_year'] = pd.to_numeric(disney['release_year'])
    prime['release_year'] = pd.to_numeric(prime['release_year'])
    hulu['release_year'] = pd.to_numeric(hulu['release_year'])

    netflix["date_added"] = pd.to_datetime(netflix['date_added'])
    disney["date_added"] = pd.to_datetime(disney['date_added'])
    prime["date_added"] = pd.to_datetime(prime['date_added'])
    hulu["date_added"] = pd.to_datetime(hulu['date_added'])

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
    
    
    # creating dummy variables for str columns
    opus = pd.get_dummies(opus, columns=['genre'])
    opus['rating'] = opus['rating'].map({'G':0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4, 'Not Rated': 5})
    netflix['rating'] = netflix['rating'].map({'TV-Y': 0, 'TV-Y7': 1, 'TV-Y7-FV': 2,
                        'G': 3, 'TV-G': 4, 'PG': 5, 'TV-PG': 6, 'PG-13': 7, 'TV-14': 8,
                        'R': 9, 'TV-MA': 10, 'NC-17': 11, 'NR': 12, 'UR': 13})
    prime['rating'] = prime['rating'].map({'TV-Y': 0, 'TV-Y7': 1, 'TV-Y7-FV': 2,
                        'G': 3, 'TV-G': 4, 'PG': 5, 'TV-PG': 6, 'PG-13': 7, 'TV-14': 8,
                        'R': 9, 'TV-MA': 10, 'NC-17': 11, 'NR': 12, 'UR': 13})
    disney['rating'] = disney['rating'].map({'TV-Y': 0, 'TV-Y7': 1, 'TV-Y7-FV': 2,
                        'G': 3, 'TV-G': 4, 'PG': 5, 'TV-PG': 6, 'PG-13': 7, 'TV-14': 8,
                        'R': 9, 'TV-MA': 10, 'NC-17': 11, 'NR': 12, 'UR': 13})

    # st.download_button(
    #    label="Download",
     #   data=opus.to_csv().encode("utf-8"),
      #  file_name='opus-genre.csv',
       # mime='text/csv',
        #)

    st.write(netflix.head())
    return opus, netflix, prime, disney, hulu

clean_data()