import pandas as pd
import numpy as np
import streamlit as st
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from pages.function_folder import A_data_loading

movies = pd.read_csv('movie_clean.csv')


# doing data preprocessing
movies = movies[['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated']]

#st.write(movies.isnull().sum())

movies.dropna(inplace=True)

movies = movies.drop_duplicates()

movies['tags'] = movies['Genre'] + " " +  movies['Plot'] + " " + movies['Actors'] + " " + movies['Director'] + " " + movies['Writer'] + " " + movies['Rated']

new_df = movies[['imdbID', 'Title', 'tags']]

# using bag of words technique to calculate the 5000 most frequent words, 
# with stop words removed
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
#st.write(vectors)
#st.write(cv.get_feature_names_out())

# bag of words contains similar words in different forms
# to fix this we use stemming of the words
# removies commoner morphological and inflexional endings from words
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
st.write(new_df.head())
#st.write(new_df['tags'])

# finding similarities between movies
# the lesser the distance between the movies, the more similarities there are
similarity = cosine_similarity(vectors)
#st.write(similarity.shape)

# finding the similarity vector of the movie provided in the input
# in order to sort the movie, we index them using enumerate and sort them according to scores
def recommend(movie):

    #find the index of the movies
    movie_index = new_df[new_df['Title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    #to fetch movies from indeces
    for i in movies_list:
        print(new_df.iloc[i[0]].Title)

st.write(recommend('Megamind'))