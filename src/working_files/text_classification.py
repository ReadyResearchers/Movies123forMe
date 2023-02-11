import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
from src.working_files import data_loading

## importing necessary files
duplicates = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\movie_search.csv'
inFile = open('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\movie_search.csv', 'r')
outFile = open('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\movie_clean.csv', 'w')

# remove any \n characters in file
dups = []
for line in inFile:
    if line in dups:
        continue
    else:
        outFile.write(line)
        dups.append(line)

outFile.close()
inFile.close()

data = pd.read_csv('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\movie_clean.csv')
print(data.head())

# dropping an nan values
for i in data['BoxOffice']:
    if i is not None:
        data['BoxOffice'] = data['BoxOffice'].apply(str).str.replace("$", "")
        data['BoxOffice'] = data['BoxOffice'].apply(str).str.replace(',', '')

for i in data['DVD']:
    if i is not None:
        data['DVD'] = data['DVD'].str.replace("$", "")
        data['DVD'] = data['DVD'].str.replace(',', '')

print(data['BoxOffice'])
print(data['DVD'])


# creating movie success column based on mean production budget of opus data
if data['BoxOffice'].astype(float).any() >= 55507312.604108825:
    data['movie_success'] = 1
else:
    data['movie_success'] = 0

data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
'Production','Website','Response', 'movie_success']

data = data.drop_duplicates(subset = ['Title'], keep='first')

# keeping the same columns as netflix model
movies = data[['Type', 'Director', 'Actors', 'Year', 'Rated', 'Country', 'Plot', 'Genre', 'movie_success']]
st.write(movies.head())

# checking for duplicate data
movies.duplicated().sum()

# changing all strings in columns to lowercase
movies = movies.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
#print(movies.head())

# removing stop words
cv = CountVectorizer(max_features=5000, stop_words='english')
y = movies.drop('movie_success', axis=1, inplace=True)
vectors = cv.fit_transform(movies['Plot']).toarray()
#print(vectors)

cv.get_feature_names_out()

## get rid of similar words
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['Plot'] = movies['Plot'].apply(stem)

#print(movies['Plot'])

## finding the 'distance' between every movie for movie
cosine_similarity(vectors)
similarity = cosine_similarity(vectors).shape
#print(similarity)
def recommend(movie):
    movie_index = movies[movies['Type'] == 'movie'].index[0]

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    # fetching movies from indeces
    for i in movies_list:
        print(movies.iloc[i[0]].Title)

title = st.text_input("Type the title of a Movie for recommendations:")


#recommend(title)
recommend('Avatar')