import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from src.working_files import etl
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

netflix = etl.clean_data()[1]

netflix_cols = ['type', 'director', 'cast', 'release_year', 'rating', 'country','description', 'listed_in']
netflix = netflix.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
st.write(netflix.head())

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(netflix[netflix_cols])
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

clf = MultinomialNB().fit(X_train_tfidf, netflix.duration_season)

predicted = clf.predict(netflix[netflix_cols].values)
np.mean(predicted == netflix.duration_season)