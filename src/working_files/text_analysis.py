import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import streamlit as st

nltk.download('stopwords')

import pickle
from nltk.corpus import stopwords
from src.working_files import etl
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

netflix = load_files(r"C:/Users/solis/OneDrive/Documents/comp/Movies123forMe/src/movie_data/netflix/archive/")

st.write(netflix)
nX, nY = netflix.data, netflix.target

nDocs = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(nX)):
    # remove all special characters
    nDoc = re.sub(r'W', ' ', str(nX[sen]))
    # remove all single characters
    nDoc = re.sub(r'\s+[a-zA-Z]\s+', ' ', nDoc)
    # remove single characters from the start
    nDoc = re.sub(r'\^[a-zA-Z]\s+', ' ', nDoc)
    # substitute multiple spcaes with single space
    nDoc = re.sub(r'\s+', ' ', nDoc, flags=re.I)
    # Removing prefixed 'b'
    nDoc = re.sub(r'^b\s+', '', nDoc)
    # converting to Lowercase
    nDoc = nDoc.lower()
    # lemmatization
    nDoc = nDoc.split()
    nDoc = [stemmer.lemmatize(word) for word in nDoc]
    nDoc = ' '.join(nDoc)
    nDocs.append(nDoc)

st.write(nDocs)

# create bag of words to create numerical features
vectorizer = CountVectorizer(max_features=1500, min_df = 5, max_df = 0.7, stop_words = stopwords.words('english'))
nX = vectorizer.fit_transform(nDocs).toarray()

# finding the term frequency of a word
tfidfconverter = TfidfTransformer()
nX = tfidfconverter.fit_transform(nX).toarray()

# training and testing sets
nX_train, nX_test, nY_train, nY_test = train_test_split(nX, nY, test_size=0.2, random_state = 0)

#training text classification model and predicting sentiment
nClassifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)
nClassifier.fit(nX_train, nY_train)
nY_pred = nClassifier.predict(nX_test)

# evaluating the model
st.write(confusion_matrix(nY_test, nY_pred))
st.write(classification_report(nY_test, nY_pred))
st.write(accuracy_score(nY_test, nY_pred))