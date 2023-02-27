import streamlit as st
import pandas as pd
import numpy as np
from src.pages.function_folder import B_etl 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer


st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Subpage 4 🎈")

st.write("---")
## importing necessary files
duplicates = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_search.csv'

train_data = pd.read_csv('C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\movie_clean.csv',encoding='ISO-8859-1')

train_data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
'Production','Website','Response', 'movie_success','earnings']

train_data = train_data.drop_duplicates(subset = ['Title'], keep='first').reset_index()
movies = train_data[['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated', 'movie_success']]
#st.write(movies.isnull().sum())
movies.dropna(inplace=True)
movies = movies.drop_duplicates().reset_index()
movies['tags'] = movies['Genre'] + " " +  movies['Plot'] + " " + movies['Actors'] + " " + movies['Director'] + " " + movies['Writer'] + " " + movies['Rated']
t_data = movies[['Actors', 'movie_success']]

#stopword removal and lemmatization
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
# st.write(t_data.head())

train_X_non = t_data['Actors']   # '0' refers to the review text
train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = t_data['Actors']
test_y = t_data['movie_success']
train_X=[]
test_X=[]

#text pre processing
for i in range(0, len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_X.append(review)

#text pre processing
for i in range(0, len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)

# st.write(train_X[10])

#tf idf
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)
# st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

#transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)
# st.write("n_samples: %d, n_features: %d" % X_test_tf.shape)

#naive bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)
#predicted y
y_pred = naive_bayes_classifier.predict(X_test_tf)

# st.write(metrics.classification_report(test_y, y_pred, target_names=['movie_success']))
# st.write("Confusion matrix:")
# st.write(metrics.confusion_matrix(test_y, y_pred))
test_data = st.text_input("Input a movie actor/actresses for testing:")

review = re.sub('[^a-zA-Z]', ' ', test_data)
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
test_processed =[ ' '.join(review)]

# st.write(test_processed)
test_input = tf_idf.transform(test_processed)
#0= bad review
#1= good review
res=naive_bayes_classifier.predict(test_input)[0]
if res==1:
    st.write("Movie Success")
elif res==0:
    st.write("Not A Movie Success")