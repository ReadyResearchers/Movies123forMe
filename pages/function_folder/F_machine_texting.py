import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.corpus import stopwords as sw
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.util import ngrams
import nltk

from sklearn.feature_selection import chi2

train_data = pd.read_csv('movie_clean.csv')
st.write(train_data.head())
# train_data['earnings'] = train_data["BoxOffice"].replace(np.nan,"0")
# train_data['earnings'] = train_data['earnings'].str.replace(r'[^\w\s]+', "", regex=True)
# st.write(train_data['earnings'])
# train_data = train_data[train_data['earnings'].str.contains("TRUE") == False]
# st.write(train_data['earnings'].astype(float))

# train_data['movie_success'] = np.where(
#     train_data['earnings'] > 55507312, 1, 0)


train_data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
'Production','Website','Response', 'movie_success','earnings']

train_data = train_data.drop_duplicates(subset = ['Title'], keep='first').reset_index()
train_data = train_data[['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated', 'movie_success']]
# #st.write(movies.isnull().sum())
train_data = train_data.drop_duplicates().reset_index()
train_data['tags'] = train_data['Genre'] + " " +  train_data['Plot'] + " " + train_data['Actors'] + " " + train_data['Director'] + " " + train_data['Writer'] + " " + train_data['Rated']

# file_format = st.sidebar.radio('Select file format:', ('csv', 'excel'), key='file_format')
# dataset = st.sidebar.file_uploader(label = '')

def predict_text_example():
    t_data = train_data[['tags', 'movie_success']].dropna().reset_index()
    #stopword removal and lemmatization
    stopw = sw.words('english')
    lemmatizer = WordNetLemmatizer()
    # nltk.download('stopwords')
    # st.write(t_data.head())

    train_X_non = t_data['tags']   # '0' refers to the review text
    train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
    test_X_non = t_data['tags']
    test_y = t_data['movie_success']
    train_X=[]
    test_X=[]

    #text pre processing
    for i in range(0, len(train_X_non)):
        review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopw)]
        review = ' '.join(review)
        train_X.append(review)

    #text pre processing
    for i in range(0, len(test_X_non)):
        review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopw)]
        review = ' '.join(review)
        test_X.append(review)

    #tf idf
    tf_idf = TfidfVectorizer()
    #applying tf idf to training data
    X_train = tf_idf.fit(train_X)
    X_train_tf = tf_idf.transform(train_X)

    #transforming test data into tf-idf matrix
    X_test = tf_idf.fit(test_X)
    X_test_tf = tf_idf.transform(test_X)
    #naive bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, train_y)
    #predicted y
    y_pred = naive_bayes_classifier.predict(X_test_tf)

    test_data = st.text_input("Input any feature of a movie to see if it is correlated with a successful movie:" +
    "For example, a movie's: plot, MPAA genre, director, writer, actor/actress, title, genre can be " + 
    "inputted into the text box for analysis of our machine learning model.")

    review = re.sub('[^a-zA-Z]', ' ', test_data)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopw)]
    test_processed =[ ' '.join(review)]

    test_input = tf_idf.transform(test_processed)
    res=naive_bayes_classifier.predict(test_input)[0]
    if len(test_data) == 0:
        st.write("MOVIE_FEATURE is NOT predicted to be a title of a successful movie!")
    else:
        if res==1:
            st.write(f"{test_data} is predicted to be a feature of a successful movie!")
        elif res==0:
            st.write(f"{test_data} is NOT predicted to be a feature of a successful movie!")
    countvec = CountVectorizer(ngram_range=(1,4), 
            stop_words='english',  
            strip_accents='unicode', 
            max_features=1000)
    st.write("Generating the Confusion Matrix for the 'Title' column in the dataset")
    X = train_data.Title.values
    y = train_data.movie_success.values
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size = 0.3, 
                                            random_state = 42)
    # Instantiate classifier
    mnb = MultinomialNB()

    # Create bag of words
    X_train = countvec.fit_transform(X_train)
    X_test = countvec.transform(X_test)

    # Train the classifier/Fit the model
    mnb.fit(X_train, y_train)
            
    scores = mnb.score(X_test, y_test)

    st.write('Accuracy of Predicting Movie Success Given all Titles in the Sample: ', scores)
    # Make predictions
    y_pred = mnb.predict(X_test)

    # y_pred = naive_bayes_classifier.predict(X_test_tf)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    st.write("Heatmap of the Confusion Matrix:")
    fig, ax = plt.subplots()
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
            cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
            cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    st.write(fig)

    def classification():
        data_cols = ['Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated']
        classification = st.sidebar.selectbox("Please choose a column to sort: ", data_cols)
        grouping = st.sidebar.selectbox("Please choose a column to group by: ", data_cols, key=np.random)
        if classification == grouping:
            classification = 'Rated'
            grouping = 'Title'
        x = train_data[data_cols]

        # add column encoding the type as an integer and create dictionaries
        x[f'{classification}_id'] = x[classification].factorize()[0]
        category_id_df = x[[classification, f'{classification}_id']].drop_duplicates().sort_values(f'{classification}_id')
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[[f'{classification}_id', classification]].values)

        # checking to see the balance of classes
        fig = plt.figure(figsize=(8,6))
        x.groupby(train_data[classification])[grouping].count().plot.bar(ylim=0)
        st.subheader(f"Count of {grouping} associated with a certain {classification}:")
        st.pyplot(fig)
    classification()
