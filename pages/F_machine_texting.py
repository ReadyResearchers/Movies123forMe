import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from pages.function_folder import A_data_loading
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
from nltk.util import ngrams
from sklearn import metrics
import nltk

from sklearn.feature_selection import chi2

st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Subpage 4 🎈")

st.write("---")
duplicates = 'movie_search.csv'

train_data = pd.read_csv('movie_clean.csv',encoding='ISO-8859-1')

train_data.columns = ['Title','Year','Rated','Released','Runtime','Genre','Director',
'Writer','Actors','Plot','Language','Country','Awards','Poster','Ratings',
'Metascore','imdbRating','imdbVotes','imdbID','Type','DVD','BoxOffice',
'Production','Website','Response', 'movie_success','earnings']

train_data = train_data.drop_duplicates(subset = ['Title'], keep='first').reset_index()
data_cols = ['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated', 'movie_success']
train_data[['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated', 'movie_success']]
# #st.write(movies.isnull().sum())
train_data = train_data.drop_duplicates().reset_index()
# movies['tags'] = movies['Genre'] + " " +  movies['Plot'] + " " + movies['Actors'] + " " + movies['Director'] + " " + movies['Writer'] + " " + movies['Rated']


def predict_text():
    def actor():
        t_data = train_data[['Actors', 'movie_success']].dropna().reset_index()
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie actor/actresses to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_ACTOR/ACTRESS is NOT predicted to be an actor/actress of a successful movie!")
        else:
            if res==1:
                st.write(f"{test_data} is predicted to be an actor/actress a successful movie!")
            elif res==0:
                st.write(f"{test_data} is NOT predicted to be an actor/actress a successful movie!")

    def description():  
        t_data = train_data[['Plot', 'movie_success']].dropna().reset_index()
        #stopword removal and lemmatization
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        # st.write(t_data.head())

        train_X_non = t_data['Plot']   # '0' refers to the review text
        train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
        test_X_non = t_data['Plot']
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie description to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_DESCRIPTION is NOT predicted to be a Plot of a successful movie!")
        else:
            if res==1:
                st.write(f"{test_data} is predicted to be a Plot of a successful movie!")
            elif res==0:
                st.write(f"{test_data} is NOT predicted to be a Plot of a successful movie!")

    def genre():
            
        t_data = train_data[['Genre', 'movie_success']].dropna().reset_index()
        #stopword removal and lemmatization
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        # st.write(t_data.head())

        train_X_non = t_data['Genre']   # '0' refers to the review text
        train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
        test_X_non = t_data['Genre']
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie genre to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_GENRE is NOT predicted to be a genre of a successful movie!")
        else:
            if res==1:
                st.write(f"{test_data} is predicted to be a genre of a successful movie!")
            elif res==0:
                st.write(f"{test_data} is NOT predicted to be a genre of a successful movie!")

    def title():  
        t_data = train_data[['Title', 'movie_success']].dropna().reset_index()
        #stopword removal and lemmatization
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        # st.write(t_data.head())

        train_X_non = t_data['Title']   # '0' refers to the review text
        train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
        test_X_non = t_data['Title']
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie title to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_TITLE is NOT predicted to be a title of a successful movie!")
        else:
            if res==1:
                st.write(f"{test_data} is predicted to be a title of a successful movie!")
            elif res==0:
                st.write(f"{test_data} is NOT predicted to be a title of a successful movie!")


    def director():
        t_data = train_data[['Director', 'movie_success']].dropna().reset_index()
        #stopword removal and lemmatization
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        # st.write(t_data.head())

        train_X_non = t_data['Director']   # '0' refers to the review text
        train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
        test_X_non = t_data['Director']
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie director to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_DIRECTOR is NOT predicted to be a Director of a successful movie!")
        
        if res==1:
            st.write(f"{test_data} is predicted to be a Director of a successful movie!")
        elif res==0:
            st.write(f"{test_data} is NOT predicted to be a Director of a successful movie!")


    def rating():
        t_data = train_data[['Rated', 'movie_success']].dropna().reset_index()
        #stopword removal and lemmatization
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        # st.write(t_data.head())

        train_X_non = t_data['Rated']   # '0' refers to the review text
        train_y = t_data['movie_success']   # '1' corresponds to Label (1 - positive and 0 - negative)
        test_X_non = t_data['Rated']
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
        X_train = tf_idf.fit(train_X)
        X_train_tf = tf_idf.transform(train_X)
        # st.write("n_samples: %d, n_features: %d" % X_train_tf.shape)

        #transforming test data into tf-idf matrix
        X_test = tf_idf.fit(test_X)
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
        test_data = st.text_input("Input a movie MPAA rating to see if they would be in a successful movie:")

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
        if len(test_data) == 0:
            st.write("MOVIE_RATING is NOT predicted to be a rating of a successful movie!")
        else:
            if res==1:
                st.write(f"{test_data} is predicted to be a rating of a successful movie!")
            elif res==0:
                st.write(f"{test_data} is NOT predicted to be a rating of a successful movie!")


    col = ['Title', 'Director', 'Actors', 'Rated', 'Genre', 'Plot']
    st.subheader("Choose which feature to predict movie success!")
    types = st.selectbox("", col)
    
    if types == 'Title':
        title()
    if types == 'Director':
        director()
    if types == 'Actors':
        actor()
    if types == 'Rated':
        rating()
    if types == 'Genre':
        genre()
    if types == 'Plot':
        description()


def classification():
    classification = st.selectbox("Please choose a column to find the unigrams and bigrams for: ", data_cols)
    grouping = st.selectbox("Please choose a column to group by: ", data_cols, key=np.random)
    x = train_data[data_cols]
    y = train_data[data_cols].drop('movie_success', axis=1)

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

    # extracting features from text using the measure term frequency inverse document frequency (tfidf)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(x[grouping]).toarray()
    labels = x[f'{classification}_id']

    st.subheader(f"Correlated words grouped by {classification}")
    N = 2
    for Product, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        st.write("# '{}':".format(Product))
        st.write("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        st.write("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        st.write("---")


def wordcloud():
    columns = ['Title', 'Director', 'Actors', 'Rated', 'Genre', 'Plot']
    st.subheader("Choose which feature to generate a word cloud for!")
    choice = st.selectbox("", columns, key=columns)
    stop_words_file = 'pages\\SmartStoplist.txt'

    stop_words = []

    with open(stop_words_file, "r") as f:
        for line in f:
            stop_words.extend(line.split()) 
            
    stop_words = stop_words  

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def preprocess(raw_text):
        
        #regular expression keeping only letters 
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

        # convert to lower case and split into words -> convert string into list ( 'hello world' -> ['hello', 'world'])
        words = letters_only_text.lower().split()

        cleaned_words = []
        lemmatizer = PorterStemmer() #plug in here any other stemmer or lemmatiser you want to try out
        
        # remove stopwords
        for word in words:
            if word not in stop_words:
                cleaned_words.append(word)
        
        # stemm or lemmatise words
        stemmed_words = []
        for word in cleaned_words:
            word = lemmatizer.stem(word)   #dont forget to change stem to lemmatize if you are using a lemmatizer
            stemmed_words.append(word)
        
        # converting list back to string
        return " ".join(stemmed_words)
    train_data['prep'] = train_data[f'{choice}'].dropna().reset_index().apply(preprocess)

    most_common = Counter(" ".join(train_data["prep"]).split()).most_common(10)

    #nice library to produce wordclouds
    all_words = '' 

    #looping through all incidents and joining them to one text, to extract most common words
    for arg in train_data["prep"]: 

        tokens = arg.split()  
        
        all_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 700, height = 700, 
                    background_color ='white', 
                    min_font_size = 10).generate(all_words) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (5, 5), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    st.pyplot(plt.show())

    n_gram = 2
    n_gram_dic = dict(Counter(ngrams(all_words.split(), n_gram)))

    for i in n_gram_dic:
        if n_gram_dic[i] >= 2:
            pairs = i, n_gram_dic[i]



predict_text()
# wordcloud()
# classification

