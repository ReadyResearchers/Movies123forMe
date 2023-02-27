import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from pages.function_folder import A_data_loading
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
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

from io import StringIO
from sklearn.feature_selection import chi2

st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Subpage 4 🎈")

st.write("---")
data = A_data_loading.load_data_netflix(1000)
datap = A_data_loading.load_data_prime(1000)
datad = A_data_loading.load_data_disney(1000)

netflix = data.merge(datap, how='left').merge(datad, how='left')
netflix = netflix[netflix['type'].str.contains("TV Show") == False]

# making movie success where the 4 top produced movies are categorized as a success
netflix['movie_success'] = np.where(
    netflix['rating'] == 'TV-MA', 1, np.where(
    netflix['rating'] == 'PG-13', 1, np.where(netflix['rating'] == 'TV-14', 1, np.where(
        netflix['rating'] == 'R', 1, 0))))

#dropping columns
netflix_drop = ['date_added', 'duration']
if netflix.columns.any() in netflix_drop:
    netflix.drop(netflix_drop, inplace=True, axis=1)

#dropping nan values
netflix = netflix.dropna()
netflix['release_year'] = pd.to_datetime(netflix['release_year'])
netflix["date_added"] = pd.to_datetime(netflix['date_added'])

st.write(netflix)
# setting up the variables and dataframes to be used
netflix_cols = ['title', 'type', 'director', 'cast', 'release_year', 'rating', 'country','description', 'listed_in', 'movie_success']
netflix = netflix.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

def text_classification():
    x = netflix[netflix_cols]
    y = netflix[netflix_cols].drop('movie_success', axis=1)

    # add column encoding the type as an integer and create dictionaries
    x['rating_id'] = x['rating'].factorize()[0]
    category_id_df = x[['rating', 'rating_id']].drop_duplicates().sort_values('rating_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['rating_id', 'rating']].values)

    # checking to see the balance of classes
    fig = plt.figure(figsize=(8,6))
    x.groupby('rating').cast.count().plot.bar(ylim=0)
    st.subheader("Count of Cast members associated with a certain rating:")
    st.pyplot(fig)

    # extracting features from text using the measure term frequency inverse document frequency (tfidf)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(x.cast).toarray()
    labels = x.rating_id

    st.subheader("Correlated words grouped by MPAA Rating")
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


    def actor():
        X_train, X_test, y_train, y_test = train_test_split(netflix['cast'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input an actor/actress to see if they would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the actor/actress '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")


    def description():  
        X_train, X_test, y_train, y_test = train_test_split(netflix['description'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a Movie description to see if they would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the description '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")
        

    def genre():
            
        X_train, X_test, y_train, y_test = train_test_split(netflix['listed_in'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a genre to see if they would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the genre '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")
    

    def title():  
        X_train, X_test, y_train, y_test = train_test_split(netflix['title'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a movie title to see if they would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the title '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")
    
    def director():
        X_train, X_test, y_train, y_test = train_test_split(netflix['director'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a movie director to see if they would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the director '{test_cast.upper()}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")
    

    def country():
        X_train, X_test, y_train, y_test = train_test_split(netflix['country'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a country for a movie to see if it would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the country '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")

    def rating():
        X_train, X_test, y_train, y_test = train_test_split(netflix['rating'], netflix['movie_success'], random_state = 42)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        st.subheader("Please input a rating for a movie to see if it would make a successful movie:")
        test_cast = st.text_input(" ")
        if len(test_cast) != 0:
            st.write(f"Predicting the success of a movie with the rating '{test_cast}'")
            prediction = clf.predict(count_vect.transform([test_cast]))[0]
            if prediction == 0:
                st.write(f"{test_cast.upper()} is predicted to be in a Successful Movie!")
            else:
                st.write(f"{test_cast.upper()} is predicted to NOT be in a Successful Movie!")


    col = ['title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']
    st.subheader("Choose which feature to predict movie success!")
    types = st.selectbox("", col)
    
    if types == 'title':
        title()
    if types == 'director':
        director()
    if types == 'cast':
        actor()
    if types == 'country':
        country()
    if types == 'rating':
        rating()
    if types == 'listed_in':
        genre()
    if types == 'description':
        description()



def wordcloud():
    columns = ['title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']
    st.subheader("Choose which feature to generate a word cloud for!")
    choice = st.selectbox("", columns, key=columns)
    stop_words_file = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\pages\\SmartStoplist.txt'

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

    netflix['prep'] = netflix[f'{choice}'].apply(preprocess)

    most_common = Counter(" ".join(netflix["prep"]).split()).most_common(10)

    #nice library to produce wordclouds
    all_words = '' 

    #looping through all incidents and joining them to one text, to extract most common words
    for arg in netflix["prep"]: 

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


text_classification()
wordcloud()

