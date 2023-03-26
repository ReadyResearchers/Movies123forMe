import streamlit as st
import pandas as pd
import numpy as np
import re
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
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.util import ngrams
from sklearn import metrics
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

def predict_text():
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
    test_data = st.sidebar.text_input("Input any feature of a movie to see if it is correlated with a successful movie:")
    st.sidebar.write("For example, a movie's: plot, MPAA genre, director, writer, actor/actress, title, genre can be " + 
    "inputted into the text box for analysis of our machine learning model.")

    review = re.sub('[^a-zA-Z]', ' ', test_data)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopw)]
    test_processed =[ ' '.join(review)]

    # st.write(test_processed)
    test_input = tf_idf.transform(test_processed)
    #0= bad review
    #1= good review
    res=naive_bayes_classifier.predict(test_input)[0]
    if len(test_data) == 0:
        st.sidebar.write("MOVIE_FEATURE is NOT predicted to be a title of a successful movie!")
    else:
        if res==1:
            st.sidebar.write(f"{test_data} is predicted to be a feature of a successful movie!")
        elif res==0:
            st.sidebar.write(f"{test_data} is NOT predicted to be a feature of a successful movie!")

    countvec = CountVectorizer(ngram_range=(1,4), 
            stop_words='english',  
            strip_accents='unicode', 
            max_features=1000)
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

    st.sidebar.write('Accuracy of Predicting Movie Success Given all Titles in Sample: ', scores)
    # Make predictions
    y_pred = mnb.predict(X_test)

    # y_pred = naive_bayes_classifier.predict(X_test_tf)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    st.write('Confusion matrix given all titles in the sample:', cm)
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
    data_cols = ['imdbID', 'Title', 'Plot', 'Genre', 'Actors', 'Director', 'Writer', 'Rated', 'movie_success']
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
    stop_words_file = 'pages/SmartStoplist.txt'

    stop_words = []

    with open(stop_words_file, "r") as f:
        for line in f:
            stop_words.extend(line.split()) 
            
    stop_words = stop_words  

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def preprocess(raw_text):
        text = " ".join(str(e) for e in raw_text)
        #regular expression keeping only letters 
        letters_only_text = re.sub("[^a-zA-Z]", " ", text)

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
    train_data['tags'].dropna().reset_index()
    train_data['tags'].apply(preprocess)

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
wordcloud()
classification()

