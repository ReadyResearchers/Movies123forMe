import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from src.working_files import etl
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from nltk.util import ngrams

netflix = etl.clean_data()[1]
field = 'text desc'
feature_rep = 'binary'
top_k = 8
# setting up the variables and dataframes to be used
netflix_cols = ['type', 'director', 'cast', 'release_year', 'rating', 'country','description', 'listed_in', 'movie_success']
netflix = netflix.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

x = netflix[netflix_cols]
y = netflix[netflix_cols].drop('movie_success', axis=1)

stop_words_file = 'C:\\Users\\solis\\OneDrive\\Documents\\comp\\Movies123forMe\\src\\working_files\\SmartStoplist.txt'

stop_words = []

with open(stop_words_file, "r") as f:
    for line in f:
        stop_words.extend(line.split()) 
        
stop_words = stop_words  

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

netflix['prep'] = netflix['description'].apply(preprocess)
st.write(netflix.head())

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
        st.write(i, n_gram_dic[i])
    