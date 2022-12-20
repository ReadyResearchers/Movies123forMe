import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

st.title('Movies123ForMe')
df = pd.read_csv("src\\movie_data\\movie_data\\movie_data.csv")
df = df.dropna()
if st.checkbox("Show dataframe"):
    st.write(df)

genres = st.multiselect("Show genre of movies",
df['genre'].unique())
col1 = st.selectbox('Which option on x?', df.columns[0:6])
col2 = st.selectbox('Which option on y?', df.columns[0:6])

new_df = df[(df['genre'].isin(genres))]
st.write(new_df)
fig = px.scatter(new_df, x = col1, y = col2, color = 'genre')
st.plotly_chart(fig)

feature = st.selectbox("Which factor?", df.columns[0:6])
# filter dataframe
new_df2 = df[(df['genre'].isin(genres))][feature]
fig2 = px.histogram(new_df, x=feature, color='genre', marginal="rug")
st.plotly_chart(fig2)

features= df[['production_year', 'movie_odid', 'production_budget', 'domestic_box_office', 'international_box_office', 'sequel', 'running_time']].values
labels = df['genre'].values
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)
elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)