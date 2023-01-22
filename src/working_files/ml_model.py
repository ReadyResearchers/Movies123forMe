import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from working_files import etl

def machine_model():
    opus = etl.clean_data()[0]
    if st.checkbox("Show dataframe", key="111"):
        st.write(opus)

    genres = st.multiselect("Show genre of movies",
    opus['genre'].unique())
    col1 = st.selectbox('Which option on x?', opus.columns[0:6])
    col2 = st.selectbox('Which option on y?', opus.columns[0:6])

    new_opus = opus[(opus['genre'].isin(genres))]
    st.write(new_opus)
    fig = px.scatter(new_opus, x = col1, y = col2, color = 'genre')
    st.plotly_chart(fig)

    feature = st.selectbox("Which factor?", opus.columns[0:6])
    # filter dataframe
    new_opus2 = opus[(opus['genre'].isin(genres))][feature]
    fig2 = px.histogram(new_opus, x=feature, color='genre', marginal="rug")
    st.plotly_chart(fig2)

    features= opus[['production_year', 'movie_odid', 'production_budget', 'domestic_box_office', 'international_box_office', 'sequel', 'running_time']].values
    labels = opus['genre'].values
    X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
    alg = ['Decision Tree', 'Support Vector Machine']
    classifier = st.selectbox('Which algorithm?', alg, key="122")
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
    return opus, new_opus2

machine_model()