from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import streamlit as st

from pages.functions import B_etl
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import joblib

st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Subpage 2 🎈")

## setting up environment
st.set_option('deprecation.showPyplotGlobalUse', False)

opus = B_etl.clean_data()[0]
    
opus_cols = ['production_budget', 'rating', 'sequel', 'genre_Action', 'genre_Comedy', 'genre_Drama', 
'genre_Adventure', 'genre_Black Comedy', 'genre_Concert/Performance', 'genre_Documentary', 'genre_Horror',
'genre_Musical', 'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']
    
ox, oy = opus[opus_cols], opus['movie_success']

ox_train, ox_test, oy_train, oy_test = train_test_split(ox, oy, test_size=.7, random_state=42)
    

def lr():
    lr = LinearRegression()
    # model building
    lr_fit = lr.fit(ox_train, oy_train)
    scores_lr = lr.score(ox_test, oy_test)
    return lr_fit, scores_lr


def rf():
    rf = RandomForestRegressor(max_depth=2, random_state=42)

    rf_fit = rf.fit(ox_train, oy_train)
    scores_rf = rf.score(ox_test, oy_test)
    return rf_fit, scores_rf


def et():
    et = ExtraTreeRegressor(random_state=42)

    et_fit = et.fit(ox_train, oy_train)
    scores_et = et.score(ox_test, oy_test)
    return et_fit, scores_et

def dtc():
    dtc = DecisionTreeClassifier()

    dtc_fit = dtc.fit(ox_train, oy_train)
    scores_dtc = dtc.score(ox_test, oy_test)
    return dtc_fit, scores_dtc

def svm():
    svm = SVC()

    svm_fit = svm.fit(ox_train, oy_train)
    scores_svm = svm.score(ox_test, oy_test)
    return svm_fit, scores_svm


def logreg():
    logreg = LogisticRegression()

    logreg_fit = logreg.fit(ox_train, oy_train)
    scores_logreg = logreg.score(ox_test, oy_test)
    return logreg_fit, scores_logreg

def load_models():
    st.write("Loading models...")
    # saving the different results of the model to the disk
    lr_filename = 'lr_model.sav'
    joblib.dump(lr()[0], lr_filename)
    #random forest
    rf_filename = 'rf_model.sav'
    joblib.dump(rf()[0], rf_filename)
    #extra tree
    et_filename = 'et_model.sav'
    joblib.dump(et()[0], et_filename)
    #decision tree
    dtc_filename = 'dtc_model.sav'
    joblib.dump(dtc()[0], dtc_filename)
    #svm
    svm_filename = 'svm_model.sav'
    joblib.dump(svm()[0], svm_filename)
    # logistic regression
    logreg_filename = 'logreg_model.sav'
    joblib.dump(logreg()[0], logreg_filename)
    st.write("Done! Saved to disk.")

def main_dashboard():
    st.subheader("Running a machine learning model on movie success:")
    
    algorithms = ['Linear Regression', 'Random Forest Regressor', 'Extra Tree Regressor', 
    'Decision Tree', 'Support Vector Machine', 'Logistic Regression', 'Load Models']
    choice = st.selectbox("Choose an algorithm to train the model on: ", algorithms, key=algorithms)
    
    if choice == 'Linear Regression':
        lr = LinearRegression()
        # model building
        lr_fit = lr.fit(ox_train, oy_train)
        scores_lr = lr.score(ox_test, oy_test)
        st.write("Accuracy: ", scores_lr)
        otrain_pred = lr.predict(ox_train)
        otest_pred = lr.predict(ox_test)

        plt.figure(figsize=(5,5))
        plt.scatter(x=oy_train, y=otrain_pred, c="#7CAE00", alpha=0.3)
        z = np.polyfit(oy_train, otrain_pred, 1)
        p = np.poly1d(z)
        plt.plot(oy_train,p(otrain_pred),"#F8766D")
        plt.ylabel('Predicted LogS of Movie Success')
        plt.xlabel('Experimental LogS of Movie Success')
        st.write("Visualizing the difference between the train and test data when prediciting 'movie_success'")
        st.pyplot(plt.plot())
    if choice == 'Random Forest Regressor':
        rf = RandomForestRegressor(max_depth=2, random_state=42)

        rf_fit = rf.fit(ox_train, oy_train)
        scores_rf = rf.score(ox_test, oy_test)
        st.write("Accuracy: ", scores_rf)
        otrain_pred = rf.predict(ox_train)
        otest_pred = rf.predict(ox_test)

        plt.figure(figsize=(5,5))
        plt.scatter(x=oy_train, y=otrain_pred, c="#7CAE00", alpha=0.3)
        z = np.polyfit(oy_train, otrain_pred, 1)
        p = np.poly1d(z)
        plt.plot(oy_train,p(otrain_pred),"#F8766D")
        plt.ylabel('Predicted LogS of Movie Success')
        plt.xlabel('Experimental LogS of Movie Success')
        st.write("Visualizing the difference between the train and test data when prediciting 'movie_success'")
        st.pyplot(plt.plot())
    if choice == 'Extra Tree Regressor':
        et = ExtraTreeRegressor(random_state=42)

        et_fit = et.fit(ox_train, oy_train)
        scores_et = et.score(ox_test, oy_test)
        st.write("Accuracy: ", scores_et)
        otrain_pred = et.predict(ox_train)
        otest_pred = et.predict(ox_test)

        plt.figure(figsize=(5,5))
        plt.scatter(x=oy_train, y=otrain_pred, c="#7CAE00", alpha=0.3)
        z = np.polyfit(oy_train, otrain_pred, 1)
        p = np.poly1d(z)
        plt.plot(oy_train,p(otrain_pred),"#F8766D")
        plt.ylabel('Predicted LogS of Movie Success')
        plt.xlabel('Experimental LogS of Movie Success')
        st.write("Visualizing the difference between the train and test data when prediciting 'movie_success'")
        st.pyplot(plt.plot())
    if choice == 'Decision Tree':
        dtc = DecisionTreeClassifier()

        dtc_fit = dtc.fit(ox_train, oy_train)

        scores_dtc = dtc.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', scores_dtc)

        o_pred = dtc.predict(ox_test)

        o_cm = confusion_matrix(oy_test,o_pred)

        st.write('Confusion matrix for Opus: ', o_cm)
        st.write("Heatmap of the Confusion Matrix:")
        fig, ax = plt.subplots()
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                o_cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                    o_cm.flatten()/np.sum(o_cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(o_cm, annot=labels, fmt='', cmap='Blues', ax=ax)
        st.write(fig)
    if choice == 'Support Vector Machine':
        svm = SVC()

        svm_fit = svm.fit(ox_train, oy_train)

        scores_svm = svm.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', scores_svm)

        o_pred = svm.predict(ox_test)

        o_cm = confusion_matrix(oy_test,o_pred)

        st.write('Confusion matrix for Opus: ', o_cm)
        st.write("Heatmap of the Confusion Matrix:")
        fig, ax = plt.subplots()
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                o_cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                    o_cm.flatten()/np.sum(o_cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(o_cm, annot=labels, fmt='', cmap='Blues', ax=ax)
        st.write(fig)
    if choice == 'Logistic Regression':
        logreg = LogisticRegression()

        logreg_fit = logreg.fit(ox_train, oy_train)

        scores_logreg = logreg.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', scores_logreg)

        o_pred = logreg.predict(ox_test)

        o_cm = confusion_matrix(oy_test,o_pred)

        st.write('Confusion matrix for Opus: ', o_cm)
        st.write("Heatmap of the Confusion Matrix:")
        fig, ax = plt.subplots()
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                o_cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                    o_cm.flatten()/np.sum(o_cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(o_cm, annot=labels, fmt='', cmap='Blues', ax=ax)
        st.write(fig)
    if choice == 'Load Models':
        load_models()

main_dashboard()