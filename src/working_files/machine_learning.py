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

from src.working_files import etl
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)

def machine_learning():
    st.subheader("Running a machine learning model on movie success:")
    opus = etl.clean_data()[0]

    algorithms = ['Linear Regression', 'Random Forest Regressor', 'Extra Tree Regressor', 
    'Decision Tree', 'Support Vector Machine', 'Logistic Regression', 'All']
    choice = st.selectbox("Choose an algorithm to train the model on: ", algorithms)
    
    
    opus_cols = ['production_budget', 'rating', 'sequel', 'genre_Action', 'genre_Comedy', 'genre_Drama', 
    'genre_Adventure', 'genre_Black Comedy', 'genre_Concert/Performance', 'genre_Documentary', 'genre_Horror',
    'genre_Musical', 'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']
    
    ox, oy = opus[opus_cols], opus['movie_success']

    ox_train, ox_test, oy_train, oy_test = train_test_split(ox, oy, test_size=.7, random_state=42)
    
    lr = LinearRegression()
    rf = RandomForestRegressor(max_depth=2, random_state=42)
    et = ExtraTreeRegressor(random_state=42)
    dtc = DecisionTreeClassifier()
    svm = SVC()
    logreg = LogisticRegression()

    ## training the model and saving it for predictions!
    logreg.fit(ox_train, oy_train)
    filename = 'finalized_movie_model.sav'
    joblib.dump(logreg, filename)

    if choice == 'Linear Regression':
        # model building
        lr.fit(ox_train, oy_train)

        otrain_pred = lr.predict(ox_train)
        otest_pred = lr.predict(ox_test)

        lr_train_mse = mean_squared_error(oy_train, otrain_pred)
        lr_train_r2 = r2_score(oy_train, otrain_pred)
        lr_test_mse = mean_squared_error(oy_test, otest_pred)
        lr_test_r2 = r2_score(oy_test, otest_pred)

        lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
        lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

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
        rf.fit(ox_train, oy_train)

        otrain_pred = rf.predict(ox_train)
        otest_pred = rf.predict(ox_test)

        lr_train_mse = mean_squared_error(oy_train, otrain_pred)
        lr_train_r2 = r2_score(oy_train, otrain_pred)
        lr_test_mse = mean_squared_error(oy_test, otest_pred)
        lr_test_r2 = r2_score(oy_test, otest_pred)

        lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
        lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

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
        et.fit(ox_train, oy_train)

        otrain_pred = et.predict(ox_train)
        otest_pred = et.predict(ox_test)

        lr_train_mse = mean_squared_error(oy_train, otrain_pred)
        lr_train_r2 = r2_score(oy_train, otrain_pred)
        lr_test_mse = mean_squared_error(oy_test, otest_pred)
        lr_test_r2 = r2_score(oy_test, otest_pred)

        lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
        lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

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
        dtc.fit(ox_train, oy_train)

        o_acc = dtc.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', o_acc)

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
    if choice == 'Support Vectore Machine':
        svm.fit(ox_train, oy_train)

        o_acc = svm.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', o_acc)

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
        logreg.fit(ox_train, oy_train)


        o_acc = logreg.score(ox_test, oy_test)

        st.write('Accuracy for Opus: ', o_acc)

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

machine_learning()