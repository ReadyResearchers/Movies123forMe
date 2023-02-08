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
from sklearn import metrics


st.set_option('deprecation.showPyplotGlobalUse', False)

def ml():
    st.subheader("Running a machine learning model on movie success:")
    opus = etl.clean_data()[0]
    st.write("Head of Opus Data", opus.head())


    opus_cols = ['production_budget', 'rating', 'sequel', 'genre_Action', 'genre_Comedy', 'genre_Drama', 
    'genre_Adventure', 'genre_Black Comedy', 'genre_Concert/Performance', 'genre_Documentary', 'genre_Horror',
    'genre_Musical', 'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']

    x, y = opus[opus_cols], opus['movie_success']

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # model building
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

    # random forest
    rf = RandomForestRegressor(max_depth=2, random_state=42)
    rf.fit(X_train, y_train)

    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    # model performance
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    rf_results = pd.DataFrame(['Random forest',rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']


    # using a regressor
    et = ExtraTreeRegressor(random_state=42)
    et.fit(X_train, y_train)

    y_et_train_pred = et.predict(X_train)
    y_et_test_pred = et.predict(X_test)

    et_train_mse = mean_squared_error(y_train, y_et_train_pred)
    et_train_r2 = r2_score(y_train, y_et_train_pred)
    et_test_mse = mean_squared_error(y_test, y_et_test_pred)
    et_test_r2 = r2_score(y_test, y_et_test_pred)

    et_results = pd.DataFrame(['Extra Tree',et_train_mse, et_train_r2, et_test_mse, et_test_r2]).transpose()
    et_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']


    # display in one table
    st.write("Displaying results of running the different machine learning methods:", pd.concat([lr_results, rf_results, et_results]))

    # data visualization
    plt.figure(figsize=(5,5))
    plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
    z = np.polyfit(y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_train,p(y_train),"#F8766D")
    plt.ylabel('Predicted LogS')
    plt.xlabel('Experimental LogS')
    st.write("Visualizing the difference between the train and test data:")
    st.pyplot(plt.plot())


def machine_model():
    st.subheader("Running another machine learning model over movie success: ")
    opus = etl.clean_data()[0]
    if st.checkbox("Show dataframe", key="111"):
        st.write(opus)

    col1 = st.selectbox('Which option on x?', opus.columns[:])
    col2 = st.selectbox('Which option on y?', opus.columns[:])

    fig = px.scatter(opus, x = col1, y = col2, color = 'rating', title = f'Effect of {col2} on {col1} according to rating')
    st.plotly_chart(fig)

    feature = st.selectbox("Which factor?", opus.columns[:])
    # filter dataframe
    fig2 = px.histogram(opus, x=feature, color='rating', marginal="rug", title = f'Effect of count of {feature} compared to rating')
    st.plotly_chart(fig2)

    x = opus[['production_year', 'production_budget', 'sequel', 'running_time']].values
    y = opus.movie_success
    X_train,X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
    alg = ['Decision Tree', 'Support Vector Machine', 'Logistic Regression']
    classifier = st.selectbox('Which algorithm?', alg, key="122")
    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
        st.write("Heatmap of the Confusion Matrix:")
        fig, ax = plt.subplots()
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                cm_dtc.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                     cm_dtc.flatten()/np.sum(cm_dtc)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm_dtc, annot=labels, fmt='', cmap='Blues', ax=ax)
        st.write(fig)
    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm)
        st.write("Heatmap of the Confusion Matrix:")
        fig1, ax1 = plt.subplots()
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax1)
        st.write(fig1)

    elif classifier == 'Logistic Regression':
        features = ['production_budget', 'rating', 'sequel', 'genre_Action', 'genre_Comedy', 'genre_Drama', 
        'genre_Adventure', 'genre_Black Comedy', 'genre_Concert/Performance', 'genre_Documentary', 'genre_Horror',
        'genre_Musical', 'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']
        x = opus[features]
        y = opus.movie_success

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
        logreg = LogisticRegression(random_state=16)
        logreg.fit(x_train, y_train)
        acc = logreg.score(x_test, y_test)
        st.write('Accuracy: ', acc)
        y_pred = logreg.predict(x_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write('Confusion matrix:', cnf_matrix)

        class_names = [0,1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion Matrix', y=1.1)
        plt.ylabel("Actual Movie Success")
        plt.xlabel("Predicted Movie Success")
        st.write(fig)

ml()
machine_model()
