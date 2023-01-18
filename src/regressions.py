import pandas as pd
import streamlit as st
import numpy as np
import data_loading
import etl

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

st.set_option('deprecation.showPyplotGlobalUse', False)

def linear_regression():
    opus = etl.clean_data()[0]
    opus_binary = opus[['domestic_box_office', 'production_budget']]
    opus_binary500 = opus_binary[:][:300]
    # Selecting the 1st 500 rows of the data
    st.pyplot(sns.lmplot(x ='production_budget', y ='domestic_box_office', data = opus_binary500,
                        order = 2, ci = None).set(
                        title='Comparison of 500 Domestic Box Office Sales with Production Budget'))
    st.pyplot(sns.lmplot(y = 'domestic_box_office', x = 'production_budget', data = opus_binary, 
                        order = 2, ci = None).set(
                        title='Comparison of Domestic Box Office Sales with Production Budget'))

    # training the model
    X = np.array(opus_binary['production_budget']).reshape(-1,1)
    y = np.array(opus_binary['domestic_box_office']).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    st.write(regr.score(X_test, y_test))
    y_pred = regr.predict(X_test)
    plt.scatter(X_test, y_test, color ='r')
    plt.plot(X_test, y_pred, color ='k')
    st.pyplot(plt.show())
    
    X = np.array(opus_binary500['production_budget']).reshape(-1, 1)
    y = np.array(opus_binary500['domestic_box_office']).reshape(-1, 1)
    opus_binary500.dropna(inplace = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    st.write(regr.score(X_test, y_test))
    y_pred = regr.predict(X_test)
    plt.scatter(X_test, y_test, color ='b')
    plt.plot(X_test, y_pred, color ='k')
    st.pyplot(plt.show())
  
    mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
    #squared True returns MSE value, False returns RMSE value.
    mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
    rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
    
    st.write("MAE:",mae)
    st.write("MSE:",mse)
    st.write("RMSE:",rmse)
    return opus

linear_regression()