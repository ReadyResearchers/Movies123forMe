from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import streamlit as st

data = pd.read_csv('src\\movie_data\\movie_data\\movie_data.csv')

# data processing
data1 = data.drop(['movie_name', 'rating', 'genre', 'creative_type', 'source', 'production_method', 'domestic_box_office', 'international_box_office'], axis=1)
data2 = data1.dropna()
x = data2.drop(['total_revenue'], axis=1)
y = data2['total_revenue']

# data splitting
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
st.write(pd.concat([lr_results, rf_results, et_results]))

# data visualization
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"#F8766D")
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

st.pyplot(plt.plot())