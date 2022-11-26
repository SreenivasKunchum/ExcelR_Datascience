# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:22:42 2022

@author: Sreenivas Kumar
"""
import warnings
warnings.filterwarnings('ignore')

# Basic and Visualization libraries

#pip3 install sklearn

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns

#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.api import SimpleExpSmoothing
# Auto Regressive Integrated Moving Average
from statsmodels.tsa.arima.model import ARIMA

import streamlit as st

#***** Streamlit input gathering section START  ******
st.title("Stock Prediction")

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMPERIOD = st.sidebar.selectbox('Prediction period', (1,2,3,4,5))
    return CLMPERIOD

CLMPERIOD = user_input_features()

#st.button("SEMA")

#***** Streamlit input gathering section END ******

## Stock prediction process START *******

df_stock = pd.read_csv('D:\\Sreenivas\\ExcelR\\P163\\Data\\BHARTIAIRTEL.csv')
#df_stock.head()

#"""

#**Exploratory Data Analysis (EDA)**

#"""

#Insights: 1. No null values in the dataset. 
#2. Need to remove Series values other than 'EQ'
#3. Need to remove columns which are not useful for price predictions i.e. remove columns other than 'Date' and 'Close Price'
#4. 'Close Price' column to be renamed to 'Close_Price'
#5. Date column type to be changed from object to Date 
#6. Sort data in ascending order of date
#7. Set Date column as new index

#"""

# Dropping all Series column values which are not equal to 'EQ'
values=['EQ']
df_stock = df_stock[df_stock['Series'].isin(values)]

df_stock = df_stock[['Date', 'Close Price' ]]

# Renaming of column name 'Close Price'
df_stock = df_stock.rename(columns={'Close Price': 'Close_Price'})

# Remove duplicate rows
df_stock = df_stock.drop_duplicates()

df_stock['Date'] = pd.to_datetime(df_stock['Date'])

# Set Date column as index 
df_stock=df_stock.set_index(pd.DatetimeIndex(df_stock['Date']))

# Sort index in ascending order
df_stock = df_stock.sort_index(ascending=True)

# Providing data range for understanding
#print(f'Dataframe contains stock prices between {df_stock.Date.min().strftime("%Y-%m-%d")} and {df_stock.Date.max().strftime("%Y-%m-%d")}')
#print(f'Total days {(df_stock.Date.max() - df_stock.Date.min()).days} days')

df_stock['Date']= pd.to_datetime(df_stock["Date"]).dt.strftime("%Y%m%d")

# Convert Date column to integer
df_stock['Date'] = df_stock['Date'].astype('int64')

#"""
#**Analysis and Splitting of Data into Train and Test**
#"""

# We can use one month data for predicting/forecasting. Timeseries data may not do perfectly too long predictions 
# Actual values varies on a lot of other reasons other than momentum.
n=19
split_at = len(df_stock)-n

df_stock_train = df_stock.iloc[0:split_at, :]
df_stock_test = df_stock.iloc[split_at:, :]

# Create separate train dataset for Arima use since we append data while forecasting
df_stock_train_arima = df_stock.iloc[0:split_at, :]

df_stock_train_x = df_stock_train['Date']
df_stock_train_y = df_stock_train['Close_Price']

df_stock_train_array_x = df_stock_train_x.values.reshape(-1,1)
df_stock_train_array_y = df_stock_train_y.values.reshape(-1,1)

# Set Date column as index 
df_stock = df_stock.reset_index(drop=True)
#df_stock.info()


## ************** Functions of various prediction strategies **********##

## *****  Smoothed Exponential Moving Average    *****************###
def fn_sema(df_stock):
    
    fit1 = SimpleExpSmoothing(df_stock['Close_Price'], initialization_method="heuristic").fit(
        smoothing_level=0.2, optimized=False
    )  
    fcast1 = fit1.forecast(CLMPERIOD).rename(r"$\alpha=0.2$")

    fit2 = SimpleExpSmoothing(df_stock['Close_Price'], initialization_method="heuristic").fit(
        smoothing_level=0.6, optimized=False
    )
    fcast2 = fit2.forecast(CLMPERIOD).rename(r"$\alpha=0.6$")

    fit3 = SimpleExpSmoothing(df_stock['Close_Price'], initialization_method="estimated").fit()
    fcast3 = fit3.forecast(CLMPERIOD).rename(r"$\alpha=%s$" % fit3.model.params["smoothing_level"])

#fcast3

#fit1.fittedvalues

#"""**Evaluation of Model**"""

#mse = mean_squared_error(df_stock['Close_Price'], fit3.fittedvalues)
#rmse_sema = mse**.5
#print(mse)
#print(rmse_sema)

## Stock Preiction process END ******

# Stream lit OUTPUT display screen START *****
    st.subheader('User Input Parameters')
    st.write('Number of days needs to be predicted: ', CLMPERIOD)

    st.subheader('Predicted Stock Close Prices')
    st.write("Smoothed Exponential Moving Average Method used")
    st.write(fcast3)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(df_stock['Close_Price'].tail(19), marker="o", color="black")
    plt.plot(fit1.fittedvalues[-19:], marker="o", color="blue")
    (line1,) = plt.plot(fcast1, marker="o", color="blue")
    plt.plot(fit2.fittedvalues[-19:], marker="o", color="red")
    (line2,) = plt.plot(fcast2, marker="o", color="red")
    plt.plot(fit3.fittedvalues[-19:], marker="o", color="green")
    (line3,) = plt.plot(fcast3, marker="o", color="green")
    plt.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])

    st.pyplot(fig)
    
    # Stream lit OUTPUT display screen END ******
    return

## ******************  ARIMA **********************************####
def fn_arima(df_stcok):
     
    ### As per interactive testing, P value has chosen as 2 and Q as 10

    model_arima = ARIMA(df_stock['Close_Price'], order=(2,0,10))
    model_arima_fit = model_arima.fit()
    pred_arima = model_arima_fit.predict(start=1, end=len(df_stock))
    forecast = model_arima_fit.forecast(steps=CLMPERIOD)
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                            FutureWarning)
    
    # Stream lit OUTPUT display screen START *****
    st.subheader('User Input Parameters')
    st.write('Number of days needs to be predicted: ', CLMPERIOD)

    st.subheader('Predicted Stock Close Prices')
    st.write("Autoregressive Integrated Moving Average Method used")
    st.write(forecast)
 
    fig = plt.figure(figsize=(12, 8))
    (line0,) = plt.plot(df_stock['Close_Price'].tail(19), marker="o", color="black")
    plt.plot(pred_arima[-20:], marker="o", color="blue")
    (line1,) = plt.plot(forecast, marker="o", color="blue")
    
    plt.legend([line0, line1], ['Actual', 'Predicted'])
    
    st.pyplot(fig)

    return

#### Main program section
if st.sidebar.button("SEMA"):
   fn_sema(df_stock)

if st.sidebar.button("ARIMA"):
   fn_arima(df_stock)


