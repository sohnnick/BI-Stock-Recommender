import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBRegressor
import sklearn
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pmdarima import auto_arima
from sklearn import metrics
from datetime import datetime, timezone, date, timedelta
from pmdarima import auto_arima
from sklearn import metrics

def train_predict_arima(df, prd):
    #prd determine variable of percent used for test train. 0.33 one third of data is test.
    #split data using variable defined 
    train_data, test_data = df[0:prd],df[prd:]
    
    #external variable for ARIMAX
    exoX   =  df[['Close']]
    exotrain, exotest = exoX[0:prd], exoX[prd:]
    
    #ARIMA VARIABLE FOR SEASONALITY
    m=1
    
    ######################## model training ###########################################
    print(f'  SARIMAX m = {str(m)}') 
    model = auto_arima(train_data['Open'], exogenous = exotrain , seasonal=True, m = m,
                       d=None,D=None, trace=True, stepwise=True)

    print("-"*100)
    #output summary of model
    #model.summary()

    #forcast 
    forecast = model.predict(n_periods=-prd,exogenous =exotest, return_conf_int=False)
    

    #adjust index of predictions  
    #index shift due to predictions start from 2/3 of data subtract the -1/3 (to add) 
    dif = len(train_data)-prd        #-1 due to train_data starting at 0
    
    forecast  =   pd.DataFrame(forecast, columns=['open_pred'])
    forecast["new_index"] = range(len(train_data), dif)
    forecast  =   forecast.set_index("new_index")
    
    return model, forecast

def price_predictions(model, exot, num_pred):
    forecast = model.predict(n_periods= num_pred,exogenous = exot[-num_pred:] , return_conf_int=False)
    print()
    print("Price forcasted for next",num_pred," days" , forecast)
    print("-"*100)
    return forecast

def predictStockPrice(df_yf):
    prd = -int(len(df_yf)*0.33)
    pred_dates = [x.strftime('%Y-%m-%d') for x in list(df_yf[prd:].index)]
    model, forecast = train_predict_arima(df_yf, prd)
    nex_day_pred = price_predictions(model, df_yf[['Close']], num_pred = 1)

    rmse = mean_squared_error(df_yf[prd:]['Open'].values, list(forecast['open_pred']), squared=False)
    zip_validation = zip(list(df_yf[prd:]['Open'].values), list(forecast['open_pred']))
    mape = np.mean([abs((x-y)/x) for x,y in zip_validation])*100

    return pred_dates, forecast, nex_day_pred, rmse, mape