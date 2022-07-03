import yfinance as yf
import pandas as pd
import datetime as datetime
from datetime import timedelta
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import Timestamp
import pickle
from pmdarima import auto_arima
from sklearn import metrics

def train_predict_arima(df, prd):
    #prd determine variable of percent used for test train. 0.33 one third of data is test.
    #split data using variable defined 
    train_data, test_data = df[0:prd],df[prd:]
    
    #external variable for ARIMAX
    exoX = df[['Close_pct_change']]
    exotrain, exotest = exoX[0:prd], exoX[prd:]
    
    #ARIMA VARIABLE FOR SEASONALITY
    m=1
    
    ######################## model training ###########################################
    print(f'  SARIMAX m = {str(m)}') 
    model = auto_arima(train_data['Open_pct_change'], exogenous = exotrain , seasonal=True, m = m,
                       d=None,D=None, trace=True, stepwise=True)

    print("-"*100)
    #output summary of model
    #model.summary()

    #forcast
    forecast = model.predict(n_periods=-prd,exogenous=exotest, return_conf_int=False)
    
    #adjust index of predictions
    #index shift due to predictions start from 2/3 of data subtract the -1/3 (to add) 
    dif = len(train_data)-prd        #-1 due to train_data starting at 0
    
    forecast  =   pd.DataFrame(forecast, columns=['open_pred'])
    forecast["new_index"] = range(len(train_data), dif)
    forecast  =   forecast.set_index("new_index")
    
    return model, forecast

def get_recommendation(df_yf):
    print(df_yf)
    df_copy = df_yf.copy()

    # data preprocessing
    df_copy['Open_pct_change'] = df_copy['Open'].pct_change(periods=1)*100
    df_copy['Close_pct_change'] = df_copy['Close'].pct_change(periods=1)*100
    df_copy = df_copy[df_copy['Open_pct_change'].notnull()]

    # train model
    prd = -int(len(df_copy)*0.33)
    model, forecast = train_predict_arima(df_copy, prd)

    # shift accordingly
    recs_shifted = [None]*(len(df_copy)+prd+1) + list(forecast['open_pred'])
    next_day_pct_change = float(recs_shifted.pop())

    print(next_day_pct_change)

    # obtain next day prediction
    if next_day_pct_change <= 1 and next_day_pct_change >= -1:
        next_day_rec = 'Hold'
    elif next_day_pct_change > 1:
        next_day_rec = 'Buy'
    else:
        next_day_rec = 'Sell'

    # filter so no null
    df_copy['pct_pred'] = recs_shifted
    df_copy = df_copy[df_copy['pct_pred'].notnull()]

    # obtain validation recs and accuracy
    truth_rec = []
    pred_rec = []
    truth_pct_change = list(df_copy['Open_pct_change'])
    pred_pct_change = list(df_copy['pct_pred'])
    for i in range(len(truth_pct_change)):
        if truth_pct_change[i] <= 1 and truth_pct_change[i] >= -1:
            #print(‘hold’)
            truth_rec.append('Hold')
        elif truth_pct_change[i] > 1:
            #print(‘sell’)
            truth_rec.append('Buy')
        elif truth_pct_change[i] < -1:
            #print(‘buy’)
            truth_rec.append('Sell')

        if pred_pct_change[i] <= 1 and pred_pct_change[i] >= -1:
            #print(‘hold’)
            pred_rec.append('Hold')
        elif pred_pct_change[i] > 1:
            #print(‘sell’)
            pred_rec.append('Buy')
        elif pred_pct_change[i] < -1:
            #print(‘buy’)
            pred_rec.append('Sell')
    accuracy = metrics.accuracy_score(truth_rec, pred_rec)

    # prep send
    pred_dates = [x.strftime('%Y-%m-%d') for x in list(df_copy.index)]
    validation_truth = list(df_copy['Open_pct_change'])
    validation_pred = list(df_copy['pct_pred'])

    return pred_dates, validation_truth, validation_pred, next_day_rec, accuracy