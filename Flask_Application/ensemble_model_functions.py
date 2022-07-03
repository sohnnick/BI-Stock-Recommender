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
from pmdarima import auto_arima
from sklearn import metrics
import pickle

from financials_prep import *

def predict_using_ensemble(yf_company, df_ensemble):
    # prepare financial statements
    df_ticker_fin = get_financials_df(yf_company)
    df_ticker_fin_extended = extend_financials_df(df_ticker_fin)
    
    # prepare data frames for merge
    df_ensemble = df_ensemble.reset_index()
    df_ensemble['Date'] = pd.to_datetime(df_ensemble['Date'])
    df_ticker_fin_extended['Date'] = pd.to_datetime(df_ticker_fin_extended['Date'])
    # df_ensemble = df_ensemble.set_index('Date')
    # df_ticker_fin_extended = df_ticker_fin_extended.set_index('Date')

    # prepared merged data
    df_merged = pd.merge(df_ensemble, df_ticker_fin_extended,  how='inner', on='Date')
    # df_merged = df_merged.reset_index()

    # df_merged.to_csv('test_merge.csv')
    df_merged.rename(columns = {'Arima_Pred':'Arima_predictions'}, inplace = True)

    x_cols = ['Arima_predictions', 'Research Development', 'Income Before Tax', 'Minority Interest',
        'Net Income', 'Selling General Administrative', 'Gross Profit', 'Ebit',
        'Operating Income', 'Interest Expense', 'Total Revenue',
        'Total Operating Expenses', 'Cost Of Revenue',
        'Total Other Income Expense Net', 'Net Income From Continuing Ops',
        'Net Income Applicable To Common Shares',
        'Income Tax Expense', 'Discontinued Operations',
        'Other Operating Expenses']
    y_col = 'Open'

    for col in x_cols:
        if col not in df_merged.columns:
            df_merged[col] = None

    x = df_merged[x_cols]
    y = df_merged[y_col]

    # x.to_csv('x_train_test.csv')

    # load model
    fp_xgb = open('./models/ensemble_model.pkl','rb')
    model_xgb = pickle.load(fp_xgb)

    # make sure every column is a float (non-object)
    for col in x.columns:
        x[col] = x[col].astype('float')

    # predict
    y_pred = model_xgb.predict(x)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mape = np.mean([abs((x-y)/x) for x,y in zip(y, y_pred)])*100

    # prepare to return data
    forecast = list(y_pred)
    dates = list(df_merged['Date'])
    max_value = max(dates)
    max_index = dates.index(max_value)
    next_day = y_pred[max_index]

    # also get dates for the predictions
    pred_dates = [x.strftime("%Y-%m-%d") for x in dates]

    return pred_dates, forecast, next_day, rmse, mape