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

def get_financials_df(yf_company):
    # get info
    company_info = yf_company.info
    # info_fields = ['symbol', 'sector', 'industry', 'country']
    info_fields = ['symbol']
    symbol_sector = {k:company_info[k] for k in info_fields}

    # get financials
    q_financials = dict(yf_company.quarterly_financials)
    dates = sorted(q_financials.keys())

    if len(dates) < 2:
        return pd.DataFrame()
    
    # get price history
    df_ohlcv = yf_company.history(period='2y')

    # the date is provided by when the statement is released -> the corresponding attributes represent the % change compared to last quarter
    metrics = q_financials[dates[0]].keys()

    perc_10q_list = []
    for i in range(1,4):
        perc_10q_dict = {'Date': dates[i]}
        cur_financials = q_financials[dates[i]]
        prev_financials = q_financials[dates[i-1]]

        for metric in metrics:
            try:
                perc_10q_dict[metric] = (cur_financials[metric] - prev_financials[metric])/prev_financials[metric]
            except:
                pass
            
        perc_10q_list.append(perc_10q_dict)

    df_10q_change = pd.DataFrame(perc_10q_list)

    # get PRICE percent change of period
    mean_price_change_list = []

    for i in range(1,4):
        start_of_quarter_date = dates[i]
        if i == 3:
            end_of_quarter_date = max(df_ohlcv.index)
        else:
            try:
                end_of_quarter_date = dates[i+1] - timedelta(days=1)
            except:
                print(dates[i+1])
        
        mean_price_change_dict = {'Date': start_of_quarter_date}

        # df_percent_change = df_ohlcv[start_of_quarter_date:end_of_quarter_date].pct_change()
        # mean_price_change_dict['Mean'] = np.mean(df_percent_change['Close'])

        try:
            start_price = df_ohlcv.loc[start_of_quarter_date]['Close']
        except:
            start_price = np.mean(df_ohlcv.loc[start_of_quarter_date - timedelta(days=7):start_of_quarter_date + timedelta(days=7)]['Close'])
        try:
            end_price = df_ohlcv.loc[end_of_quarter_date]['Close']
        except:
            end_price = np.mean(df_ohlcv.loc[end_of_quarter_date - timedelta(days=7):end_of_quarter_date + timedelta(days=7)]['Close'])

        mean_price_change_dict['Mean'] = (end_price-start_price)/start_price
        mean_price_change_list.append(mean_price_change_dict)

    df_mean_price_change = pd.DataFrame(mean_price_change_list)

    df_financials = df_10q_change.merge(df_mean_price_change, on='Date', how='inner')

    for k in symbol_sector:
        df_financials[k] = symbol_sector[k]

    return df_financials

def extend_financials_df(df_financials):
    financials_dates = list(df_financials['Date'])
    extended_financials_list = []

    for i in range(len(financials_dates)):
        start_date = financials_dates[i]
        try:
            end_date = financials_dates[i+1]
        except:
            end_date = datetime.datetime.today()
        
        col_names = list(df_financials.columns)
        vals = list(df_financials[df_financials['Date'] == start_date].values[0])

        curr_10q_dict = {k:v for k,v in zip(col_names, vals)}
        date_tracker = start_date

        while(date_tracker < end_date):
            temp_dict = curr_10q_dict.copy()
            temp_dict['Date'] = date_tracker
            extended_financials_list.append(temp_dict)
            date_tracker = date_tracker + timedelta(days=1)

    df_extended_send = pd.DataFrame(extended_financials_list)
    return df_extended_send