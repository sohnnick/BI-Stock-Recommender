import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tweepy
from datetime import datetime, timezone, date, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import yfinance as yf
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# api keys
import config

def get_tweets(queryParam):
    consumer_key = config.CONSUMER_KEY
    consumer_secret = config.CONSUMER_SECRET
    access_token = config.ACCESS_TOKEN
    access_secret = config.ACCESS_SECRET
    bearer = config.BEARER

    from datetime import datetime, timezone, date, timedelta
    now = datetime.now()
    if now.hour < 12:
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        now = now.replace(hour=12, minute=0, second=0, microsecond=0)

    # obtain all tweets within the past week
    dtformat = '%Y-%m-%dT%H:%M:%SZ'
    curr_day = now.strftime(dtformat)
    dfs = []
    currs = []
    prevs = []

    for i in tqdm(range(1, 7)):
        prev_day = ((now + timedelta(hours=-12))).strftime(dtformat)

        endpoint = 'https://api.twitter.com/2/tweets/search/recent'
        headers = {'authorization': f'Bearer {bearer}'}

        params = {
            'query': queryParam,
            'max_results': '30',
            'tweet.fields': 'created_at,lang',
            'start_time': prev_day,
            'end_time': curr_day
        }

        response = requests.get(endpoint,
                                    params=params,
                                    headers=headers)
        tweets_dict = dict(response.json())['data']
        time = [datetime(now.year, now.month, now.day) for x in range(0, len(tweets_dict))]
        tweets = []
        time_actual = []

        for tweet in tweets_dict:
            tweets.append(tweet['text'])
            time_actual.append(tweet['created_at'])

        df_tweets = pd.DataFrame({
            'end time':time,
            'actual time': time_actual,
            'tweet': tweets
        })

        currs.append(curr_day)
        prevs.append(prev_day)

        dfs.append(df_tweets)

        if now.hour == 0:
            now = datetime.strptime(curr_day, dtformat) + timedelta(hours=-12)
        else:
            now = datetime.strptime(curr_day, dtformat) + timedelta(days=-1)
        curr_day = now.strftime(dtformat)
    return pd.concat(dfs).reset_index(drop=True)

def nlp_prediction(ticker):
    # load nlp model
    fp_nlp = open('./models/nlp_model.pkl','rb')
    pickled_model = pickle.load(fp_nlp)

    # vectorizer
    fp_vect = open('./models/vectorizer.pkl','rb')
    vectorizer = pickle.load(fp_vect)

    result = pickled_model.predict(vectorizer.transform(get_tweets(ticker)['tweet']))
    perc_positive = round(sum(result)/len(result)*100,3)
    return perc_positive