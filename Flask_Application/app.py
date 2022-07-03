# Flask Libraries
from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

import pickle
from datetime import timezone, date, timedelta
from datetime import datetime as dt

# from stockinfo import *
from model_functions import *
from prepsend import *
from ensemble_model_functions import *
from recommendation_functions import *
from nlp_functions import *

app = Flask(__name__, static_url_path='/static')

app.config.update(
    # DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True
)

### Constants
today = dt.today()
date_today = str(today.year) + '-' + str(today.month) + '-' + str(today.day)

# open page
@app.route("/")
def index():
	print('hello')
	return render_template( 'index.html', date=date_today)

@app.route("/predict", methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':
		######################### 
		### GET TICKER
		######################### 

		ticker = request.form['stock-tickers']

		######################### 
		### GET STOCK INFO
		######################### 
		yf_company = yf.Ticker(ticker)
		df_yf = yf_company.history(period='3y')

		######################### 
		### FORECAST FROM ARIMA
		######################### 
		pred_dates, val_forecast, price_next_day, val_rmse, val_mape = predictStockPrice(df_yf)
		val_forecast = list(val_forecast['open_pred'])

		arima_dict = {
			"pred_dates":pred_dates,
			"val_forecast": val_forecast,
			"price_next_day": round(price_next_day[0],2),
			"val_rmse": round(val_rmse,3),
			"val_mape": round(val_mape,3)
		}

		######################### 
		### ENSEMBLE MODEL FORECAST
		######################### 

		# prep dataset
		prd = -int(len(df_yf)*0.33)
		arima_join = [None]*(len(df_yf)+prd) + val_forecast
		df_ensemble = df_yf.copy()
		df_ensemble['Arima_Pred'] = arima_join

		assert(len(arima_join) == len(df_ensemble))
		df_ensemble = df_ensemble[df_ensemble['Arima_Pred'].notnull()]
		df_ensemble['Company'] = ticker

		pred_dates, val_forecast, price_next_day, val_rmse, val_mape = predict_using_ensemble(yf_company, df_ensemble)

		ensemble_dict = {
			"pred_dates": pred_dates,
			"val_forecast": [float(x) for x in val_forecast],
			"price_next_day": round(float(price_next_day),2),
			"val_rmse": round(float(val_rmse),3),
			"val_mape": round(float(val_mape),3)
		}

		#########################
		### GET TWEETS AND SENTIMENT
		#########################

		nlp_sentiment = nlp_prediction(ticker)
		print('nlp sentiment:', nlp_sentiment)

		#########################
		### RECOMMENDATION ALG.
		#########################
		df_yf = yf_company.history(period='3y')
		pred_dates, validation_truth, validation_pred, next_day_rec, accuracy = get_recommendation(df_yf)
		rec_history_dict = {
			'Dates': pred_dates,
			'Open': validation_truth
		}
		rec_val_dict = {
			"pred_dates": pred_dates,
			"val_forecast": validation_pred,
			"next_day_rec": next_day_rec,
			"accuracy": round(accuracy,3)
		}

		#########################
		### SEND TO FRONTEND
		#########################
		send_all = sendDict(date_today, yf_company, df_yf, ticker, arima_dict, ensemble_dict, rec_history_dict, rec_val_dict, nlp_sentiment)

		# direct to new page
		return render_template('analysis.html', **send_all)

	else:
	    return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=False,threaded=False)