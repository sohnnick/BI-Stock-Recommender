import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone, date, timedelta

def sendDict(date_today, yf_company, df_yf, ticker, arima_dict, ensemble_dict, rec_history_dict, rec_val_dict, nlp_sentiment):

	yf_info = yf_company.info

	#########################
	### Historical Price JSON
	#########################
	send_historical_dict = {
		"Dates":[x.strftime("%Y-%m-%d") for x in list(df_yf.index)],
		"Open": list(df_yf['Open']),
		"Close": list(df_yf["Close"]),
		"Volume": list(df_yf["Volume"])
	}
	# send_historical_json = json.dumps(send_historical_dict, default=str)

	#########################
	### Historical Table Prep
	#########################
	print(len(send_historical_dict["Dates"]))
	print(len(send_historical_dict["Open"]))
	print(len(send_historical_dict["Close"]))
	print(len(send_historical_dict["Volume"]))

	historical_table = []
	keys_ = list(send_historical_dict.keys())
	for i in range(len(send_historical_dict["Dates"])):
		row_hist = []
		for key in keys_:
			row_hist.append(send_historical_dict[key][i])
		historical_table.append(row_hist)

	#########################
	### ARIMA Predictions JSON
	#########################
	# send_arima_json = json.dumps(arima_dict, default=str)

	#########################
	### Ensemble Predictions JSON
	#########################
	# send_ensemble_json = json.dumps(ensemble_dict, default=str)

	#########################
	### Summary Metrics
	#########################	
	if 'trailingPE' in yf_info:
		pe_ = round(yf_info['trailingPE'],2)
	else:
		pe_ = ''

	try:
		free_cash_flow_ = round(yf_info["freeCashflow"]/1000000000,2)
	except:
		free_cash_flow_ = ''
	
	summary_metrics = {
		"high_52w": yf_info['fiftyTwoWeekHigh'],
		"low_52w": yf_info['fiftyTwoWeekLow'],
		"change_52w": round(yf_info['52WeekChange']*100,2),
		"day_open": yf_info['open'],
		"day_low": yf_info['dayLow'],
		"day_high": yf_info['dayHigh'],
		"pe_ratio": pe_,
		"div_yield": round(yf_info['trailingAnnualDividendYield']*100, 2),
		"market_cap": round(yf_info['marketCap']/1000000000,2),
		"ebitda": round(yf_info["ebitda"]/1000000000,2),
		"eps": yf_info["trailingEps"],
		"revenue": round(yf_info["totalRevenue"]/1000000000,2),
		"shares_outstanding": round(yf_info["sharesOutstanding"]/1000000000,2),
		"gross_profit": round(yf_info["grossProfits"]/1000000000,2),
		"free_cash_flow": free_cash_flow_
	}
	# send_summary_json = json.dumps(summary_metrics)

	#########################
	### Company Information
	#########################
	company_info = {
		"ticker":ticker,
		"city": yf_info['city'],
		"country": yf_info['country'],
		"business_summary": yf_info['longBusinessSummary'],
		"industry": yf_info['industry'],
		"exchange": yf_info['exchange'],
		"logo_url": yf_info['logo_url'],
		"sector":yf_info['sector']
	}
	# send_info_json = json.dumps(company_info)

	print(company_info["industry"])

	#########################
	### News
	#########################
	yf_news = yf_company.news
	news_info = {'title':[], 'publisher':[], 'link':[], 'providerPublishTime':[]}
	for i in range(0,5):
		news_obj = yf_news[i]
		news_info['title'].append(news_obj['title'])
		news_info['publisher'].append(news_obj['publisher'])
		news_info['link'].append(news_obj['link'])
		news_info['providerPublishTime'].append( str(datetime.fromtimestamp(news_obj['providerPublishTime'])) )
	# news_info = json.dumps(news_info, default=str)

	#########################
	### Finalize JSON
	#########################
	send_all = {
		"date_today": date_today,
		"ticker": ticker, 
		"historical": send_historical_dict, 
		"arima": arima_dict, 
		"rec_historical": rec_history_dict,
		"rec": rec_val_dict,
		"ensemble": ensemble_dict,
		"news_info": news_info,
		"historical_table": historical_table,
		"twitter_sentiment": nlp_sentiment
	}

	send_all = {**send_all, **summary_metrics, **company_info}

	return send_all