import pandas as pd
import requests, pickle, datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import style
from sklearn import preprocessing, svm



def get_data(coin, exchange= 'bitfimex', after='2018-04-01'):
    url = 'https://api.cryptowat.ch/markets/{}/{}usd/ohlc'.format(exchange, coin)
    resp = requests.get(url, params={'periods': '3600',
                                   'after': str(int(pd.Timestamp(after).timestamp()))
                                   })
#     resp.raise_for_status()
    data = resp.json()
    #print(data)
    df = pd.DataFrame(data['result']['3600'],
                        columns= ['CloseTime', 'OpenPrice','HighPrice', 'LowPrice', 'ClosePrice', 'Volume','NA'])
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit = 's')
    df.set_index('CloseTime', inplace= True)
    return (df)

today = str(datetime.date.today())
btc = get_data('btc', 'bitstamp')
print(btc.head())

true_data = btc['2020-05-01': today]                                           #data we wish to compare
forecast_range = len(true_data)

btc['HL_Change'] = (btc.HighPrice - btc.LowPrice)/ btc.LowPrice * 100.0
btc['OC_Change'] = (btc.ClosePrice - btc.OpenPrice)/ btc.OpenPrice * 100.0

#data we will be walking with
btc = btc[['ClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'Volume']]
#btc.dropna(inplace=True)

btc['Prediction'] = btc.ClosePrice.shift(-forecast_range)

X = np.array(btc.drop('Prediction',1))
X = X[:-forecast_range]

X = preprocessing.scale(X)
X_to_predict = X[-forecast_range:]

btc.dropna(inplace = True)
Y = np.array(btc.Prediction)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.5)

#i already trained a model stored in this wprk path
# pickled = open('LinearRegression.pickle', 'rb')
# clf = pickle.load(pickled)
classifier = LinearRegression()
classifier.fit(X_train, Y_train)
accuracy = classifier.score(X_test, Y_test)
print(accuracy)
forecast = classifier.predict(X_to_predict)
print(forecast)
btc['forecast'] = np.nan
btc.forecast.loc[-forecast_range:]= forecast

btc['ClosePrice'].plot()
btc.forecast.plot()
plt.legend()
plt.show()
