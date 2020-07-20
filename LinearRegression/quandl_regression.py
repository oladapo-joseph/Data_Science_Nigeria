import quandl as Qaund
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle


# style.use('ggplot')

df = Qaund.get("WIKI/GOOGL")
#there is a limit to the number of pulls/day without an API key

df['HL_Change'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] *100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0
df = df[['Adj. Open','Adj. Close', 'HL_Change', 'PCT_Change', 'Adj. Volume',]]
#creates a new dataframe that contains what i need


#drops all lines that is empty or not available
df.dropna(inplace=True)

forecast_range = int(.01*len(df))
df['Prediction'] = df['Adj. Close'].shift(-forecast_range)
#shifts the adj. close up by forecast_range and stores in prediction

X = np.array(df.drop(['Prediction'],1))
#drops the prediction range, to create new df
X = X[:-forecast_range]
X = preprocessing.scale(X)
X_to_predict = X[-forecast_range:]
print(X_to_predict)

df.dropna(inplace=True)
Y = np.array(df['Prediction'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

# LinearRegression Algorithm
classifier  = LinearRegression()                    #the algorithm that compares
classifier.fit(X_train, Y_train)                    #fit implies training data

"""
    to save time the algorithmcan with the pre trained data can be saved to used
     only on similar datasets, you will need the pickle module to save it.
     uncomment the below to see.
     once saved, you can delete the algorithm initialisation above
"""
# with open('LinearRegression.pickle', 'wb') as f:
#     pickle.dump(classifier, f)
#
# pick = open('LinearRegression.pickle', 'rb')
# classifier = pickle.load(pick)


accuracy = classifier.score(X_test, Y_test)                       #the accuracy of the prediction, the score implies the test
print(accuracy)
forecast = classifier.predict(X_to_predict)
print(forecast)

#SVM algorithm
classifier2 = svm.SVR()
print(classifier2.fit(X_train,Y_train))
print(classifier2.score(X_test,Y_test))


#to plot the graph of the output
#assign the dates of the new values, the date is the index of the original data
last_date = df.iloc[-2].name
# this returns the last date

#the next date we want is a day after, converting the date to timestamp in secs
# and adding a new day which implies 86400 secs
last_date = last_date.timestamp()

#to initialise an empty label 
df['Forecast'] = np.nan

for i in forecast:
    next_Date = datetime.datetime.fromtimestamp(last_date)
    df.loc[next_Date] =[np.nan for w  in range(len(df.columns)-1)] + [i]
    last_date += 86400

df['Adj. Close'].plot()
df.Forecast.plot()
plt.legend()
plt.show()
