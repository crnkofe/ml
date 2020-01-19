from pandas import DataFrame, read_csv, to_datetime
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing, model_selection
import statsmodels.api as sm
import datetime
import time
import tarfile

def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

FILENAME = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.tar.gz"
with tarfile.open(FILENAME, "r:*") as tar:
    csv_path = tar.getnames()[0]
    df = read_csv(
        tar.extractfile(csv_path),
        parse_dates=True, date_parser=dateparse, index_col='Timestamp',
        header=0, sep=",")

# there are a lot of NaN's in price values
df.dropna(inplace=True)

print(type(to_datetime(df.index)))
# add time columns
df['time_year'] = to_datetime(df.index).year
df['time_month'] = to_datetime(df.index).month
df['time_day'] = to_datetime(df.index).day
df['time_minute'] = to_datetime(df.index).minute

print (df)

"""
# plot BTC Close price and volume
dates = to_datetime(df.index)#df.index.values.tolist()
print(to_datetime(df.index))
close_prices = df["Close"].tolist()
volume_btc = df["Volume_(BTC)"].tolist()
fig, ax = plt.subplots(1)
ax.plot(dates, close_prices, label='Close')
ax.plot(dates, volume_btc, label='Volume_(BTC)')
ax.set_ylim(ymin=0)
ax.legend()
plt.show()
"""

"""
# scatter plot showing a linear relation between variables
# volume and close price don't seem to have any linear relation
# which won't stop me from doing a predictor based on it anyway
plt.scatter(df['Volume_(BTC)'], df['Close'], color='red')
plt.title('Volume vs. Close', fontsize=14)
plt.xlabel('Volume_(BTC)', fontsize=14)
plt.ylabel('Close', fontsize=14)
plt.grid(True)
plt.show()
"""

## first attempt at making a simple line
# this is pointless as basically all meaningful features are discarded
# but works nontheless
X = df[['time_year', 'time_month', 'time_day', 'time_minute']]
Y = df['Close']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

accuracy = regr.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
base = datetime.datetime(2015, 1, 1)
date_list = [ base + datetime.timedelta(days=x) for x in range(365*5)]
unix_time_list = [[t.year, t.month, t.day, t.minute] for t in date_list]
predictions = regr.predict(unix_time_list)
print('Long term predictions:', predictions)

# create a 14 day forecast feature and a model

## PLOT IT!
values = predictions
fig, ax = plt.subplots(1)
ax.plot(date_list, predictions, label='BTC Price Interpolation')

btc_dates = to_datetime(df.index) #df.index.values.tolist()
close_prices = df["Close"].tolist()
ax.plot(btc_dates, close_prices, label='BTC Actual Close Prices')

ax.set_ylim(ymin=0, ymax=20000)
ax.legend()
plt.show()
