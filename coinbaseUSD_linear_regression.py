from pandas import DataFrame, read_csv, to_datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import datetime
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


# df = DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

X = df.index # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Close']
print("asda")
print(X.values.reshape(1, -1))
print("asda")
print(Y)
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X.values.reshape(1, -1), Y.values.reshape(1, -1))

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
base = datetime.datetime(2020, 1, 1)
date_list = [base + datetime.timedelta(days=x) for x in range(365)]
#Year = 2021
#Month = 1

#print ('Predicted Stock Index Price: \n', regr.predict([[Year,Month]]))
predictions = regr.predict(date_list)
print('Long term predictions:', predictions)

"""
names = time_pairs
values = predictions
fig, ax = plt.subplots(1)
ax.plot([datetime.date(y, m, 1) for [y, m] in time_pairs], predictions, label='Stock Price')
ax.set_ylim(ymin=0)
ax.legend()
plt.show()


# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)
"""
