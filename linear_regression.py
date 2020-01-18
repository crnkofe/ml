from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import datetime

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
                                }

df = DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

print (df)

"""
# scatter plot showing a linear relation between variables
# time seems to show a linear relationship with the stock price
plt.scatter(df['Year'], df['Stock_Index_Price'], color='red')
plt.title('Year Vs Interest Rate', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Month'], df['Stock_Index_Price'], color='green')
plt.title('Month Vs Unemployment Rate', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()
"""

df = DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

X = df[['Year','Month']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Stock_Index_Price']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
Year = 2020
Month = 1

print ('Predicted Stock Index Price: \n', regr.predict([[Year,Month]]))
time_pairs = []
for year in range(2018, 2021):
    for month in range(1, 12):
        time_pairs.append([year, month])
predictions = regr.predict(time_pairs)
print('Long term predictions:', predictions)

names = time_pairs
values = predictions
fig, ax = plt.subplots(1)
ax.plot([datetime.date(y, m, 1) for [y, m] in time_pairs], predictions, label='Stock Price')
ax.set_ylim(ymin=0)
ax.legend()
plt.show()
"""
# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)
"""
