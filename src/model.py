import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
import plotly.express as px
import datetime as dt
from datetime import timedelta, date
import sklearn.metrics as mt
from sklearn.metrics import mean_absolute_error
from arch import arch_model

#Just to get a range for data, subject to change
start_date = '2021-07-23'
end_date = '2024-02-07'

#Basic data to just get a basic database
df = pd.read_csv('data\zomato.csv')

#Preparing data
df = df[['Adj Close']]
df['ma_30'] = df['Adj Close'].rolling(window=30).mean()
df['ma_90'] = df['Adj Close'].rolling(window=90).mean()
df['daily returns'] = df['Adj Close'].pct_change()*100

#Building Model
df2 = df[['daily returns']]
cutoff = '2023-02-07'
y_train = df2.loc[:cutoff, :]
y_test = df2.loc[cutoff:, :]

if y_train.isnull().values.any() or np.isinf(y_train).any():
    y_train = y_train.dropna()
    y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()

y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)

model = arch_model(y_train, p=1, q=0, rescale=False).fit()

print(model.forecast(horizon=1, reindex=False).mean)

#Plotting data
""" plot_data = df.loc[start_date:end_date]
fig = px.line(
    data_frame=plot_data,
    x=plot_data.index,
    y=['ma_30', 'ma_90'])
fig.show()

plt.Figure(figsize=(10,25))
fig2 = px.line(
    data_frame=df, 
    x=df.index, 
    y=['daily returns']
)
fig2.show() """