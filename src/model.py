import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
import plotly.express as px
import datetime as dt
from datetime import timedelta, date

#Just to get a range for data, subject to change
start_date = date.today() - timedelta(days=365*9)
end_date = date.today() + timedelta(days=365*1)

#Basic data to just get a basic database
df = yf.download('AAPL', start=start_date, end=end_date)

df = df[['Adj Close']]
df['ma_20'] = df['Adj Close'].rolling(window=20).mean()
df['ma_50'] = df['Adj Close'].rolling(window=50).mean()

plot_data = df.loc[start_date:end_date]
fig = px.line(
    data_frame=plot_data,
    x=plot_data.index,
    y=['Adj Close', 'ma_20', 'ma_50'])

fig.show()