import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import timedelta, date
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from arch import arch_model

# Just to get a range for data, subject to change
start_date = date.today() - timedelta(days=365*9)
end_date = date.today() + timedelta(days=365*1)

# Basic data to just get a basic database
df = yf.download('AAPL', start=start_date, end=end_date)

# Preparing data
df = df[['Adj Close']]
df['ma_30'] = df['Adj Close'].rolling(window=30).mean()
df['ma_90'] = df['Adj Close'].rolling(window=60).mean()
df['daily returns'] = df['Adj Close'].pct_change()*100

# Building Model
df2 = df[['daily returns']]
cutoff = date.today() - timedelta(days=365*4)
y_train = df2.loc[:cutoff, :]

if y_train.isnull().values.any() or np.isinf(y_train).any():
    y_train = y_train.dropna()
    y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()

y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)

model = arch_model(y_train, p=1, q=0, rescale=False).fit()

# Plotting data
plot_data = df.loc[start_date:end_date]
fig = px.line(
    data_frame=plot_data,
    x=plot_data.index,
    y=['ma_30', 'ma_90'])

fig2 = px.line(
    data_frame=df, 
    x=df.index, 
    y=['daily returns']
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div(children=[
    html.H1(children='Data Analysis of Zomato stock'),

    html.Div(children='''
        MA_30 and MA_90 Plot
    '''),

    dcc.Graph(
        id='ma-graph',
        figure=fig
    ),

    html.Div(children='''
        Daily Returns Plot
    '''),

    dcc.Graph(
        id='returns-graph',
        figure=fig2
    )
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)