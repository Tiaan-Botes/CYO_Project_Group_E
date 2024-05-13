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
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#Just to get a range for data, subject to change
start_date = '2021-07-23'
end_date = '2025-02-07'

#Basic data to just get a basic database
parent_dir = os.path.dirname(os.getcwd())

# Construct the path to the zomato.csv file inside the data directory
file_path = os.path.join(parent_dir, "data", "data.csv")

# Read the CSV file
df = pd.read_csv('data/data.csv')

#Preparing data
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

df.drop(columns=['Unnamed: 0', 'Year', 'Month', 'Day', 'Weekday'], inplace=True)
df['ma_30'] = df['Adj Close'].rolling(window=30).mean()
df['ma_90'] = df['Adj Close'].rolling(window=90).mean()
df['daily returns'] = df['Adj Close'].pct_change()*100

#Building Model
df2 = df[['daily returns']]
cutoff = end_date
y_train = df2.loc[:cutoff, :]
y_test = df2.loc[cutoff:, :]

if y_train.isnull().values.any() or np.isinf(y_train).any():
    y_train = y_train.dropna()
    y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()

y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)

model = arch_model(y_train, p=1, q=0, rescale=False).fit()

print(model.forecast(horizon=3, reindex=False)._mean)

#Plotting data
plot_data = df.loc[start_date:end_date]
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
fig2.show()

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