import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
import plotly.express as px
import datetime as dt
from datetime import timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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

mae = mean_absolute_error(y_train, y_pred_baseline)
mse = mean_squared_error(y_train, y_pred_baseline)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

day = 1
week = 7
month = 31
year = 366

#Predicting the future volatility
forecast_day = model.forecast(horizon=day)
next_day = np.sqrt(forecast_day.variance.iloc[-1].values[-1])

forecast_week = model.forecast(horizon=week)
next_week = np.sqrt(forecast_week.variance.iloc[-1].values[-1])

forecast_month = model.forecast(horizon=month)
next_month = np.sqrt(forecast_month.variance.iloc[-1].values[-1])

forecast_year = model.forecast(horizon=year)
next_year = np.sqrt(forecast_year.variance.iloc[-1].values[-1])

print("Next day's volatility forecast:", next_day)
print("Next week's volatility forecast:", next_week)
print("Next month's volatility forecast:", next_month)
print("Next year's volatility forecast:", next_year)

#Predict future returns
forecast_day = model.forecast(horizon=day)
next_day_return = forecast_day.mean.iloc[-1].values[-1]
previous_close = df['Adj Close'].iloc[-1]
next_day_predicted_price = previous_close * (1 + next_day_return / 100)

forecast_week = model.forecast(horizon=week)
next_week_return = forecast_week.mean.iloc[-1].values[-1]
next_week_predicted_price = previous_close * (1 + next_week_return / 100)

forecast_month = model.forecast(horizon=month)
next_month_return = forecast_month.mean.iloc[-1].values[-1]
next_month_predicted_price = previous_close * (1 + next_month_return / 100)

forecast_year = model.forecast(horizon=year)
next_year_return = forecast_year.mean.iloc[-1].values[-1]
next_year_predicted_price = previous_close * (1 + next_year_return / 100)

# Printing the results
print("Next day's predicted price:", next_day_predicted_price)
print("Next week's predicted price:", next_week_predicted_price)
print("Next month's predicted price:", next_month_predicted_price)
print("Next year's predicted price:", next_year_predicted_price)

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