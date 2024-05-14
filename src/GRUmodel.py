import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
from datetime import date

# Function to prep csv file
def prepare_data():
    # Read stock price data
    url = 'https://raw.githubusercontent.com/Tiaan-Botes/CYO_Project_Group_E/52663ba4e6833e232f1bbd7a0ab48edb23f52b91/data/data.csv'
    data = pd.read_csv(url)

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)
    data.drop(columns=['Unnamed: 0', 'Year', 'Month', 'Day', 'Weekday'], inplace=True)

    data['ma_30'] = data['Adj Close'].rolling(window=30).mean()
    data['ma_90'] = data['Adj Close'].rolling(window=90).mean()
    
    data['daily_returns'] = data['Adj Close'].pct_change()*100
    data.dropna(inplace=True)

    return data

df = prepare_data()
df.info()
#print(df.head())

current_date = date.today().strftime('%Y-%m-%d')
plot_data = df.loc['2023-01-01':current_date]

# Plotting stock price along with moving averages
plt.figure(figsize=(20, 25))
fig = px.line(
    data_frame=plot_data, 
    x=plot_data.index, 
    y=['Adj Close', 'ma_30', 'ma_90']
)
#fig.show()

# Plotting daily returns
plt.figure(figsize=(10, 25))
fig = px.line(
    data_frame=df, 
    x=df.index, 
    y=['daily_returns']
)
#fig.show()

# Function to prepare data for GRU model
def prepare_gru_data():
    data = prepare_data()  
    dataset = data[['Adj Close']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset

# Function to build GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Function to train GRU model
def train_gru_model(dataset):
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])

        return np.array(X), np.array(Y)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = build_gru_model((1, look_back))
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    return model, testX, testY

# Function to make predictions using GRU model
def predict_gru_model(model, testX):
    return model.predict(testX)

# Preparing data
gru_dataset = prepare_gru_data()

# Training 
gru_model, testX, testY = train_gru_model(gru_dataset)

# predictions
gru_predictions = predict_gru_model(gru_model, testX)

# Calculating Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(testY, gru_predictions))
print(f'Root Mean Squared Error: {rmse}')

#calculate growh of stock 
def calculate_growth():
    # Read the data from data.csv
    data = pd.read_csv('data.csv')

    # Calculate the growth based on the period
    day_growth = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[-2]) / data['Adj Close'].iloc[-2] * 100
    month_growth = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[-30]) / data['Adj Close'].iloc[-30] * 100
    year_growth = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[-365]) / data['Adj Close'].iloc[-365] * 100

    return day_growth, month_growth, year_growth

def sudden_drop():
    # Read the data from data.csv
    data = pd.read_csv('data.csv')

    # Get the current and previous prices
    current_price = data[data['Date'] == max(data['Date'])]['Adj Close'].values[0]
    previous_price = data[data['Date'] == max(data['Date']) - pd.Timedelta(days=1)]['Adj Close'].values[0]

    # Check if the stock has dropped more than 10%
    if (previous_price - current_price) / previous_price > 0.2:
        return f"Alert: Zomato has dropped more than 20% from the previous day!"
    else:
        return "Stock still normal"


# Plotting actual vs. predicted values
plt.plot(testY, label='Actual')
plt.plot(gru_predictions, label='Predicted')
plt.legend()
#plt.show()
