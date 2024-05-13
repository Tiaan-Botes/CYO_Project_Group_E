import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error

# Function to prepare stock price data
def prepare_data():
    url = 'https://raw.githubusercontent.com/Tiaan-Botes/CYO_Project_Group_E/52663ba4e6833e232f1bbd7a0ab48edb23f52b91/data/data.csv'
    data = pd.read_csv(url)

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort data by date
    data.sort_values('Date', inplace=True)

    # Set 'Date' column as index
    data.set_index('Date', inplace=True)

    # Drop unnecessary columns
    data.drop(columns=['Unnamed: 0', 'Year', 'Month', 'Day', 'Weekday'], inplace=True)

    # Calculating 20-day and 50-day moving averages
    data['ma_20'] = data['Adj Close'].rolling(window=20).mean()
    data['ma_50'] = data['Adj Close'].rolling(window=50).mean()
    
    # Calculating daily returns
    data['daily_returns'] = data['Adj Close'].pct_change()*100
    data.dropna(inplace=True)

    return data

df = prepare_data()
df.info()
print(df.head())

plot_data = df.loc['2023-01-01':'2023-12-31']

# Plotting stock price along with moving averages
plt.figure(figsize=(20, 25))
fig = px.line(
    data_frame=plot_data, 
    x=plot_data.index, 
    y=['Adj Close', 'ma_20', 'ma_50']
)
fig.show()

# Plotting daily returns
plt.figure(figsize=(10, 25))
fig = px.line(
    data_frame=df, 
    x=df.index, 
    y=['daily_returns']
)
fig.show()

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

# Plotting actual vs. predicted values
plt.plot(testY, label='Actual')
plt.plot(gru_predictions, label='Predicted')
plt.legend()
plt.show()
