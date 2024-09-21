import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set up the app title and input for stock ID
st.title("Stock Market Forecasting App")
st.title("Prediction")

# Input field for the user to enter a stock symbol (default is GOOG)
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set up the start and end dates for fetching historical data (last 20 years)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetch historical stock data using yfinance
google_data = yf.download(stock, start, end)

# Load the pre-trained LSTM model for prediction
model = load_model("Latest_stock_price_model.keras")

# Display the fetched stock data
st.subheader("Stock Data")
st.write(google_data)

# Split the data for training and testing (70% training, 30% testing)
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')  # Plot rolling average or any other values
    plt.plot(full_data.Close, 'b')  # Plot original closing price
    if extra_data:  # Option to plot additional data like a second moving average
        plt.plot(extra_dataset)
    return fig

# Plot Moving Averages (MA) for different periods
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale the test data for input to the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare the input data using the sliding window approach (100 days lookback)
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])  # Append the last 100 data points
    y_data.append(scaled_data[i])          # Append the target (next day)

x_data, y_data = np.array(x_data), np.array(y_data)

# Use the LSTM model to make predictions
predictions = model.predict(x_data)

# Inverse transform the predictions and actual values to get real prices
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame to compare original vs predicted values
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]  # Use the corresponding date index
)

# Display the original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot the original vs predicted values
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Forecast section title
st.title("Forecast")

# Input field for the user to specify the number of days to forecast
days = st.text_input("Enter number of days to forecast within a Year", 30)

# Convert input days to integer, and validate the input
if days.isdigit():
    days = int(days)  # Convert to integer if valid
else:
    st.error("Please enter a valid number.")

# Prepare the last 730 days as input for the forecasting model
last_730_days = scaled_data[-730:]
last_730_days = np.reshape(last_730_days, (1, last_730_days.shape[0], 1))

# Forecasting function for the specified number of days
def predict_days(model, last_730_days, days=days):
    pred_list = []
    input_sequence = last_730_days
    for _ in range(days):
        next_pred = model.predict(input_sequence, verbose=0)  # Predict the next day
        pred_list.append(next_pred[0][0])

        # Update the input sequence with the new prediction, discarding the oldest value
        input_sequence = np.append(input_sequence[:, 1:, :], [[next_pred[0]]], axis=1)

    return pred_list

# Make predictions for the next N days
next_days_scaled = predict_days(model, last_730_days)

# Inverse transform the predictions to get actual prices
next_days = scaler.inverse_transform(np.array(next_days_scaled).reshape(-1, 1))

# Create a date range for the forecast
start_date = datetime.now()
date_range = pd.date_range(start=start_date, periods=days, freq='B').date  # Business days

# Ensure forecasted prices match the date range
if len(date_range) != len(next_days):
    raise ValueError("Length of forecast prices must match the number of business days.")

# Create DataFrame to hold forecasted prices
df = pd.DataFrame(data=next_days, index=date_range, columns=['forecast_price'])

# Filter recent data for comparison with forecast
df_filtered = google_data.loc[google_data.index >= '2023-07-01', ['Adj Close']]

# Plot the forecast alongside actual adjusted close prices
st.subheader(f'Forecasting for {days} days')
fig = plt.figure(figsize=(15, 6))
plt.plot(df_filtered.index, df_filtered['Adj Close'], label='Adjusted Close')
plt.plot(df.index, df['forecast_price'], label='Forecasted Price')
plt.title('Adjusted Close Prices and Forecasted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
st.pyplot(fig)

# Add a disclaimer about the forecast accuracy
st.markdown("""
## Disclaimer
The information provided by this app is for educational purposes only and should not be considered as financial advice. The forecasts generated by the model are based on historical data and are not guaranteed to predict future stock prices accurately. Please do your own research and consult with a financial advisor before making any investment decisions.
""")
