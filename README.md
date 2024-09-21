# Stock Price Prediction with LSTM

This project forecasts stock prices using a Long Short-Term Memory (LSTM) model built with Keras and TensorFlow. It provides an interactive Streamlit-based frontend that allows users to visualize historical stock prices and forecast stock trends for any number of days (up to a year).

### Project Overview

Stock price prediction is a challenging task due to market volatility and noise. This project leverages deep learning techniques, particularly the LSTM architecture, to predict future stock prices based on past price data. Users can choose how many days in the future (within a year) they want to forecast.

The application consists of two main components:
- **Backend**: A machine learning model built with Keras and TensorFlow to predict future stock prices.
- **Frontend**: A Streamlit-based web app to visualize stock price trends and forecast future prices.

### Features

- **Historical Data Input**: Uses historical stock price data for training the model.
- **LSTM Model**: The model learns from past data using a sliding window approach to predict the next stock price.
- **Streamlit Frontend**: An interactive dashboard that allows users to:
  - Visualize historical stock prices
  - Predict stock prices for any number of days (up to 365 days)
  - Select different stock symbols (if integrated with real stock data)

### Setup Instructions

#### Prerequisites

- Python 3.x
- Libraries:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Streamlit
  - yfinance (optional, for fetching real-time stock data)

#### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prapul007/stock-market-forecasting-app.git
   cd stock-market-forecasting-app

2. **Install the dependencies**:
    ```bash
   pip install -r requirements.txt
   
3. **Run the Streamlit app**:   
    ```bash
   streamlit run web_stock_price_predictor.py

4. **Navigate to http://localhost:8501/ in your browser to use the app**.


#### Project Structure
```bash
.
├── web_stock_price_predictor.py         # Streamlit frontend code
├── stock_price.ipynb                    # Jupyter notebook containing LSTM model training code
├── model.h5                             # Saved LSTM model (if already trained)
├── requirements.txt                     # List of required Python libraries
└── data                                 # Stock price data from yfinance
├── README.md                            # Project documentation (this file)
```

## web_stock_price_predictor.py
This file contains the Streamlit code for the frontend. It visualizes the stock price predictions and allows users to:

* Upload historical stock data
* Choose the number of days (up to 365) for future price predictions
* Visualize the historical data and forecasted prices interactively


## stock_price.ipynb
This Jupyter notebook includes:

* Data preprocessing steps (e.g., scaling stock prices)
* The creation of a sliding window to prepare data for the LSTM model
* Training the LSTM model to predict future stock prices
* Saving the trained model for use in the Streamlit app

### How It Works
1. **Data Preprocessing**:

* The stock price data is normalized using MinMaxScaler to prepare it for the LSTM model.
* A sliding window of past data points (e.g., the previous 100 days) is used to predict the next stock price.
2. **Model Training**:

* The LSTM model is trained on the historical stock data.
* After training, the model is saved for reuse without retraining.
3. **Prediction**:

* Users can forecast stock prices for any number of days (up to 365).
* The LSTM model uses the most recent stock prices to predict future values.
4. **Frontend Visualization**:

* The Streamlit app visualizes both historical stock prices and the forecast for the user-selected number of days.

## Contributing
Feel free to submit a pull request if you'd like to contribute to the project. Please ensure that your code follows the PEP 8 style guide.




