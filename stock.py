import base64
import io
import requests
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.dates as mdates
from datetime import datetime
from pandas import date_range, to_datetime
import numpy as np
import pandas as pd
from modules.logger import logger
from modules.cache import cache
from modules.dataprocess import process_stock_data_for_training

from config import Config
API_KEY = Config.API_KEY


def get_stock_data(symbol):
    # Check cache first
    cached_data = cache.load_data(f"{symbol}_data")
    if cached_data:
        logger.debug("Data loaded from cache for symbol: {}".format(symbol))
        return cached_data

    # API call
    api_data = fetch_data_from_api(symbol)  # Define this function as needed
    cache.save_data(f"{symbol}_data", api_data)
    logger.debug("Data loaded from API for symbol: {}".format(symbol))
    return api_data


def fetch_data_from_api(symbol):
    api_key = API_KEY
    URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&interval=5min&apikey={api_key}"
    response = requests.get(URL)
    data = response.json()
    if 'Error Message' in data or 'Information' in data or 'Note' in data:
        return {'error': 'API request limit reached or invalid API call'}
    return data


def get_stock_overview(symbol):
    # Similar caching mechanism as get_stock_data
    cached_overview = cache.load_data(f"{symbol}_overview")
    if cached_overview:
        logger.debug("Overview data loaded from cache for symbol: {}".format(symbol))
        return cached_overview

    overview_data = fetch_overview_from_api(symbol)  # Define this function as needed
    cache.save_data(f"{symbol}_overview", overview_data)
    logger.debug("Overview data loaded from API for symbol: {}".format(symbol))
    return overview_data


def fetch_overview_from_api(symbol):
    api_key = API_KEY
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'Error Message' in data or 'Information' in data or 'Note' in data:
        return {'error': 'API request limit reached or invalid API call'}
    return data


def make_prediction(model, last_30_days):
    """
    Make a prediction based on the last 30 days of data.

    Args:
    model (tf.keras.Model): Trained TensorFlow model to use for predictions.
    last_30_days (np.array): Array of the last 30 days of prices, shape (30,).

    Returns:
    np.array: Predicted prices for the next horizon days.
    """
    # Ensure data is in the correct shape [batch_size, window_size, num_features]
    last_30_days = last_30_days.reshape((1, -1, 1))  # Assuming only one feature, reshape accordingly

    # Make predictions
    predictions = model.predict(last_30_days)
    return predictions.flatten()  # Flatten to simplify usage


class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = get_stock_data(symbol)
        self.overview = get_stock_overview(symbol)
        self.model = load_model('models/model_25s_7d.h5')

    def plot_stock(self, days=60):  # Set default to 30 days for a month of data
        key = 'Time Series (Daily)'  # Adjusted for daily data keys
        dates = list(self.data[key].keys())[:days]  # Fetch the latest 'days' data points
        prices = [float(self.data[key][date]['4. close']) for date in dates]
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Convert dates to datetime objects

        plt.figure(figsize=(10, 5))
        plt.plot(dates, prices, label='Closing Price')
        plt.title(f'Stock Prices for {self.symbol} - Last {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Format x-axis date labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show a tick every 5 days
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        logger.info("Stock plot for {} generated successfully.".format(self.symbol))
        return plot_url

    def retrain_model(self):
        horizon = 7  # Number of days to predict
        window_size = 30  # Number of days to look at in the past

        new_model = extend_model(self.model, horizon=horizon)

        # Process the data for training
        train_data = self.data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(train_data, orient='index')
        df = df.apply(pd.to_numeric)
        df.index = pd.to_datetime(df.index)
        close_prices = df['4. close']
        close_prices = close_prices.sort_index()[30:]  # do not use the last 30 days

        # create windows and labels and split the data
        train_windows, test_windows, train_labels, test_labels = process_stock_data_for_training(
            close_prices.values, window_size=window_size, horizon=horizon)

        # Train the model
        new_model.fit(x=train_windows, y=train_labels,
                      epochs=100, batch_size=32, verbose=0,
                      validation_data=(test_windows, test_labels))

        # reassign the new model to the class
        logger.info("Model retrained successfully.")
        self.model = new_model

    def predict_prices_7days(self):
        key = 'Time Series (Daily)'
        prices = [float(self.data[key][date]['4. close']) for date in sorted(self.data[key].keys())[-30:]]
        # Assume model expects input shape [1, 30, 1] for one feature per day
        prices = np.array(prices).reshape(1, -1, 1)
        predictions = self.model.predict(prices)
        logger.info("Stock price predictions for the next 7 days for {} generated successfully.".format(self.symbol))
        return predictions.flatten().tolist()

    def plot_predictions(self, retrain=False):
        key = 'Time Series (Daily)'
        # Ensure dates are sorted in ascending order
        sorted_dates = sorted(self.data[key].keys())
        last_30_dates = sorted_dates[-30:]  # Get the actual last 30 days
        historical_data = [float(self.data[key][date]['4. close']) for date in last_30_dates]
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in last_30_dates]  # Convert dates to datetime objects

        # Retrain the model if specified
        if retrain:
            logger.debug("Retraining model...")
            self.retrain_model()

        predictions = self.predict_prices_7days()  # Get 7-day future predictions

        plt.figure(figsize=(12, 6))
        # Calculate the number of days in historical data
        total_days = len(historical_data)
        future_days = len(predictions)

        last_date = to_datetime(dates[-1])
        prediction_dates = date_range(last_date, periods=future_days + 1, freq='D')[1:]

        # Append prediction dates to the end of the historical dates
        full_dates = dates + list(prediction_dates)

        # Plot settings
        plt.plot(full_dates[:total_days], historical_data, label="Last 30 Days", color='orange')
        plt.plot(full_dates[total_days - 1:total_days + future_days],
                 np.concatenate(([historical_data[-1]], predictions)), label="Predictions", color='red')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.title(f"Stock Price Predictions for {self.symbol} for the Next 7 Days")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.gcf().autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        logger.info("Stock price predictions plot for {} generated successfully.".format(self.symbol))
        return plot_url


def extend_model(model, horizon=7):
    """
    Extends a model by adding a new output layer for predicting horizon days into the future.
    """
    # Get the shape of the input layer of the model
    new_input = Input(shape=model.input_shape[1:])

    # Pass the input through the model
    x = new_input
    for layer in model.layers[:-1]:
        x = layer(x)
        layer.trainable = False

    # Add a new dense layer with ReLU activation
    x = Dense(64, activation='relu')(x)

    # Add the output layer with linear activation for predicting horizon days
    new_output = Dense(horizon, activation='linear')(x)

    # Create a new model with the input and output
    new_model = Model(new_input, new_output, name='new_model')

    # Compile the new model
    new_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    return new_model