import base64
import io
import requests
from flask import jsonify
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
# from tensorflow.keras.metrics import MeanAbsoluteError
# from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pandas import date_range, to_datetime
import numpy as np
import os
import pickle

API_KEY = 'NSQ25HG8ERO35TPU'


class Cache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_filename(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol}.pkl")

    def save_data(self, symbol, data):
        filename = self.get_filename(symbol)
        with open(filename, 'wb') as f:
            pickle.dump({
                'date': datetime.now(),
                'data': data
            }, f)

    def load_data(self, symbol):
        filename = self.get_filename(symbol)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                cached = pickle.load(f)
                # Check if the cache is still valid, let's say we refresh it every day
                if cached['date'].date() == datetime.now().date():
                    return cached['data']
                else:
                    # Delete old data if it's older than one day
                    os.remove(filename)
        return None

    def clear_cache(self):
        """ Clears all cached files if they are older than one day. """
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            with open(file_path, 'rb') as f:
                cached = pickle.load(f)
                if (datetime.now() - cached['date']) >= timedelta(days=1):
                    os.remove(file_path)


# Create a cache instance to use in the Stock class
cache = Cache()


def get_stock_data(symbol):
    # Check cache first
    cached_data = cache.load_data(f"{symbol}_data")
    if cached_data:
        return cached_data

    # API call
    api_data = fetch_data_from_api(symbol)  # Define this function as needed
    cache.save_data(f"{symbol}_data", api_data)
    return api_data


def fetch_data_from_api(symbol):
    api_key = API_KEY
    URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&interval=5min&apikey={api_key}"
    response = requests.get(URL)
    data = response.json()
    return data


def get_stock_overview(symbol):
    # Similar caching mechanism as get_stock_data
    cached_overview = cache.load_data(f"{symbol}_overview")
    if cached_overview:
        return cached_overview

    overview_data = fetch_overview_from_api(symbol)  # Define this function as needed
    cache.save_data(f"{symbol}_overview", overview_data)
    return overview_data


def fetch_overview_from_api(symbol):
    api_key = API_KEY
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if 'Information' in data or 'Note' in data or 'Error Message' in data:
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
        return plot_url

    def predict_prices_7days(self):
        key = 'Time Series (Daily)'
        prices = [float(self.data[key][date]['4. close']) for date in sorted(self.data[key].keys())[-30:]]
        # Assume model expects input shape [1, 30, 1] for one feature per day
        prices = np.array(prices).reshape(1, -1, 1)
        predictions = self.model.predict(prices)
        return predictions.flatten().tolist()

        # Test data for prediction
        # test_data = np.array([1, 2, 3, 4, 5, 6,
        #                       7, 8, 9, 10, 11, 12,
        #                       13, 14, 15, 16, 17, 18,
        #                       19, 20, 21, 22, 23, 24,
        #                       25, 26, 27, 28, 29, 30])
        # predictions = self.make_prediction(self.model, test_data)
        # return predictions

    def plot_predictions(self):
        key = 'Time Series (Daily)'
        # Ensure dates are sorted in ascending order
        dates = sorted(list(self.data[key].keys())[-30:])
        historical_data = [float(self.data[key][date]['4. close']) for date in dates]
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Convert dates to datetime objects

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

        plt.title("Stock Price Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.gcf().autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return plot_url
