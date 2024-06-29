import base64
import io
import requests
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pandas import date_range, to_datetime
import numpy as np
import os
import pickle
import pandas as pd
from modules.logger import logger

API_KEY = 'NSQ25HG8ERO35TPU'


class Cache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        logger.debug("Cache directory created at: {}".format(cache_dir))

    def get_filename(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol}.pkl")

    def save_data(self, symbol, data):
        filename = self.get_filename(symbol)
        with open(filename, 'wb') as f:
            pickle.dump({
                'date': datetime.now(),
                'data': data
            }, f)
        logger.debug("Data cached for symbol: {}".format(symbol))

    def load_data(self, symbol):
        filename = self.get_filename(symbol)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                cached = pickle.load(f)
                # Check if the cache is still valid, let's say we refresh it every day
                if cached['date'].date() == datetime.now().date():
                    logger.info("Cached data found for symbol: {}".format(symbol))
                    return cached['data']
                else:
                    # Delete old data if it's older than one day
                    os.remove(filename)
        logger.info("No cached data found for symbol: {}".format(symbol))
        return None

    def clear_cache(self):
        """ Clears all cached files if they are older than one day. """
        logger.info("Starting clearing cache...")
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            with open(file_path, 'rb') as f:
                cached = pickle.load(f)
                if (datetime.now() - cached['date']) >= timedelta(days=1):
                    os.remove(file_path)
                    logger.info("Removed cached file: {}".format(filename))


# Create a cache instance to use in the Stock class
cache = Cache()


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


def get_labelled_windows(x, horizon=7):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size=30, horizon=7):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    # print(f"Window step:\n {window_step}")

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)),
                                                  axis=0).T  # create 2D array of windows of size window_size
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1 - test_split))  # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def process_stock_data_for_training(data, window_size=30, horizon=7, test_split=0.2):
    """
    Processes stock data into windows and labels for training a model.
    """
    # Make windows
    windows, labels = make_windows(data, window_size=window_size, horizon=horizon)

    # Make train and test splits
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows, labels,
                                                                                    test_split=test_split)
    logger.debug("Stock data processed successfully for training.")
    return train_windows, test_windows, train_labels, test_labels
