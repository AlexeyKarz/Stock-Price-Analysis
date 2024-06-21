import base64
import io
import requests
from flask import jsonify
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
# from tensorflow.keras.metrics import MeanAbsoluteError
# from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.dates as mdates
from datetime import datetime
from pandas import date_range, to_datetime
import numpy as np



import matplotlib.dates as mdates
from datetime import datetime

API_KEY = 'NSQ25HG8ERO35TPU'


def get_stock_data(symbol):
    api_key = API_KEY
    URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&interval=5min&apikey={api_key}"
    response = requests.get(URL)
    data = response.json()
    return data


def get_stock_overview(symbol):
    api_key = API_KEY
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if 'Information' in data or 'Note' in data or 'Error Message' in data:
        return {'error': 'API request limit reached or invalid API call'}

    return data


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

    def make_prediction(self, model, last_30_days):
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

    def predict_prices_7days(self):
        # key = 'Time Series (Daily)'
        # prices = [float(self.data[key][date]['4. close']) for date in sorted(self.data[key].keys())[-30:]]
        # # Assume model expects input shape [1, 30, 1] for one feature per day
        # prices = np.array(prices).reshape(1, -1, 1)
        # predictions = self.model.predict(prices)
        # return predictions.flatten().tolist()

        # Test data for prediction
        test_data = np.array([1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12,
                              13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30])
        predictions = self.make_prediction(self.model, test_data)
        return predictions

    def plot_predictions_ipynb(self, show_full_history=True):
        """
        Plot historical data, the most recent 30 days, and the predicted prices with a connected view, using date formatting on the x-axis.

        Args:
        dates (list or pd.Series): Dates corresponding to the historical data.
        historical_data (np.array): Full array of historical prices.
        last_30_days (np.array): Array of the last 30 days of prices used for the prediction.
        predicted_prices (np.array): Predicted future prices from the model.
        show_full_history (bool): If True, show all historical data; if False, show only the last 30 days with every third day.
        """

        key = 'Time Series (Daily)'
        dates = list(self.data[key].keys())[-30:]  # last 30 days dates
        historical_data = [float(self.data[key][date]['4. close']) for date in dates]
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Convert dates to datetime objects

        predictions = self.predict_prices_7days()  # Get 7-day future predictions

        plt.figure(figsize=(12, 6))
        total_days = len(historical_data)
        future_days = len(predictions)

        # Create a date range for the predictions
        last_date = to_datetime(dates[-1])
        prediction_dates = date_range(last_date, periods=future_days + 1, freq='D')[
                           1:]  # start from the day after last_date

        # Combine historical dates and prediction dates
        full_dates = to_datetime(dates).append(prediction_dates)

        if show_full_history:
            # Plot all historical data
            plt.plot(full_dates[:total_days], historical_data, label="Historical Data", color='blue')
            # Highlight the last 30 days in a different color
            plt.plot(full_dates[total_days - 30:total_days], historical_data[-30:], label="Last 30 Days", color='orange')
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # locate months
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format dates as Year-Month
        else:
            # Start days at total_days - 30 to only show the last 30 days
            plt.plot(full_dates[total_days - 30:total_days], historical_data[-30:], label="Last 30 Days", color='orange')
            plt.plot(full_dates[total_days - 1:total_days + future_days],
                     np.concatenate(([historical_data[-1]], predictions)), label="Predictions", color='red')
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))  # locate every third day
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # format dates as Year-Month-Day

        plt.title("Stock Price Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.gcf().autofmt_xdate()  # auto-format x-axis dates to fit nicely
        # plt.show()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return plot_url


    def plot_predictions(self, show_full_history=False):
        key = 'Time Series (Daily)'
        dates = list(self.data[key].keys())[-30:]  # last 30 days dates
        historical_data = [float(self.data[key][date]['4. close']) for date in dates]
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Convert dates to datetime objects

        predictions = self.predict_prices_7days()  # Get 7-day future predictions

        # Create a date range for the predictions
        last_date = dates[-1]
        prediction_dates = date_range(last_date, periods=len(predictions) + 1, freq='D')[
                           1:]  # start from the day after last_date

        # Combine historical dates and prediction dates
        full_dates = dates + list(prediction_dates)

        plt.figure(figsize=(12, 6))
        if show_full_history:
            # Assume historical_data contains all historical data
            plt.plot(full_dates[:-len(predictions)], historical_data, label="Historical Data", color='blue')
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            plt.plot(full_dates[-30:], historical_data[-30:], label="Last 30 Days", color='orange')

        plt.plot(full_dates[-1:] + list(prediction_dates), [historical_data[-1]] + predictions, label="Predictions",
                 color='red')
        plt.title(f"Future Price Predictions for {self.symbol}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.gcf().autofmt_xdate()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return plot_url

