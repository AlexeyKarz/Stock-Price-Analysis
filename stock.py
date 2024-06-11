import base64
import io
import requests
from flask import jsonify
from matplotlib import pyplot as plt

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

