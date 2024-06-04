from flask import Flask, render_template, jsonify
from flask.views import MethodView
import requests

app = Flask(__name__)


def get_stock_data(symbol):
    API_KEY = 'NSQ25HG8ERO35TPU'
    URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}"
    response = requests.get(URL)
    data = response.json()
    return data



# @app.route('/')
class HomePage(MethodView):
    def get(self):
        # price = get_stock_data('AAPL')
        return render_template('index.html')


app.add_url_rule('/', view_func=HomePage.as_view('home_page'))

if __name__ == '__main__':
    app.run(debug=True)
