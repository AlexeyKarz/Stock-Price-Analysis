from flask import Flask, render_template, request, redirect, url_for
from flask.views import MethodView
import requests
import matplotlib.pyplot as plt
import io
import base64
import stock

app = Flask(__name__)


# def get_stock_data(symbol):
#     API_KEY = 'NSQ25HG8ERO35TPU'
#     URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}"
#     response = requests.get(URL)
#     data = response.json()
#     return data


# @app.route('/', methods=['GET', 'POST'])
class HomePage(MethodView):
    def get(self):
        return render_template('index.html')

    def post(self):
        stock_name = request.form['stock-name']
        return redirect(url_for('analysis_page', symbol=stock_name))


# @app.route('/analysis/<symbol>')
class AnalysisPage(MethodView):
    def get(self, symbol):
        stock_i = stock.Stock(symbol)  # Ensure this Stock class is correctly imported and used
        plot_url = stock_i.plot_stock()
        return render_template('analysis_page.html', symbol=symbol, plot_url=plot_url)


app.add_url_rule('/', view_func=HomePage.as_view('home_page'))
app.add_url_rule('/analysis/<symbol>', view_func=AnalysisPage.as_view('analysis_page'))  # Include <symbol> parameter

if __name__ == '__main__':
    app.run(debug=True)
