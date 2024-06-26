from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask.views import MethodView
import requests
import io
import base64
import stock

app = Flask(__name__)


def search_api_for_symbols(query):
    API_KEY = stock.API_KEY
    URL = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={API_KEY}"
    response = requests.get(URL)
    data = response.json()
    # Parse the response to match the expected format
    results = [{
        'name': item['2. name'],
        'symbol': item['1. symbol']
    } for item in data.get('bestMatches', [])]
    return results


@app.route('/search_stock_symbols')
def search_stock_symbols():
    query = request.args.get('query', '')
    return jsonify(search_api_for_symbols(query))


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
        # if stock_i.data["Information"] exists, then the API request limit is achieved

        overview_data = stock_i.overview

        if 'error' in overview_data or 'Information' in stock_i.data:
            return jsonify(overview_data), 429

        plot_url = stock_i.plot_stock()
        pred_plot_url = stock_i.plot_predictions(retrain=True)

        # Render both the plot and the overview data in the same template
        return render_template('analysis_page.html',
                               symbol=symbol,
                               plot_url=plot_url,
                               overview=overview_data,
                               pred_plot_url=pred_plot_url)


app.add_url_rule('/', view_func=HomePage.as_view('home_page'))
app.add_url_rule('/analysis/<symbol>', view_func=AnalysisPage.as_view('analysis_page'))  # Include <symbol> parameter

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
