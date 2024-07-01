from flask import render_template, request, redirect, url_for, jsonify
from flask.views import MethodView

import app.stock as stock
from app.modules.logger import logger


# def not_found_error(error):
#     return render_template('error_page.html', error_message="Page not found"), 404
#
#
# def internal_server_error(error):
#     return render_template('error_page.html', error_message="Internal server error"), 500
#
#
# def handle_exception(error):
#     return render_template('error_page.html', error_message=f"An error occurred: {str(error)}"), 500


# @app.route('/', methods=['GET', 'POST'])
class HomePage(MethodView):
    def get(self):
        return render_template('index.html')

    def post(self):
        stock_name = request.form['stock-name']
        retrain = 'retrain' in request.form and request.form['retrain'] == 'true'  # Capture the state of 'retrain'
        logger.debug("User searched for stock: {}, retrain: {}".format(stock_name, retrain))
        return redirect(
            url_for('analysis_page', symbol=stock_name, retrain=retrain))  # Redirect to the analysis page


# @app.route('/analysis/<symbol>')
class AnalysisPage(MethodView):
    def get(self, symbol):
        retrain = request.args.get('retrain', 'false').lower() == 'true'  # capture 'retrain' from URL query
        try:
            stock_i = stock.Stock(symbol)  # import the Stock class from the stock module
            if 'error' in stock_i.data:
                logger.error("Error in fetching data for stock: {}".format(symbol))
                raise ValueError("Stock symbol not found")
            overview_data = stock_i.overview  # Fetch the overview data for the stock

            # Check if the API returned an error or if the stock is not found
            if 'error' in overview_data or 'error' in stock_i.data:
                logger.error("Error in fetching data for stock: {}".format(symbol))
                return jsonify(overview_data['error']), 429

            # if the stock is not found in the API response

            plot_url = stock_i.plot_stock()  # Plot the stock data

            # Retrain the model if the user requested so
            pred_plot_url = stock_i.plot_predictions(retrain=retrain)

            logger.debug("Rendering analysis page for stock: {}".format(symbol))

            # Render both the plot and the overview data in the same template
            return render_template('analysis_page.html',
                                   symbol=symbol,
                                   plot_url=plot_url,
                                   overview=overview_data,
                                   pred_plot_url=pred_plot_url)
        except ValueError as e:
            return render_template('error_page.html', error_message=str(e)), 404


def init_app_routes(app):
    @app.route('/search_stock_symbols')
    def search_stock_symbols():
        query = request.args.get('query', '')
        return jsonify(stock.search_api_for_symbols(query))

    # Here, you add the url rules
    app.add_url_rule('/', view_func=HomePage.as_view('home_page'))
    app.add_url_rule('/analysis/<symbol>', view_func=AnalysisPage.as_view('analysis_page'))
