from unittest.mock import patch
import unittest
from main import app


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        """Set up a test client before each test."""
        # Creates a test client for the app.
        app.config['TESTING'] = True  # Enable testing mode
        self.client = app.test_client()

    def test_home_page(self):
        """Test the home page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Welcome to the Stock Market Price Predictor', response.data.decode())

    def test_search_stock_symbols(self):
        """Test the stock symbol search endpoint."""
        response = self.client.get('/search_stock_symbols?query=AAPL')
        self.assertEqual(response.status_code, 200)
        self.assertIn('AAPL', response.data.decode())

    def test_error_page_not_found(self):
        """Test accessing an undefined route."""
        response = self.client.get('/nothing')
        self.assertEqual(response.status_code, 404)

    @patch('main.requests.get')
    def test_search_api(self, mock_get):
        """Test the API call with mocked response."""
        # Configure the mock to return a response with an OK status code.
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'bestMatches': [{'1. symbol': 'AAPL', '2. name': 'Apple Inc.'}]
        }

        response = self.client.get('/search_stock_symbols?query=apple')
        self.assertIn('Apple Inc.', response.data.decode())
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
