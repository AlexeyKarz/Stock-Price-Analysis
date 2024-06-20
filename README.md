# Stock Analysis Web Application

This is a web application that allows users to search for a stock symbol and view its analysis. The application is built using Python, JavaScript, and HTML/CSS.

## Features

- Search for a stock symbol by entering company name
- View analysis of the selected stock

## Technologies Used

- Python
- Flask
- TensorFlow
- JavaScript
- HTML/CSS

## Methodology

- The prediction is done using the basic model using 2 Dense layers. The model is trained on 3 years of close price data of 25 stocks from different sectors. The architecture and training methods can be accessed in the `stockpredictor.ipynb` file.
- The model could be improved by using more complex models like LSTM, GRU, etc. and by using more data for training.

## Setup and Installation

1. Clone the repository to your local machine.
2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

3. Install the required JavaScript packages using npm:

```bash
npm install
```

4. Run the Flask application:

```bash
python main.py
```

The application will be accessible at `http://localhost:3000`.

**Note 1:** You will need to obtain an API key from [Alpha Vantage](https://www.alphavantage.co/) to fetch stock data. 

**Note 2:** You can set up the application by building the Docker image and running the container

## Usage

1. Enter a stock name in the input field on the homepage.
2. Click the "Calculate" button to view the analysis of the stock.

## File Structure

- `main.py`: This is the main Python file that runs the Flask application.
- `stock.py`: This file contains the Stock class used for fetching and analyzing stock data.
- `static/script.js`: This file contains the JavaScript code for handling user interactions.
- `static/main.css`: This file contains the CSS styles for the application.
- `templates/index.html`: This is the HTML template for the homepage.
- `templates/analysis_page.html`: This is the HTML template for the analysis page.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

The project is available as open source under the terms of the [MIT License](https://choosealicense.com/licenses/mit/)