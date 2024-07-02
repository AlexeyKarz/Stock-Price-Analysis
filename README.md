# Stock Market Price Predictor


### <i>The project is currently under development. The deployment will be done later. If you want to try out the application, you can clone this repository and launch it on your local machine.</i>


## Project Overview
This Flask application provides a platform for users to query stock symbols, view historical price data, and receive future price predictions using a time-series machine learning model. The application fetches data through the Alpha Vantage API and uses a deep learning model to predict future stock prices.

## Features
- **Symbol Search**: Users can search for stock symbols using company names.
- **Historical Data Visualization**: Displays historical price data for the chosen stock symbol.
- **Price Prediction**: Offers short-term price predictions based on historical data.
- **Dynamic Retraining**: Users have the option to retrain the predictive model with recent data.
- **Responsive UI**: The application includes a user-friendly interface that adapts to various devices.
- **Logging**: Systematic logging of events for troubleshooting and monitoring.
- **Caching**: Temporary storage of API data to minimize repeat calls and enhance performance.

## Directory Structure

```
/Stock-Price-Analysis
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── stock.py                 # Core functionalities for stock data processing
│   ├── templates/               # HTML templates for the web interface
│   │   ├── index.html
│   │   ├── analysis_page.html
│   │   └── error_page.html
│   ├── static/                  # CSS, JS, and image files
│   │   ├── style.css
│   │   ├── script.js
│   │   └── images/
│   ├── modules/                 # Additional modules like caching and logging
│   │   ├── cache.py
│   │   └── logger.py
│   │   └── dataprocess.py
│   └── models/                  # Machine learning models for price prediction
│       └── model_25s_7d.h5      # Pre-trained deep learning model
│
├── tests/                       # Unit tests for the application
│   └── test_app.py
│
├── cache/                       # Temporary storage for API data
├── logs/                        # Log files for the application
├── notebooks/                   # Jupyter notebooks for model training and analysis
│   ├── stockpredictor.ipynb     # Model training and evaluation
│   └── experiments.ipynb        # Experiments for model retraining 
│
├── config.py                    # Configuration settings for the application
├── venv/                        # Virtual environment for the project
├── README.md                    # This file
├── requirements.txt             # Python dependencies for pip
└── run.py                       # Entry point for the Flask application
```

## Setup

### Requirements
- Python 3.8+
- Pip
- Virtualenv (optional, recommended)

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://Stock-Price-Analysis.git
   cd Stock-Price-Analysis
   ```

2. **Create and Activate Virtual Environment** (optional)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   Replace `your_api_key` with your actual Alpha Vantage API key.
   - <i>Right now there is my API key, but it has limit of 25 requests per day, so you can create your own API key on Alpha Vantage website for experimenting with the application.</i>
   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   export API_KEY='your_api_key'
   ```

5. **Run the Application**
   ```bash
   flask run
   ```

   Or directly using Python:
   ```bash
   python run.py
   ```

## Usage
Navigate to `http://localhost:5000` in your web browser to access the application. Use the search functionality to explore stock symbols and view their analysis.

## Contributing
Feel free to fork the repository and submit pull requests. You can also open issues for bugs you've found or features you think would be beneficial.

## Contact
If you want to contact me you can reach me at: al.karzanov@gmail.com or [LinkedIn](https://www.linkedin.com/in/aleksei-karzanov/)

## License
This project is licensed under the MIT License.

## Future Enhancements
- Implement more robust error handling and user feedback mechanisms.
- Extend predictive analytics features, possibly integrating more complex machine learning models.
- Optimize caching mechanism to auto-refresh based on data age or specific intervals.
- Improve the UI/UX for better user engagement.
