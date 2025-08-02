# 📈 Stock Price Predictor

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557c.svg?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/ShamsRupak/Stock-Price-Predictor?style=for-the-badge)

A simple yet powerful machine learning application that predicts the next day's closing price of any stock using historical data and Linear Regression. Built with Python, this project fetches real-time data, engineers meaningful features, and provides visual insights into prediction accuracy.

## 🚀 Features

- **Real-time Data Fetching**: Automatically downloads the latest 60 days of stock data using Yahoo Finance
- **Smart Feature Engineering**: Creates rolling averages, volatility measures, and price change indicators
- **Visual Analytics**: Generates clear charts showing actual vs predicted prices and prediction errors
- **Command-line Flexibility**: Run with any stock ticker symbol as an argument or interactive input
- **Performance Metrics**: Displays MAE, MSE, RMSE, and R² scores for model evaluation
- **Error Handling**: Gracefully handles invalid ticker inputs and data issues

## 🛠️ Tech Stack

- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning model (Linear Regression)
- **yfinance** - Yahoo Finance API for stock data
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **numpy** - Numerical computations

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/ShamsRupak/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 How to Run

### Option 1: Command-line argument
```bash
python stock_predictor.py AAPL
```

### Option 2: Interactive mode
```bash
python stock_predictor.py
# Then enter ticker when prompted
```

### Examples:
```bash
# Predict Apple stock
python stock_predictor.py AAPL

# Predict Tesla stock
python stock_predictor.py TSLA

# Predict Microsoft stock
python stock_predictor.py MSFT
```

## 📊 Sample Output

### Console Output:
```
📈 Stock Price Predictor for AAPL
========================================
Fetching historical data...
Creating features...
Training Linear Regression model...

📊 Model Performance Metrics:
Mean Absolute Error (MAE): $2.34
Mean Squared Error (MSE): $8.67
Root Mean Squared Error (RMSE): $2.94
R² Score: 0.9234

🔮 Prediction for AAPL:
Current price: $150.25
Predicted next day price: $151.87
Expected change: $1.62 (1.08%)

Generating visualization...
✅ Analysis complete!
```

### Generated Visualization:
*The script automatically generates a PNG file (`[TICKER]_prediction_results.png`) showing:*
- Left panel: Actual vs Predicted prices over the test period
- Right panel: Prediction errors for each day

## 🔍 How It Works

<details>
<summary><b>Behind the Scenes (Click to expand)</b></summary>

### Data Collection
The predictor fetches 60 days of historical stock data from Yahoo Finance, including:
- Open, High, Low, Close prices
- Trading volume
- Adjusted close prices

### Feature Engineering
The model creates several technical indicators:
1. **Previous Close**: Yesterday's closing price
2. **Moving Averages**: 5-day and 10-day simple moving averages
3. **Price Change**: Percentage change from previous day
4. **Volatility**: 5-day rolling standard deviation

### Model Training
- Uses **Linear Regression** from scikit-learn
- Splits data: 80% training, 20% testing
- Preserves temporal order (no shuffling for time series data)

### Prediction Process
1. Train the model on historical features
2. Evaluate performance on test set
3. Use the latest available features to predict tomorrow's price
4. Generate comprehensive visualizations

### Why Linear Regression?
- Simple and interpretable
- Fast training and prediction
- Works well for short-term price trends
- Great baseline model for stock prediction
- Easy for beginners to understand and modify

</details>

## 📁 Project Structure
```
Stock-Price-Predictor/
│
├── stock_predictor.py      # Main prediction script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore            # Git ignore file
└── [ticker]_prediction_results.png  # Generated after running
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new features (more ML models, technical indicators)
- Improve error handling
- Enhance visualizations
- Add more documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always do your own research and consult with financial advisors.

## 🏷️ Tags

`finance` `machine-learning` `stock-prediction` `python` `linear-regression` `data-science` `yahoo-finance` `scikit-learn` `time-series`

---

Made with ❤️ by [Shams Rupak](https://github.com/ShamsRupak)
