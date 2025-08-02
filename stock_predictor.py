#!/usr/bin/env python3
"""
Stock Price Predictor
A simple machine learning model to predict next day's closing price using historical data.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta


def fetch_stock_data(ticker, period="60d"):
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Time period for data (default: 60 days)
    
    Returns:
        pd.DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        sys.exit(1)


def create_features(df):
    """
    Create features for the ML model.
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
    
    Returns:
        pd.DataFrame: DataFrame with features
    """
    # Create a copy to avoid modifying the original
    features_df = df[['Close']].copy()
    
    # Previous day's closing price
    features_df['Prev_Close'] = features_df['Close'].shift(1)
    
    # Rolling averages
    features_df['MA_5'] = features_df['Close'].rolling(window=5).mean()
    features_df['MA_10'] = features_df['Close'].rolling(window=10).mean()
    
    # Price change
    features_df['Price_Change'] = features_df['Close'].pct_change()
    
    # Volatility (rolling standard deviation)
    features_df['Volatility'] = features_df['Close'].rolling(window=5).std()
    
    # Drop NaN values created by rolling operations
    features_df.dropna(inplace=True)
    
    return features_df


def prepare_data(features_df):
    """
    Prepare data for training by creating X (features) and y (target).
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
    
    Returns:
        tuple: X (features), y (target), dates
    """
    # Features: everything except the current Close price
    X = features_df.drop(['Close'], axis=1)
    
    # Target: Next day's closing price (shift by -1)
    y = features_df['Close'].shift(-1)
    
    # Remove the last row (no next day price available)
    X = X[:-1]
    y = y[:-1]
    dates = features_df.index[:-1]
    
    return X, y, dates


def train_model(X, y):
    """
    Train a Linear Regression model.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target values
    
    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test
    """
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle for time series
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
    
    Returns:
        np.array: Predictions
    """
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print("\n📊 Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return predictions


def plot_results(y_test, predictions, ticker, test_dates):
    """
    Create a visualization of actual vs predicted prices.
    
    Args:
        y_test: Actual prices
        predictions: Predicted prices
        ticker: Stock ticker symbol
        test_dates: Dates for test data
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.plot(test_dates, y_test.values, label='Actual', color='blue', linewidth=2)
    plt.plot(test_dates, predictions, label='Predicted', color='red', linewidth=2, alpha=0.8)
    plt.title(f'{ticker} Stock Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot prediction error
    plt.subplot(1, 2, 2)
    error = y_test.values - predictions
    plt.bar(test_dates, error, color='green', alpha=0.6)
    plt.title('Prediction Error (Actual - Predicted)')
    plt.xlabel('Date')
    plt.ylabel('Error ($)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def predict_next_day(model, features_df, ticker):
    """
    Predict the next trading day's closing price.
    
    Args:
        model: Trained model
        features_df: DataFrame with features
        ticker: Stock ticker symbol
    
    Returns:
        float: Predicted price
    """
    # Get the latest features (last row)
    latest_features = features_df.drop(['Close'], axis=1).iloc[-1:].values
    
    # Make prediction
    next_day_price = model.predict(latest_features)[0]
    
    print(f"\n🔮 Prediction for {ticker}:")
    print(f"Current price: ${features_df['Close'].iloc[-1]:.2f}")
    print(f"Predicted next day price: ${next_day_price:.2f}")
    print(f"Expected change: ${next_day_price - features_df['Close'].iloc[-1]:.2f} "
          f"({((next_day_price / features_df['Close'].iloc[-1]) - 1) * 100:.2f}%)")
    
    return next_day_price


def main():
    """Main function to run the stock price predictor."""
    parser = argparse.ArgumentParser(description='Predict stock prices using machine learning')
    parser.add_argument('ticker', nargs='?', default=None, 
                       help='Stock ticker symbol (e.g., AAPL)')
    args = parser.parse_args()
    
    # Get ticker from command line or user input
    if args.ticker:
        ticker = args.ticker.upper()
    else:
        ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    
    if not ticker:
        print("Error: No ticker symbol provided.")
        sys.exit(1)
    
    print(f"\n📈 Stock Price Predictor for {ticker}")
    print("=" * 40)
    
    # Fetch data
    print("Fetching historical data...")
    stock_data = fetch_stock_data(ticker)
    
    # Create features
    print("Creating features...")
    features_df = create_features(stock_data)
    
    # Prepare data
    X, y, dates = prepare_data(features_df)
    
    # Train model
    print("Training Linear Regression model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Get test dates for plotting
    test_dates = dates[-len(X_test):]
    
    # Evaluate model
    predictions = evaluate_model(model, X_test, y_test)
    
    # Predict next day
    next_day_prediction = predict_next_day(model, features_df, ticker)
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(y_test, predictions, ticker, test_dates)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
