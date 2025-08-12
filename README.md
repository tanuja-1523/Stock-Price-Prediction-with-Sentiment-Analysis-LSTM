# Stock Price Prediction with Sentiment Analysis & LSTM
Author-Terni Tanuja Adlakha

This project predicts future stock prices by combining historical stock data with sentiment analysis of news headlines. The prediction model is based on a Long Short-Term Memory (LSTM) neural network that processes time-series data and sentiment scores to forecast stock prices.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Code Explanation](#code-explanation)
4. [Model Evaluation](#model-evaluation)
5. [License](#license)

## Overview
This project leverages two key data sources to predict stock prices:
1. **Historical Stock Data**: Using Yahoo Finance, historical closing prices of stocks are obtained.
2. **Sentiment Analysis of News Headlines**: News headlines related to the stock are parsed via Yahoo Finance's RSS feed and processed using VADER sentiment analysis to quantify market sentiment.

By combining these two data sources and processing them through an LSTM model, the goal is to predict future stock prices based on both historical trends and public sentiment around the stock.

## Features
- **Sentiment Analysis**: Collects stock-related news headlines and calculates sentiment scores using the VADER SentimentIntensityAnalyzer.
- **Stock Price Data**: Retrieves historical stock prices and adjusts them for stock splits.
- **Data Preprocessing**: Handles missing values by interpolating missing sentiment and price data.
- **LSTM Model**: Uses LSTM (Long Short-Term Memory) for time-series prediction based on closing prices and sentiment scores.
- **Prediction for Future Days**: Predicts future stock prices for a specified number of trading days, excluding weekends.
- **Evaluation**: Measures the performance of the model using Mean Squared Error (MSE) and R-squared (R2).

## Code Explanation

### 1. **Sentiment Analysis**:
   - News headlines related to a given stock ticker are fetched via Yahoo Finance RSS feeds.
   - Sentiment scores for these headlines are computed using the **VADER SentimentIntensityAnalyzer** from `nltk`.

### 2. **Historical Stock Data**:
   - Stock prices for the last two years (excluding the current day) are fetched using the **yfinance** library.
   - The data is adjusted for stock splits to ensure accurate price calculations.

### 3. **Data Preprocessing**:
   - Missing sentiment scores are interpolated using linear interpolation.
   - Closing prices are also interpolated where necessary.

### 4. **LSTM Model**:
   - A Sequential model is built using **Keras**.
   - The model consists of an LSTM layer followed by a Dense layer, and is compiled using the Adam optimizer and mean squared error loss function.
   - The model is trained using the training data and validated on a test set.

### 5. **Prediction**:
   - The model predicts stock prices for a given number of future trading days.
   - Predictions are made while skipping weekends, and the results are transformed back to the original price scale.

## Model Evaluation

The model’s performance is evaluated using two common metrics:
1. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
2. **R-squared (R2)**: Indicates how well the model explains the variance in the target variable.

### Example Output:
```
Mean Squared Error (MSE): 0.0054
R-squared (R2): 0.85
Predicted closing prices for the next 5 trading days:
2024-11-21: 145.23
2024-11-22: 146.50
2024-11-25: 147.80
2024-11-26: 149.10
2024-11-27: 150.60

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **VADER Sentiment Analysis**: The sentiment analysis in this project uses the VADER lexicon from `nltk`.
- **Yahoo Finance**: Historical stock data is sourced from Yahoo Finance via the `yfinance` library.
- **Keras and TensorFlow**: Used for building and training the LSTM model.

Feel free to open an issue if you encounter any bugs or have suggestions for improvements. Happy coding!




