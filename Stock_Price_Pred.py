# Stock Price Prediction with Sentiment Analysis using LSTM
# Created and customized by Tanuja Adlakha
# Original notebook adapted from open-source resourcespip install feedparser pandas_market_calendars

import pandas as pd
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas_market_calendars as mcal

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

# Function to generate date range for the past 2 years (excluding today's date)
def generate_date_range():
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=2*365)
    return pd.date_range(start=start_date, end=end_date)

# Function to fetch headlines for a given date from Yahoo Finance RSS feed for a specific stock ticker
def fetch_headlines(date, stock):
    url = f"https://finance.yahoo.com/rss/headline?s={stock}"
    feed = feedparser.parse(url)
    headlines = [
        entry.title for entry in feed.entries
        if datetime(*entry.published_parsed[:6]).date() == date.date()
    ]
    return headlines

# Function to fill missing sentiment scores with backward interpolation
def fill_missing_scores(df):
    df['Sentiment'] = df['Sentiment'].interpolate(method='linear', limit_direction='backward')
    return df

# Function to adjust historical prices for stock splits
def adjust_for_splits(hist):
    splits = hist['Stock Splits']
    for date, split in splits[splits != 0].items():
        hist.loc[:date, 'Close'] /= split
    return hist

def get_trading_days():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=(datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d'),
                              end_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))
    return schedule.index.to_pydatetime()


# Get stock ticker from user
stock = input("Enter the stock ticker: ")

# Fetch historical stock data for the past 2 years (excluding today's date)
stock_data = yf.Ticker(stock)
hist = stock_data.history(start=(datetime.now() - timedelta(days=2*365)), end=(datetime.now() - timedelta(days=1)))


# Adjust historical prices for stock splits
hist = adjust_for_splits(hist)

# Generate date range for the past 2 years (excluding today's date)
date_range = generate_date_range()

# Create a DataFrame with dates
df = pd.DataFrame(date_range, columns=['Date'])
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.set_index('Date', inplace=True)

# Generate Sentiment Scores
df['Headlines'] = df.index.to_series().apply(lambda date: fetch_headlines(datetime.strptime(date, '%Y-%m-%d'), stock))
df['Sentiment'] = df['Headlines'].apply(lambda headlines: get_sentiment_score(' '.join(headlines)) if headlines else None)
df = fill_missing_scores(df)

# Add closing prices to the DataFrame
# Convert the index to DatetimeIndex if it's not already
hist.index = pd.to_datetime(hist.index)
hist['Date'] = hist.index.strftime('%Y-%m-%d')
hist.set_index('Date', inplace=True)
df['Close'] = hist['Close']

# Interpolate missing closing prices
df['Close'] = df['Close'].interpolate(method='linear', limit_direction='backward').interpolate(method='linear', limit_direction='forward')
df = df[['Close', 'Sentiment']]

# Fill any remaining NaN values in closing prices with forward interpolation as a fallback
df['Close'] = df['Close'].interpolate(method='linear', limit_direction='forward')

# Remove the Headlines column and keep only Date, Closing Price, Sentiment Score
df = df[['Close', 'Sentiment']]

# Standardize the closing prices and sentiment scores
scaler = StandardScaler()
df[['Close', 'Sentiment']] = scaler.fit_transform(df[['Close', 'Sentiment']])

# Create Features and Labels
X = df[['Close', 'Sentiment']].values

train_size = int(len(X) * 0.8)
train, test = X[:train_size], X[train_size:]

train_X, train_y = train[:-1], train[1:, 0]
test_X, test_y = test[:-1], test[1:, 0]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# Get the number of trading days to predict from user
n_days = int(input("Enter the number of trading days to predict: "))

# Filter for trading days
trading_days = get_trading_days()
future_predictions_scaled = []
last_sequence_scaled = X[-1].reshape((1, 1, X.shape[1]))

predicted_dates = []
for current_date in trading_days:
    if len(future_predictions_scaled) >= n_days:
        break

    next_pred_scaled = model.predict(last_sequence_scaled)
    future_predictions_scaled.append(next_pred_scaled[0][0])
    predicted_dates.append(current_date.strftime('%Y-%m-%d'))

    last_sequence_scaled = np.concatenate([last_sequence_scaled[:, :, 1:], next_pred_scaled.reshape(1, 1, 1)], axis=2)


# Transform back to Original Scale
future_predictions_scaled_array = np.array(future_predictions_scaled).reshape(-1, 1)
future_predictions_original_array = scaler.inverse_transform(
    np.concatenate((future_predictions_scaled_array, np.zeros_like(future_predictions_scaled_array)), axis=1)
)[:, 0]


#Dispaly Predictions
print(f"Predicted closing prices for the next {n_days} trading days:")
for date, price in zip(predicted_dates, future_predictions_original_array):
    print(f"{date}: {price:.2f}")

# Get the number of trading days to predict from user
n_days = int(input("Enter the number of trading days to predict: "))


# Filter for trading days
trading_days = get_trading_days()
future_predictions_scaled = []
last_sequence_scaled = X[-1].reshape((1, 1, X.shape[1]))


predicted_dates = []
for current_date in trading_days:
    if len(future_predictions_scaled) >= n_days:
        break

    next_pred_scaled = model.predict(last_sequence_scaled)
    future_predictions_scaled.append(next_pred_scaled[0][0])
    predicted_dates.append(current_date.strftime('%Y-%m-%d'))

    last_sequence_scaled = np.concatenate([last_sequence_scaled[:, :, 1:], next_pred_scaled.reshape(1, 1, 1)], axis=2)


# Transform back to Original Scale
future_predictions_scaled_array = np.array(future_predictions_scaled).reshape(-1, 1)
future_predictions_original_array = scaler.inverse_transform(
    np.concatenate((future_predictions_scaled_array, np.zeros_like(future_predictions_scaled_array)), axis=1)
)[:, 0]


#Dispaly Predictions
print(f"Predicted closing prices for the next {n_days} trading days:")
for date, price in zip(predicted_dates, future_predictions_original_array):
    print(f"{date}: {price:.2f}")


