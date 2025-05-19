import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import fmpsdk
from transformers import pipeline
import optuna
import warnings
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# 1. Environment & API Setup
# ===============================
load_dotenv()
apikey = os.getenv("apikey")  # Stock API key from .env file
newsapi = NewsApiClient(api_key="93bb9150efc64e83ac0facc882cbad1c")  # Hardcoded NewsAPI key

# Set up FinBERT for sentiment analysis
finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Define cache file for sentiment results
SENTIMENT_CACHE_FILE = "../sentiment_cache.csv"


def get_daily_sentiment_with_cache(symbol, date, cache_file=SENTIMENT_CACHE_FILE):
    """
    Returns the average sentiment for a given symbol on a specific date.
    Checks a local CSV cache first to avoid hitting API rate limits.
    If the date is before the allowed free plan start (2025-04-12), returns np.nan.
    """
    date_str = date.strftime('%Y-%m-%d')

    # Load cache if it exists; otherwise create an empty DataFrame.
    if os.path.exists(cache_file):
        cache_df = pd.read_csv(cache_file)
    else:
        cache_df = pd.DataFrame(columns=["date", "sentiment"])

    # Check if the sentiment for this date is already cached.
    row = cache_df[cache_df["date"] == date_str]
    if not row.empty:
        return row["sentiment"].values[0]

    # Free plan permits articles as far back as 2025-04-12.
    allowed_date = datetime.datetime(2025, 4, 12)
    if date < allowed_date:
        sentiment = np.nan  # or return 0.0 as default
    else:
        try:
            articles = newsapi.get_everything(
                q=symbol,
                language="en",
                from_param=date_str,  # Use 'from_param' because 'from' is reserved.
                to=date_str,
                sort_by="relevancy"
            )
            headlines = [article["title"] for article in articles.get("articles", [])]
            if not headlines:
                print(f"⚠️ No headlines for {symbol} on {date_str}.")
                sentiment = np.nan
            else:
                sentiments = [finbert_pipeline(headline)[0]["score"] for headline in headlines]
                sentiment = np.mean(sentiments) if sentiments else np.nan
        except Exception as e:
            print(f"⚠️ NewsAPI error for {symbol} on {date_str}: {e}")
            sentiment = np.nan

    if np.isnan(sentiment):
        sentiment = 0.0  # Impute missing sentiment with 0.0

    # Save the new row to the cache using pd.concat (since .append is deprecated)
    new_row = {"date": date_str, "sentiment": sentiment}
    new_row_df = pd.DataFrame([new_row])
    cache_df = pd.concat([cache_df, new_row_df], ignore_index=True)
    cache_df.to_csv(cache_file, index=False)

    return sentiment


# ===============================
# 2. Load Local Historical Stock Data & Current API Data
# ===============================
df = pd.read_csv("sp500_processed_final.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
stock_symbol = "MSFT"

# Extract historical MSFT closing prices from local dataset
historical_stock_data = df[df["Symbol"] == stock_symbol]["Close"].dropna()
# Ensure the index is datetime (if not, convert it)
if not isinstance(historical_stock_data.index, pd.DatetimeIndex):
    historical_stock_data.index = pd.to_datetime(historical_stock_data.index, errors='coerce')

# Fetch latest MSFT prices via API
api_stock_data = fmpsdk.historical_price_full(apikey, stock_symbol)
if isinstance(api_stock_data, list):
    api_df = pd.DataFrame(api_stock_data)
    api_df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
    api_df["Date"] = pd.to_datetime(api_df["Date"])
    api_df.set_index("Date", inplace=True)
else:
    raise ValueError(f"⚠️ Unexpected API response format! Full response: {api_stock_data}")

# Merge local historical data and real-time API data; remove duplicates and sort by index.
full_stock_data = pd.concat([historical_stock_data, api_df["Close"]], axis=0)
full_stock_data.index = pd.to_datetime(full_stock_data.index, errors='coerce')
full_stock_data = full_stock_data.drop_duplicates().sort_index()

# **Fix:** Remove rows with NaT in the index.
full_stock_data = full_stock_data[full_stock_data.index.notna()]

# Scale merged stock data
scaler = StandardScaler()
historical_data = full_stock_data.values.reshape(-1, 1)
scaler.fit(historical_data)
stock_data_scaled = scaler.transform(historical_data).flatten()

print(f"✅ Loaded & Merged MSFT Stock Data:\n{full_stock_data.tail()}")

# ===============================
# 3. Dynamic Sentiment Analysis (with Caching)
# ===============================
sentiment_list = []
for d in full_stock_data.index:
    sentiment = get_daily_sentiment_with_cache(stock_symbol, d)
    sentiment_list.append(sentiment)
dynamic_sentiment_series = np.array(sentiment_list)

# Normalize the daily sentiment values with min-max normalization.
min_sent = np.min(dynamic_sentiment_series)
max_sent = np.max(dynamic_sentiment_series)
if max_sent == min_sent:
    normalized_sentiment_series = dynamic_sentiment_series
else:
    normalized_sentiment_series = (dynamic_sentiment_series - min_sent) / (max_sent - min_sent)

print("📰 Daily Sentiment Scores (Normalized):")
print(normalized_sentiment_series)

# ===============================
# 4. Data Preparation for Training
# ===============================
backcast_length = 60
forecast_length = 1


def create_sequences(data, sentiment_series, backcast, forecast, include_sentiment=True):
    X, y = [], []
    for i in range(len(data) - backcast - forecast + 1):
        sequence = data[i: i + backcast]
        if include_sentiment:
            sentiment_window = sentiment_series[i: i + backcast]
            # Concatenate prices and sentiment (multiplied by 2)
            sequence = np.concatenate([sequence, sentiment_window * 2])
        X.append(sequence)
        y.append(data[i + backcast: i + backcast + forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if len(normalized_sentiment_series) != len(stock_data_scaled):
    raise ValueError("Mismatch between sentiment series and stock data lengths!")

sentiment_series = normalized_sentiment_series
X_seq_with_sentiment, y_seq = create_sequences(stock_data_scaled, sentiment_series, backcast_length, forecast_length,
                                               include_sentiment=True)
X_seq_without_sentiment, _ = create_sequences(stock_data_scaled, sentiment_series, backcast_length, forecast_length,
                                              include_sentiment=False)

dataset_with_sentiment = TensorDataset(torch.tensor(X_seq_with_sentiment), torch.tensor(y_seq))
dataset_without_sentiment = TensorDataset(torch.tensor(X_seq_without_sentiment), torch.tensor(y_seq))
loader_with_sentiment = DataLoader(dataset_with_sentiment, batch_size=128, shuffle=False)
loader_without_sentiment = DataLoader(dataset_without_sentiment, batch_size=128, shuffle=False)

# ===============================
# 5. Define N-BEATS Model
# ===============================
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size, n_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())
              for _ in range(n_layers - 1)]
        )
        self.theta = nn.Linear(hidden_size, forecast_size)

    def forward(self, x):
        out = self.net(x)
        forecast = self.theta(out)
        return forecast

# ===============================
# 6. Hyperparameter Optimization
# ===============================
def optimize_hyperparameters(loader, input_size):
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", input_size, 128)
        n_layers = trial.suggest_int("n_layers", 2, 5)
        learning_rate = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        model = NBeatsBlock(input_size, forecast_length, hidden_size, n_layers)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(10):
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                forecast = model(X_batch)
                loss = criterion(forecast, y_batch)
                loss.backward()
                optimizer.step()
        total_loss = sum(criterion(model(X_batch), y_batch).item() * len(X_batch)
                         for X_batch, y_batch in loader) / len(loader.dataset)
        return total_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

# ===============================
# 7. Model Training & Forecasting
# ===============================
def train_and_forecast(loader, best_params, input_size, use_sentiment=False):
    model = NBeatsBlock(input_size, forecast_length, best_params["hidden_size"], best_params["n_layers"])
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    for epoch in range(500):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            forecast = model(X_batch).squeeze(-1)
            loss = criterion(forecast, y_batch.view(-1))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        if use_sentiment:
            # Rebuild input: concatenation of last backcast_length prices and sentiment (multiplied by 2)
            last_prices = stock_data_scaled[-backcast_length:]
            last_sent = normalized_sentiment_series[-backcast_length:]
            last_window = np.concatenate([last_prices, last_sent * 2])
        else:
            last_window = stock_data_scaled[-backcast_length:]
        last_window_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
        forecast_tensor = model(last_window_tensor).squeeze(-1)
    predicted_scaled = forecast_tensor.numpy().reshape(-1, 1)
    predicted_real = scaler.inverse_transform(predicted_scaled)[:, 0]
    return predicted_real[0]

print("⚡ Running model training and prediction...")

prediction_with_sentiment = train_and_forecast(
    loader_with_sentiment,
    optimize_hyperparameters(loader_with_sentiment, backcast_length * 2),
    backcast_length * 2,
    use_sentiment=True
)

prediction_without_sentiment = train_and_forecast(
    loader_without_sentiment,
    optimize_hyperparameters(loader_without_sentiment, backcast_length),
    backcast_length,
    use_sentiment=False
)

print(f"🔮 MSFT Tomorrow’s Prediction (With Sentiment): ${prediction_with_sentiment:.2f}")
print(f"🔮 MSFT Tomorrow’s Prediction (Without Sentiment): ${prediction_without_sentiment:.2f}")

# ===============================
# 8. Visualization + Error Tracking
# ===============================
print("📊 Running visualization...")

plt.figure(figsize=(14, 6))
plt.plot(full_stock_data.index, full_stock_data, label="Actual Closing Price", color="green")
plt.axhline(y=prediction_with_sentiment, color="blue", linestyle="--", label="Predicted (With Sentiment)")
plt.axhline(y=prediction_without_sentiment, color="red", linestyle="--", label="Predicted (Without Sentiment)")
plt.title("Actual vs. Predicted MSFT Closing Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.grid(True)
plt.show()

print("✅ Visualization completed!")

# ===============================
# 9. Feedback Loop: Logging & Updating Historical Data
# ===============================
actual_price = float(input("Enter today's actual closing price: "))

# Log the prediction and actual outcome.
log_entry = pd.DataFrame([{
    "date": datetime.datetime.today().strftime('%Y-%m-%d'),
    "predicted_price": prediction_with_sentiment,  # Change if you want to log the other prediction
    "actual_price": actual_price,
    "error": prediction_with_sentiment - actual_price
}])
log_file = "prediction_log.csv"
if os.path.exists(log_file):
    log_entry.to_csv(log_file, mode="a", header=False, index=False)
else:
    log_entry.to_csv(log_file, index=False)
print("Prediction and actual price logged.")

# Update the historical dataset to include the new day's actual data.
new_row = pd.DataFrame({
    "Date": [datetime.datetime.today().strftime('%Y-%m-%d')],
    "Symbol": [stock_symbol],
    "Close": [actual_price]
})
historical_csv_file = "sp500_processed_final.csv"
historical_df = pd.read_csv(historical_csv_file, parse_dates=["Date"])
historical_df = pd.concat([historical_df, new_row], ignore_index=True)
historical_df.to_csv(historical_csv_file, index=False)
print(f"Historical dataset updated with today's data ({datetime.datetime.today().strftime('%Y-%m-%d')}).")
