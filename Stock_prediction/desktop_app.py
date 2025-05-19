import os
import datetime
import warnings
import threading
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import fmpsdk
from transformers import pipeline
import optuna
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from newsapi import NewsApiClient
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf

# -----------------------------
# Suppress Warnings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Global settings for ticker requests
# -----------------------------
TICKER_INTERVAL = 300  # One new yfinance request every 300 seconds (5 minutes)
LAST_TICKER_REQUEST = None
CACHED_TICKER_PRICE = None

# -----------------------------
# 1. Environment & API Setup
# -----------------------------
load_dotenv()
apikey = os.getenv("apikey")  # Stock API key from .env file
newsapi = NewsApiClient(api_key="93bb9150efc64e83ac0facc882cbad1c")  # Hardcoded NewsAPI key

# Set up FinBERT for sentiment analysis
finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
SENTIMENT_CACHE_FILE = "sentiment_cache.csv"


def get_daily_sentiment_with_cache(symbol, date, cache_file=SENTIMENT_CACHE_FILE):
    """
    Returns the average sentiment for the given symbol on a specific date.
    Looks up the sentiment from a local CSV cache. For dates <=2025-04-14, returns 0.0.
    """
    date_str = date.strftime('%Y-%m-%d')
    allowed_date = datetime.datetime(2025, 4, 14)
    if date <= allowed_date:
        return 0.0
    if os.path.exists(cache_file):
        cache_df = pd.read_csv(cache_file)
    else:
        cache_df = pd.DataFrame(columns=["date", "sentiment"])
    row = cache_df[cache_df["date"] == date_str]
    if not row.empty:
        return row["sentiment"].values[0]
    try:
        articles = newsapi.get_everything(
            q=symbol, language="en",
            from_param=date_str, to=date_str,
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
        sentiment = 0.0
    new_row = {"date": date_str, "sentiment": sentiment}
    new_row_df = pd.DataFrame([new_row])
    cache_df = pd.concat([cache_df, new_row_df], ignore_index=True)
    cache_df.to_csv(cache_file, index=False)
    return sentiment


def get_current_price_yf(symbol):
    """
    Fetches the latest closing price using yfinance.
    Limits requests to once every TICKER_INTERVAL seconds.
    """
    global LAST_TICKER_REQUEST, CACHED_TICKER_PRICE
    now = datetime.datetime.now()
    if (LAST_TICKER_REQUEST is None) or ((now - LAST_TICKER_REQUEST).total_seconds() >= TICKER_INTERVAL):
        try:
            ticker = yf.Ticker(symbol)
            todays_data = ticker.history(period="1d", interval="1h")
            if not todays_data.empty:
                price = todays_data["Close"].iloc[-1]
            else:
                price = None
            LAST_TICKER_REQUEST = now
            CACHED_TICKER_PRICE = price
            return price
        except Exception as e:
            print(f"Error fetching yfinance price for {symbol}: {e}")
            LAST_TICKER_REQUEST = now
            return CACHED_TICKER_PRICE
    else:
        return CACHED_TICKER_PRICE


# -----------------------------
# 2. Load Local Historical Stock Data & API Data
# -----------------------------
df = pd.read_csv("sp500_processed_final.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
stock_symbol = "MSFT"
historical_stock_data = df[df["Symbol"] == stock_symbol]["Close"].dropna()
if not isinstance(historical_stock_data.index, pd.DatetimeIndex):
    historical_stock_data.index = pd.to_datetime(historical_stock_data.index, errors='coerce')
api_stock_data = fmpsdk.historical_price_full(apikey, stock_symbol)
if api_stock_data is None or not isinstance(api_stock_data, list):
    print("⚠️ API data could not be retrieved. Using only local historical data.")
    api_df = pd.DataFrame(columns=["Date", "Close"])
else:
    api_df = pd.DataFrame(api_stock_data)
    api_df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
    api_df["Date"] = pd.to_datetime(api_df["Date"])
    api_df.set_index("Date", inplace=True)
full_stock_data = pd.concat([historical_stock_data, api_df["Close"]], axis=0)
full_stock_data.index = pd.to_datetime(full_stock_data.index, errors='coerce')
full_stock_data = full_stock_data.drop_duplicates().sort_index()
full_stock_data = full_stock_data[full_stock_data.index.notna()]
scaler = StandardScaler()
historical_data = full_stock_data.values.reshape(-1, 1)
scaler.fit(historical_data)
stock_data_scaled = scaler.transform(historical_data).flatten()
print("✅ Loaded & Merged MSFT Stock Data:")
print(full_stock_data.tail())

# -----------------------------
# 3. Prepare Data for Training: Create Sequences & Compute Sentiment
# -----------------------------
backcast_length = 60
forecast_length = 1


def create_sequences(data, sentiment_series, backcast, forecast, include_sentiment=True):
    X, y = [], []
    for i in range(len(data) - backcast - forecast + 1):
        sequence = data[i: i + backcast]
        if include_sentiment:
            sentiment_window = sentiment_series[i: i + backcast]
            sequence = np.concatenate([sequence, sentiment_window * 2])
        X.append(sequence)
        y.append(data[i + backcast: i + backcast + forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


sentiment_list = []
for d in full_stock_data.index:
    sentiment_val = get_daily_sentiment_with_cache(stock_symbol, d)
    sentiment_list.append(sentiment_val)
dynamic_sentiment_series = np.array(sentiment_list)
min_sent = np.min(dynamic_sentiment_series)
max_sent = np.max(dynamic_sentiment_series)
if max_sent == min_sent:
    normalized_sentiment_series = dynamic_sentiment_series
else:
    normalized_sentiment_series = (dynamic_sentiment_series - min_sent) / (max_sent - min_sent)
print("📰 Daily Sentiment Scores (Normalized):")
print(normalized_sentiment_series)
sentiment_series = normalized_sentiment_series
X_seq_with_sentiment, y_seq = create_sequences(stock_data_scaled, sentiment_series, backcast_length, forecast_length,
                                               include_sentiment=True)
X_seq_without_sentiment, _ = create_sequences(stock_data_scaled, sentiment_series, backcast_length, forecast_length,
                                              include_sentiment=False)
dataset_with_sentiment = TensorDataset(torch.tensor(X_seq_with_sentiment), torch.tensor(y_seq))
dataset_without_sentiment = TensorDataset(torch.tensor(X_seq_without_sentiment), torch.tensor(y_seq))
loader_with_sentiment = DataLoader(dataset_with_sentiment, batch_size=128, shuffle=False)
loader_without_sentiment = DataLoader(dataset_without_sentiment, batch_size=128, shuffle=False)


# -----------------------------
# 4. Define N-BEATS Model, Hyperparameter Optimization, and Training/Forecasting Functions
# -----------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size, n_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()) for _ in range(n_layers - 1)]
        )
        self.theta = nn.Linear(hidden_size, forecast_size)

    def forward(self, x):
        out = self.net(x)
        forecast = self.theta(out)
        return forecast


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


# -----------------------------
# 5. Tkinter Desktop App with Controls, a Simple Line Chart, and an Exit Button
# -----------------------------
class StockPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MSFT Stock Prediction Demo")
        self.geometry("1200x900")
        self.configure(bg="white")
        self.latest_prediction = None
        self.create_widgets()
        self.initial_plot()
        self.update_ticker()
        self.update_history()

    def create_widgets(self):
        # Top frame: ticker and prediction controls
        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)
        self.ticker_label = ttk.Label(top_frame, text="Current MSFT Price: Updating...", font=("Arial", 12))
        self.ticker_label.pack(side="left", padx=5)
        self.run_button = ttk.Button(top_frame, text="Run Model Prediction", command=self.run_prediction)
        self.run_button.pack(side="left", padx=10)
        # Prediction label to display model prediction results
        self.prediction_label = ttk.Label(top_frame, text="Model predictions will appear here.", font=("Arial", 12))
        self.prediction_label.pack(side="left", padx=10)
        # Logging section: enter actual closing price and log it
        log_frame = ttk.Frame(top_frame)
        log_frame.pack(side="left", padx=10)
        ttk.Label(log_frame, text="Actual Price:").pack(side="left", padx=5)
        self.actual_entry = ttk.Entry(log_frame, width=15)
        self.actual_entry.pack(side="left", padx=5)
        self.log_button = ttk.Button(log_frame, text="Log Prediction", command=self.log_prediction)
        self.log_button.pack(side="left", padx=5)
        # NEW: Exit button to close the application
        exit_button = ttk.Button(top_frame, text="Exit", command=self.exit_app)
        exit_button.pack(side="right", padx=10)

        # Plot canvas
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Bottom frame: prediction history table
        history_frame = ttk.Frame(self)
        history_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        history_label = ttk.Label(history_frame, text="Past 5 Predictions", font=("Arial", 12, "bold"))
        history_label.pack(pady=5)
        self.history_tree = ttk.Treeview(history_frame, columns=("date", "predicted", "actual", "error"),
                                         show="headings", height=5)
        self.history_tree.heading("date", text="Date")
        self.history_tree.heading("predicted", text="Predicted")
        self.history_tree.heading("actual", text="Actual")
        self.history_tree.heading("error", text="Error")
        self.history_tree.column("date", anchor="center", width=100)
        self.history_tree.column("predicted", anchor="center", width=100)
        self.history_tree.column("actual", anchor="center", width=100)
        self.history_tree.column("error", anchor="center", width=100)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def exit_app(self):
        """Exit the application."""
        self.destroy()

    def initial_plot(self):
        """Plot a simple line chart of the actual closing prices."""
        self.ax.clear()
        self.ax.plot(full_stock_data.index, full_stock_data, label="Actual Closing Price", color="green")
        self.ax.set_title("Actual Closing Prices for MSFT")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price ($)")
        self.ax.legend()
        self.canvas.draw()

    def update_plot(self, pred_with_sent, pred_without_sent):
        """Update the line chart with the actual closing prices and horizontal prediction lines."""
        self.ax.clear()
        self.ax.plot(full_stock_data.index, full_stock_data, label="Actual Closing Price", color="green")
        self.ax.axhline(y=pred_with_sent, color="blue", linestyle="--", label="Predicted (With Sentiment)")
        self.ax.axhline(y=pred_without_sent, color="red", linestyle="--", label="Predicted (Without Sentiment)")
        self.ax.set_title("Actual vs. Predicted MSFT Closing Prices")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price ($)")
        self.ax.legend()
        self.canvas.draw()

    def update_ticker(self):
        """Update the ticker label every 10 seconds using yfinance data."""
        try:
            current_price = get_current_price_yf(stock_symbol)
            if current_price is not None:
                self.ticker_label.config(text=f"Current MSFT Price: ${current_price:.2f}")
            else:
                self.ticker_label.config(text="Current MSFT Price: N/A")
        except Exception as e:
            self.ticker_label.config(text="Current MSFT Price: Error")
            print(f"Error updating ticker: {e}")
        self.after(10000, self.update_ticker)

    def update_history(self):
        """
        Load the last 5 unique prediction days from prediction_log.csv and update the history table.
        The history is displayed in ascending chronological order (oldest at the top among these 5).
        """
        log_file = "prediction_log.csv"
        if os.path.exists(log_file):
            try:
                df_log = pd.read_csv(log_file, on_bad_lines='skip')
                df_log.columns = [col.strip().lower().replace(" ", "_") for col in df_log.columns]
                if len(df_log.columns) == 3:
                    df_log.rename(
                        columns={"date": "date", "actual_price": "actual_price", "predicted_price": "predicted_price"},
                        inplace=True)
                    df_log["error"] = df_log["predicted_price"] - df_log["actual_price"]
                # Convert date column
                df_log["date"] = pd.to_datetime(df_log["date"], format='%Y-%m-%d', errors="coerce")
                # Group by date (unique days) and take the last record for each day
                df_unique = df_log.sort_values("date").groupby("date").tail(1)
                # Then sort these unique days in ascending order and then take the last 5 unique days
                df_final = df_unique.sort_values("date", ascending=True).tail(5)
                for item in self.history_tree.get_children():
                    self.history_tree.delete(item)
                for _, row in df_final.iterrows():
                    self.history_tree.insert("", "end", values=(
                        row["date"].strftime('%Y-%m-%d'),
                        f"${row['predicted_price']:.2f}",
                        f"${row['actual_price']:.2f}",
                        f"${row['error']:.2f}"
                    ))
            except Exception as e:
                print(f"Error updating prediction history: {e}")

    def run_prediction(self):
        """Run model prediction in a separate thread; disable the prediction button during execution."""

        def task():
            self.run_button.config(state="disabled")
            try:
                best_params_sent = optimize_hyperparameters(loader_with_sentiment, backcast_length * 2)
                best_params_no_sent = optimize_hyperparameters(loader_without_sentiment, backcast_length)
                pred_with_sent = train_and_forecast(loader_with_sentiment, best_params_sent, backcast_length * 2,
                                                    use_sentiment=True)
                pred_without_sent = train_and_forecast(loader_without_sentiment, best_params_no_sent, backcast_length,
                                                       use_sentiment=False)
                prediction_text = (
                    f"MSFT Tomorrow's Prediction (With Sentiment): ${pred_with_sent:.2f}\n"
                    f"MSFT Tomorrow's Prediction (Without Sentiment): ${pred_without_sent:.2f}"
                )
                self.latest_prediction = pred_with_sent
                self.prediction_label.config(text=prediction_text)
                self.update_plot(pred_with_sent, pred_without_sent)
                self.update_history()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during prediction:\n{e}")
            finally:
                self.run_button.config(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def log_prediction(self):
        """Log the current prediction and the user-entered actual price to prediction_log.csv."""
        actual_price_str = self.actual_entry.get()
        try:
            actual_price = float(actual_price_str)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric closing price.")
            return
        if self.latest_prediction is None:
            messagebox.showwarning("No Prediction", "Please run the prediction first!")
            return
        today_str = datetime.datetime.today().strftime('%Y-%m-%d')
        log_entry = pd.DataFrame([{
            "date": today_str,
            "actual_price": actual_price,
            "predicted_price": self.latest_prediction,
            "error": self.latest_prediction - actual_price
        }])
        log_file = "prediction_log.csv"
        try:
            if os.path.exists(log_file):
                log_entry.to_csv(log_file, mode="a", header=False, index=False)
            else:
                log_entry.to_csv(log_file, index=False)
            messagebox.showinfo("Logged", "Prediction and actual price logged successfully.")
            self.update_history()
        except Exception as e:
            messagebox.showerror("File Error", f"Error logging data:\n{e}")


if __name__ == "__main__":
    app = StockPredictionApp()
    app.mainloop()
