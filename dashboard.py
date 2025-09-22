import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.title("AI Trading Dashboard")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=180))

today = datetime.today()
end_date = st.sidebar.date_input("End Date", today)
if end_date > today:
    end_date = today

strategy = st.sidebar.selectbox(
    "Select Strategy",
    ["Pivot Points", "SMA Crossover", "RSI"]
)

initial_cash = st.sidebar.number_input("Starting Cash ($)", value=10000, step=1000)

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("No data found. Please check ticker or date range.")
    st.stop()

# ------------------------------
# Strategies
# ------------------------------
def pivot_strategy(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["Pivot"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    df["R1"] = (2 * df["Pivot"]) - df["Low"].shift(1)
    df["S1"] = (2 * df["Pivot"]) - df["High"].shift(1)

    df["Signal"] = "HOLD"
    df.loc[df["Close"] > df["R1"], "Signal"] = "BUY"
    df.loc[df["Close"] < df["S1"], "Signal"] = "SELL"
    return df

def sma_strategy(df, fast=10, slow=30):
    df = df.copy()
    df["SMA_Fast"] = df["Close"].rolling(fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(slow).mean()

    df["Signal"] = "HOLD"
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = "BUY"
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = "SELL"
    return df

def rsi_strategy(df, window=14):
    df = df.copy()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Signal"] = "HOLD"
    df.loc[df["RSI"] > 70, "Signal"] = "SELL"
    df.loc[df["RSI"] < 30, "Signal"] = "BUY"
    return df

# Apply selected strategy
if strategy == "Pivot Points":
    data = pivot_strategy(data)
elif strategy == "SMA Crossover":
    data = sma_strategy(data)
elif strategy == "RSI":
    data = rsi_strategy(data)

# ------------------------------
# Trade Journal + Portfolio
# ------------------------------
cash = initial_cash
position = 0
trade_log = []

for date, row in data.iterrows():
    action = row["Signal"]
    price = row["Close"]

    if action == "BUY" and cash > 0:
        qty = cash // price
        if qty > 0:
            cash -= qty * price
            position += qty
            trade_log.append({"Date": date, "Action": "BUY", "Price": price, "Shares": qty, "Cash": cash, "Position": position})

    elif action == "SELL" and position > 0:
        cash += position * price
        trade_log.append({"Date": date, "Action": "SELL", "Price": price, "Shares": position, "Cash": cash, "Position": 0})
        position = 0

# Final portfolio value
portfolio_value = cash + position * data["Close"].iloc[-1]
trade_df = pd.DataFrame(trade_log)

# ------------------------------
# Performance Summary
# ------------------------------
st.subheader("Performance Summary")
st.metric("Final Portfolio Value", f"${portfolio_value:,.2f}")
st.metric("Cash Remaining", f"${cash:,.2f}")
st.metric("Shares Held", f"{position}")

if not trade_df.empty:
    sells = trade_df[trade_df["Action"] == "SELL"]
    wins = (sells["Price"].diff() > 0).sum()
    total_sells = len(sells)
    win_rate = wins / total_sells if total_sells > 0 else 0
    st.metric("Win Rate", f"{win_rate:.2%}")

# ------------------------------
# Plot Chart
# ------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
))

buy_signals = data[data["Signal"] == "BUY"]
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", color="green", size=12),
    name="BUY"
))

sell_signals = data[data["Signal"] == "SELL"]
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", color="red", size=12),
    name="SELL"
))

fig.update_layout(title=f"{ticker} - {strategy} Strategy", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Trade Journal
# ------------------------------
st.subheader("Trade Journal")
if not trade_df.empty:
    st.dataframe(trade_df)
    csv = trade_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Trade Log", csv, "trade_log.csv", "text/csv")
else:
    st.info("No trades executed.")
