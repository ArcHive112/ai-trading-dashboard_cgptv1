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
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=90))
end_date = st.sidebar.date_input("End Date", datetime.now())

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("No data found. Please check ticker symbol or date range.")
    st.stop()

# ------------------------------
# Pivot Point Strategy
# ------------------------------
def calculate_pivots(df):
    df = df.copy()

    # Flatten MultiIndex columns if they exist (sometimes happens with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure required columns exist
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Pivot point formulas
    df["Pivot"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    df["R1"] = (2 * df["Pivot"]) - df["Low"].shift(1)
    df["S1"] = (2 * df["Pivot"]) - df["High"].shift(1)

    # Signals
    df["Signal"] = "HOLD"
    df.loc[df["Close"] > df["R1"], "Signal"] = "BUY"
    df.loc[df["Close"] < df["S1"], "Signal"] = "SELL"

    return df

data = calculate_pivots(data)

if data.empty:
    st.error("Could not calculate pivots — try a different ticker or range.")
    st.stop()

# ------------------------------
# Trade Journal (placeholder)
# ------------------------------
trade_log = []
for date, row in data.iterrows():
    if row["Signal"] in ["BUY", "SELL"]:
        trade_log.append({
            "Date": date,
            "Action": row["Signal"],
            "Price": row["Close"]
        })

trade_df = pd.DataFrame(trade_log)

# ------------------------------
# Plot Chart
# ------------------------------
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
))

# Buy markers
buy_signals = data[data["Signal"] == "BUY"]
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", color="green", size=12),
    name="BUY"
))

# Sell markers
sell_signals = data[data["Signal"] == "SELL"]
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", color="red", size=12),
    name="SELL"
))

fig.update_layout(title=f"{ticker} Price with Pivot Signals", xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Display Trade Journal
# ------------------------------
st.subheader("Trade Journal (Beta)")
if not trade_df.empty:
    st.dataframe(trade_df)
else:
    st.info("No trades generated in this date range.")
