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
end_date = st.sidebar.date_input("End Date", datetime.now())

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

    df["Signal"]
