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
    if not
