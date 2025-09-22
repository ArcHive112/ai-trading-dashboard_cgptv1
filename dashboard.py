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

mode = st.sidebar.radio("Mode", ["Multi-Ticker", "Compare Strategies"])

if mode == "Multi-Ticker":
    tickers_input = st.sidebar.text_input("Ticker Symbols (comma-separated)", value="AAPL,MSFT,TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    tickers = [ticker]

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
today = datetime.today().date()
end_date = st.sidebar.date_input("End Date", today)
if end_date > today:
    end_date = today

if mode == "Multi-Ticker":
    strategy = st.sidebar.selectbox("Select Strategy", ["Pivot Points", "SMA Crossover", "RSI"])
else:
    strategy = None  # will run all in Compare mode

initial_cash = st.sidebar.number_input("Starting Cash ($)", value=10000, step=1000)

# Strategy Parameters
fast = st.sidebar.slider("Fast SMA Window", 5, 50, 10)
slow = st.sidebar.slider("Slow SMA Window", 20, 200, 30)
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
rsi_sell = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

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

def sma_strategy(df, fast, slow):
    df = df.copy()
    df["SMA_Fast"] = df["Close"].rolling(fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(slow).mean()
    df["Signal"] = "HOLD"
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = "BUY"
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = "SELL"
    return df

def rsi_strategy(df, window, buy_th, sell_th):
    df = df.copy()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Signal"] = "HOLD"
    df.loc[df["RSI"] > sell_th, "Signal"] = "SELL"
    df.loc[df["RSI"] < buy_th, "Signal"] = "BUY"
    return df

# ------------------------------
# Backtesting Function
# ------------------------------
def backtest(df, cash):
    cash_balance, position = cash, 0
    history = []
    for date, row in df.iterrows():
        action, price = row["Signal"], row["Close"]
        if action == "BUY" and cash_balance > 0:
            qty = cash_balance // price
            if qty > 0:
                cash_balance -= qty * price
                position += qty
                history.append({"Date": date, "Action": "BUY", "Price": price,
                                "Shares": qty, "Cash": cash_balance,
                                "Position": position,
                                "Portfolio": cash_balance + position * price})
        elif action == "SELL" and position > 0:
            cash_balance += position * price
            history.append({"Date": date, "Action": "SELL", "Price": price,
                            "Shares": position, "Cash": cash_balance,
                            "Position": 0, "Portfolio": cash_balance})
            position = 0
    final_value = cash_balance + position * df["Close"].iloc[-1]
    return pd.DataFrame(history), final_value

# ------------------------------
# Multi-Ticker Mode
# ------------------------------
if mode == "Multi-Ticker":
    results, equity_curves = {}, {}
    for ticker in tickers:
        data = load_data(ticker, start_date, end_date)
        if data.empty: continue
        if strategy == "Pivot Points":
            data = pivot_strategy(data)
        elif strategy == "SMA Crossover":
            data = sma_strategy(data, fast, slow)
        elif strategy == "RSI":
            data = rsi_strategy(data, rsi_window, rsi_buy, rsi_sell)
        trades, final_val = backtest(data, initial_cash)
        results[ticker] = {"trades": trades, "final": final_val, "data": data}
        if not trades.empty:
            equity_curves[ticker] = trades.set_index("Date")["Portfolio"]

    st.title("AI Trading Dashboard - Multi-Ticker Mode")
    perf_df = pd.DataFrame({
        t: {"Final Value": res["final"], "Return %": (res["final"] - initial_cash) / initial_cash * 100}
        for t, res in results.items()
    }).T
    if not perf_df.empty:
        st.dataframe(perf_df.style.format({"Final Value": "${:,.2f}", "Return %": "{:.2f}%"}))
    if equity_curves:
        fig_eq = go.Figure()
        for t, curve in equity_curves.items():
            fig_eq.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=t))
        fig_eq.update_layout(title="Equity Curves", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_eq, use_container_width=True)

    for ticker, res in results.items():
        st.subheader(f"{ticker} - {strategy}")
        data, trades = res["data"], res["trades"]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                     low=data["Low"], close=data["Close"], name="Candlestick"))
        buys, sells = data[data["Signal"] == "BUY"], data[data["Signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                 marker=dict(symbol="triangle-up", color="green", size=10), name="BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                 marker=dict(symbol="triangle-down", color="red", size=10), name="SELL"))
        if strategy == "SMA Crossover":
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_Fast"], mode="lines", name="Fast SMA"))
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_Slow"], mode="lines", name="Slow SMA"))
        elif strategy == "RSI":
            st.line_chart(data["RSI"], height=200)
        fig.update_layout(title=f"{ticker} Price with {strategy} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        if not trades.empty:
            st.dataframe(trades)

# ------------------------------
# Compare Strategies Mode
# ------------------------------
else:
    st.title(f"AI Trading Dashboard - Compare Strategies ({ticker})")
    data = load_data(ticker, start_date, end_date)
    strategies = {
        "Pivot Points": pivot_strategy(data),
        "SMA Crossover": sma_strategy(data, fast, slow),
        "RSI": rsi_strategy(data, rsi_window, rsi_buy, rsi_sell)
    }
    results, equity_curves = {}, {}
    for name, df in strategies.items():
        trades, final_val = backtest(df, initial_cash)
        results[name] = {"trades": trades, "final": final_val, "data": df}
        if not trades.empty:
            equity_curves[name] = trades.set_index("Date")["Portfolio"]

    perf_df = pd.DataFrame({
        s: {"Final Value": res["final"], "Return %": (res["final"] - initial_cash) / initial_cash * 100}
        for s, res in results.items()
    }).T
    st.subheader("Strategy Comparison")
    st.dataframe(perf_df.style.format({"Final Value": "${:,.2f}", "Return %": "{:.2f}%"}))

    if equity_curves:
        fig_eq = go.Figure()
        for s, curve in equity_curves.items():
            fig_eq.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=s))
        fig_eq.update_layout(title="Equity Curves by Strategy", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_eq, use_container_width=True)

    for name, res in results.items():
        st.subheader(f"{ticker} - {name}")
        df, trades = res["data"], res["trades"]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Candlestick"))
        buys, sells = df[df["Signal"] == "BUY"], df[df["Signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                 marker=dict(symbol="triangle-up", color="green", size=10), name="BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                 marker=dict(symbol="triangle-down", color="red", size=10), name="SELL"))
        if name == "SMA Crossover":
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Fast"], mode="lines", name="Fast SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Slow"], mode="lines", name="Slow SMA"))
        elif name == "RSI":
            st.line_chart(df["RSI"], height=200)
        fig.update_layout(title=f"{ticker} Price with {name} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        if not trades.empty:
            st.dataframe(trades)
