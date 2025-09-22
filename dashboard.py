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
    strategy = None  # compare all

initial_cash = st.sidebar.number_input("Starting Cash ($)", value=10000, step=1000)

# Strategy parameters (always visible so they work in both modes)
fast = st.sidebar.slider("Fast SMA Window", 5, 50, 10)
slow = st.sidebar.slider("Slow SMA Window", 20, 200, 30)
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
rsi_sell = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)

# ------------------------------
# Utils
# ------------------------------
def to_scalar(x, default=None):
    """Return a scalar from possibly list-like/Series/ndarray; fallback to default."""
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        if len(x) == 0:
            return default
        return x.iloc[0] if isinstance(x, pd.Series) else x[0]
    return x

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
        df.columns = [c[0] for c in df.columns]
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["Pivot"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    df["R1"] = 2 * df["Pivot"] - df["Low"].shift(1)
    df["S1"] = 2 * df["Pivot"] - df["High"].shift(1)
    df["Signal"] = "HOLD"
    df.loc[df["Close"] > df["R1"], "Signal"] = "BUY"
    df.loc[df["Close"] < df["S1"], "Signal"] = "SELL"
    return df

def sma_strategy(df, fast, slow):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if "Close" not in df.columns:
        return pd.DataFrame()
    df["SMA_Fast"] = df["Close"].rolling(int(fast)).mean()
    df["SMA_Slow"] = df["Close"].rolling(int(slow)).mean()
    df["Signal"] = "HOLD"
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = "BUY"
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = "SELL"
    return df

def rsi_strategy(df, window, buy_th, sell_th):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if "Close" not in df.columns:
        return pd.DataFrame()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(window)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(window)).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Signal"] = "HOLD"
    df.loc[df["RSI"] > float(sell_th), "Signal"] = "SELL"
    df.loc[df["RSI"] < float(buy_th), "Signal"] = "BUY"
    return df

# ------------------------------
# Backtesting
# ------------------------------
def backtest(df, cash):
    if df is None or df.empty:
        return pd.DataFrame(), float(cash)

    cash_balance = float(cash)
    position = 0
    history = []

    for date, row in df.iterrows():
        action = to_scalar(row.get("Signal", "HOLD"), default="HOLD")
        price = to_scalar(row.get("Close", np.nan), default=np.nan)

        # Skip rows without a valid price
        if pd.isna(price):
            continue

        is_buy = isinstance(action, str) and action == "BUY"
        is_sell = isinstance(action, str) and action == "SELL"

        if is_buy and cash_balance > 0:
            qty = int(cash_balance // float(price))
            if qty > 0:
                cash_balance -= qty * float(price)
                position += qty
                history.append({
                    "Date": date, "Action": "BUY", "Price": float(price),
                    "Shares": qty, "Cash": cash_balance, "Position": position,
                    "Portfolio": cash_balance + position * float(price)
                })

        elif is_sell and position > 0:
            cash_balance += position * float(price)
            history.append({
                "Date": date, "Action": "SELL", "Price": float(price),
                "Shares": position, "Cash": cash_balance, "Position": 0,
                "Portfolio": cash_balance
            })
            position = 0

    final_value = cash_balance + position * float(df["Close"].iloc[-1])
    return pd.DataFrame(history), float(final_value)

# ------------------------------
# Multi-Ticker Mode
# ------------------------------
if mode == "Multi-Ticker":
    results, equity_curves = {}, {}

    for tk in tickers:
        data = load_data(tk, start_date, end_date)
        if data is None or data.empty:
            continue

        if strategy == "Pivot Points":
            data = pivot_strategy(data)
        elif strategy == "SMA Crossover":
            data = sma_strategy(data, fast, slow)
        elif strategy == "RSI":
            data = rsi_strategy(data, rsi_window, rsi_buy, rsi_sell)

        trades, final_val = backtest(data, initial_cash)
        results[tk] = {"trades": trades, "final": final_val, "data": data}

        if not trades.empty and "Portfolio" in trades.columns:
            equity_curves[tk] = trades.set_index("Date")["Portfolio"]

    st.title("AI Trading Dashboard - Multi-Ticker Mode")

    perf_df = pd.DataFrame({
        t: {"Final Value": res["final"], "Return %": (res["final"] - initial_cash) / initial_cash * 100}
        for t, res in results.items()
    }).T

    if not perf_df.empty:
        st.dataframe(perf_df.style.format({"Final Value": "${:,.2f}", "Return %": "{:.2f}%"}))
        best_ticker = perf_df["Return %"].idxmax()
        st.success(f"Best performer: **{best_ticker}** ({perf_df.loc[best_ticker, 'Return %']:.2f}%)")
    else:
        st.warning("No valid data fetched. Try different tickers or dates.")

    if equity_curves:
        fig_eq = go.Figure()
        for t, curve in equity_curves.items():
            fig_eq.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=t))
        fig_eq.update_layout(title="Equity Curves", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_eq, use_container_width=True)

    for tk, res in results.items():
        st.subheader(f"{tk} - {strategy}")
        data, trades = res["data"], res["trades"]

        if data is None or data.empty:
            st.info("No data for this ticker.")
            continue

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data.get("Open", pd.Series(index=data.index)),
                                     high=data.get("High", pd.Series(index=data.index)),
                                     low=data.get("Low", pd.Series(index=data.index)),
                                     close=data.get("Close", pd.Series(index=data.index)),
                                     name="Candlestick"))
        buys, sells = data[data["Signal"] == "BUY"], data[data["Signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                 marker=dict(symbol="triangle-up", color="green", size=10), name="BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                 marker=dict(symbol="triangle-down", color="red", size=10), name="SELL"))

        if strategy == "SMA Crossover":
            if "SMA_Fast" in data and "SMA_Slow" in data:
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA_Fast"], mode="lines", name="Fast SMA"))
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA_Slow"], mode="lines", name="Slow SMA"))
        elif strategy == "RSI":
            if "RSI" in data:
                st.line_chart(data["RSI"], height=200)

        fig.update_layout(title=f"{tk} Price with {strategy} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Journal")
        if not trades.empty:
            st.dataframe(trades)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button(f"Download {tk} Trade Log", csv, f"{tk}_trade_log.csv", "text/csv")
        else:
            st.info("No trades executed for this ticker.")

# ------------------------------
# Compare Strategies Mode
# ------------------------------
else:
    st.title(f"AI Trading Dashboard - Compare Strategies ({ticker})")
    base = load_data(ticker, start_date, end_date)
    if base is None or base.empty:
        st.error("No data for this ticker/date range.")
        st.stop()

    piv = pivot_strategy(base)
    sma = sma_strategy(base, fast, slow)
    rsi = rsi_strategy(base, rsi_window, rsi_buy, rsi_sell)

    strategies = {"Pivot Points": piv, "SMA Crossover": sma, "RSI": rsi}
    results, equity_curves = {}, {}

    for name, df in strategies.items():
        trades, final_val = backtest(df, initial_cash)
        results[name] = {"trades": trades, "final": final_val, "data": df}
        if not trades.empty and "Portfolio" in trades.columns:
            equity_curves[name] = trades.set_index("Date")["Portfolio"]

    perf_df = pd.DataFrame({
        s: {"Final Value": res["final"], "Return %": (res["final"] - initial_cash) / initial_cash * 100}
        for s, res in results.items()
    }).T

    st.subheader("Strategy Comparison")
    if not perf_df.empty:
        st.dataframe(perf_df.style.format({"Final Value": "${:,.2f}", "Return %": "{:.2f}%"}))
        best_strat = perf_df["Return %"].idxmax()
        st.success(f"Best strategy: **{best_strat}** ({perf_df.loc[best_strat, 'Return %']:.2f}%)")
    else:
        st.info("No trades generated for any strategy in this window.")

    if equity_curves:
        fig_eq = go.Figure()
        for s, curve in equity_curves.items():
            fig_eq.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=s))
        fig_eq.update_layout(title="Equity Curves by Strategy", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_eq, use_container_width=True)

    for name, res in results.items():
        st.subheader(f"{ticker} - {name}")
        df, trades = res["data"], res["trades"]
        if df is None or df.empty:
            st.info("No data for this strategy.")
            continue

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df.get("Open", pd.Series(index=df.index)),
                                     high=df.get("High", pd.Series(index=df.index)),
                                     low=df.get("Low", pd.Series(index=df.index)),
                                     close=df.get("Close", pd.Series(index=df.index)),
                                     name="Candlestick"))
        buys, sells = df[df["Signal"] == "BUY"], df[df["Signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                 marker=dict(symbol="triangle-up", color="green", size=10), name="BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                 marker=dict(symbol="triangle-down", color="red", size=10), name="SELL"))

        if name == "SMA Crossover":
            if "SMA_Fast" in df and "SMA_Slow" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Fast"], mode="lines", name="Fast SMA"))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Slow"], mode="lines", name="Slow SMA"))
        elif name == "RSI":
            if "RSI" in df:
                st.line_chart(df["RSI"], height=200)

        fig.update_layout(title=f"{ticker} Price with {name} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Journal")
        if not trades.empty:
            st.dataframe(trades)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button(f"Download {name} Trade Log", csv, f"{ticker}_{name}_trade_log.csv", "text/csv")
        else:
            st.info("No trades executed for this strategy.")

