import re
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================================================
# Sidebar / Inputs
# =========================================================
st.sidebar.title("AI Trading Dashboard")

mode = st.sidebar.radio("Mode", ["Multi-Ticker", "Compare Strategies"])

def sanitize_symbol(s: str) -> str:
    # Keep letters, numbers, dot, dash, caret; strip everything else
    return re.sub(r"[^A-Za-z0-9\.\-\^]", "", s).upper().strip()

if mode == "Multi-Ticker":
    tickers_input = st.sidebar.text_input("Ticker Symbols (comma-separated)", value="AAPL,MSFT,TSLA")
    tickers = [sanitize_symbol(t) for t in tickers_input.split(",") if sanitize_symbol(t)]
else:
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    tickers = [sanitize_symbol(ticker)]

start_date = st.sidebar.date_input("Start Date", (datetime.now() - timedelta(days=365)).date())
today = datetime.today().date()
end_date = st.sidebar.date_input("End Date", today)
if end_date > today:
    end_date = today

initial_cash = st.sidebar.number_input("Starting Cash ($)", value=10000, step=1000)

# Strategy params (visible in both modes so Compare-mode can use them)
fast = st.sidebar.slider("Fast SMA Window", 5, 50, 10)
slow = st.sidebar.slider("Slow SMA Window", 20, 200, 30)
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
rsi_sell = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)

# =========================================================
# Utils
# =========================================================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def to_scalar(x, default=None):
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        if len(x) == 0:
            return default
        return x.iloc[0] if isinstance(x, pd.Series) else x[0]
    return x

# =========================================================
# Data Loader (defensive)
# =========================================================
@st.cache_data
def load_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end) + pd.Timedelta(days=1),  # make end inclusive
            auto_adjust=False,
            progress=False,
            group_by="column",
            interval="1d",
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_cols(df)
    # standardize column names (Open, High, Low, Close, Volume, etc.)
    df = df.rename(columns={c: c.title() for c in df.columns})
    return df

# =========================================================
# Strategies (robust)
# =========================================================
def pivot_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_cols(df.copy())
    req = {"High", "Low", "Close"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    # Compute pivots from prior day as Series (avoids DataFrame assignment error)
    pivot = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3.0
    r1 = 2 * pivot - df["Low"].shift(1)
    s1 = 2 * pivot - df["High"].shift(1)

    df["Pivot"] = pivot.astype(float)
    df["R1"] = r1.astype(float)
    df["S1"] = s1.astype(float)

    sig = pd.Series("HOLD", index=df.index)
    sig = sig.mask(df["Close"] > df["R1"], "BUY")
    sig = sig.mask(df["Close"] < df["S1"], "SELL")
    df["Signal"] = sig
    return df

def sma_strategy(df: pd.DataFrame, fast_w: int, slow_w: int) -> pd.DataFrame:
    df = flatten_cols(df.copy())
    if "Close" not in df.columns:
        return pd.DataFrame()
    df["SMA_Fast"] = df["Close"].rolling(int(fast_w)).mean()
    df["SMA_Slow"] = df["Close"].rolling(int(slow_w)).mean()
    sig = pd.Series("HOLD", index=df.index)
    sig = sig.mask(df["SMA_Fast"] > df["SMA_Slow"], "BUY")
    sig = sig.mask(df["SMA_Fast"] < df["SMA_Slow"], "SELL")
    df["Signal"] = sig
    return df

def rsi_strategy(df: pd.DataFrame, window: int, buy_th: float, sell_th: float) -> pd.DataFrame:
    df = flatten_cols(df.copy())
    if "Close" not in df.columns:
        return pd.DataFrame()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(int(window)).mean()
    loss = (-delta.clip(upper=0)).rolling(int(window)).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    sig = pd.Series("HOLD", index=df.index)
    sig = sig.mask(df["RSI"] > float(sell_th), "SELL")
    sig = sig.mask(df["RSI"] < float(buy_th), "BUY")
    df["Signal"] = sig
    return df

# =========================================================
# Backtest (scalar-safe)
# =========================================================
def backtest(df: pd.DataFrame, cash: float):
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(), float(cash)

    cash_balance = float(cash)
    position = 0
    history = []

    for date, row in df.iterrows():
        action = to_scalar(row.get("Signal", "HOLD"), default="HOLD")
        price = to_scalar(row.get("Close", np.nan), default=np.nan)
        if pd.isna(price):
            continue

        if isinstance(action, str) and action == "BUY" and cash_balance > 0:
            qty = int(cash_balance // float(price))
            if qty > 0:
                cash_balance -= qty * float(price)
                position += qty
                history.append({
                    "Date": date, "Action": "BUY", "Price": float(price),
                    "Shares": qty, "Cash": cash_balance, "Position": position,
                    "Portfolio": cash_balance + position * float(price)
                })

        elif isinstance(action, str) and action == "SELL" and position > 0:
            cash_balance += position * float(price)
            history.append({
                "Date": date, "Action": "SELL", "Price": float(price),
                "Shares": position, "Cash": cash_balance, "Position": 0,
                "Portfolio": cash_balance
            })
            position = 0

    final_value = cash_balance + position * float(df["Close"].iloc[-1])
    return pd.DataFrame(history), float(final_value)

# =========================================================
# Run modes (Multi-Ticker / Compare)
# =========================================================
st.title("AI Trading Dashboard")

exec_candidates = []  # collect latest actionable signals for the Execution panel

if mode == "Multi-Ticker":
    strategy = st.sidebar.selectbox("Select Strategy", ["Pivot Points", "SMA Crossover", "RSI"])

    results, equity_curves = {}, {}
    for tk in tickers:
        data = load_data(tk, start_date, end_date)
        if data.empty:
            continue

        if strategy == "Pivot Points":
            data = pivot_strategy(data)
        elif strategy == "SMA Crossover":
            data = sma_strategy(data, fast, slow)
        else:
            data = rsi_strategy(data, rsi_window, rsi_buy, rsi_sell)

        # build execution candidate from latest bar
        if not data.empty and "Signal" in data and "Close" in data:
            latest_idx = data.index[-1]
            sig = str(data["Signal"].iloc[-1])
            if sig in ("BUY", "SELL"):
                price = float(data["Close"].iloc[-1])
                ts_key = f"multi:{tk}:{strategy}:{latest_idx}"
                exec_candidates.append({"symbol": tk, "signal": sig, "price": price, "ts_key": ts_key})

        trades, final_val = backtest(data, initial_cash)
        results[tk] = {"trades": trades, "final": final_val, "data": data}
        if not trades.empty and "Portfolio" in trades.columns:
            equity_curves[tk] = trades.set_index("Date")["Portfolio"]

    st.subheader("Multi-Ticker Performance")
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
        if data.empty:
            st.info("No data for this ticker.")
            continue

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data.get("Open"), high=data.get("High"),
            low=data.get("Low"), close=data.get("Close"), name="Candlestick"
        ))
        buys = data[data["Signal"] == "BUY"]
        sells = data[data["Signal"] == "SELL"]
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
                st.line_chart(data["RSI"], height=180)
        fig.update_layout(title=f"{tk} Price with {strategy} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Trade Journal")
        if not trades.empty:
            st.dataframe(trades)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button(f"Download {tk} Trade Log", csv, f"{tk}_trade_log.csv", "text/csv")
        else:
            st.info("No trades executed for this ticker.")

else:
    # Compare strategies for the single selected ticker
    tk = tickers[0]
    base = load_data(tk, start_date, end_date)
    if base.empty:
        st.error("No data for this ticker/date range.")
        st.stop()

    piv = pivot_strategy(base)
    sma = sma_strategy(base, fast, slow)
    rsi = rsi_strategy(base, rsi_window, rsi_buy, rsi_sell)
    strategies = {"Pivot Points": piv, "SMA Crossover": sma, "RSI": rsi}

    # Choose first actionable strategy in priority order for execution candidate
    for name in ["Pivot Points", "SMA Crossover", "RSI"]:
        df = strategies.get(name)
        if df is None or df.empty or "Signal" not in df or "Close" not in df:
            continue
        latest_idx = df.index[-1]
        sig = str(df["Signal"].iloc[-1])
        if sig in ("BUY", "SELL"):
            price = float(df["Close"].iloc[-1])
            ts_key = f"compare:{tk}:{name}:{latest_idx}"
            exec_candidates.append({"symbol": tk, "signal": sig, "price": price, "ts_key": ts_key})
            break

    # Results & display
    results, equity_curves = {}, {}
    for name, df in strategies.items():
        trades, final_val = backtest(df, initial_cash)
        results[name] = {"trades": trades, "final": final_val, "data": df}
        if not trades.empty and "Portfolio" in trades.columns:
            equity_curves[name] = trades.set_index("Date")["Portfolio"]

    st.subheader(f"Strategy Comparison — {tk}")
    perf_df = pd.DataFrame({
        s: {"Final Value": res["final"], "Return %": (res["final"] - initial_cash) / initial_cash * 100}
        for s, res in results.items()
    }).T
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

    for name, df in strategies.items():
        st.subheader(f"{tk} - {name}")
        if df.empty:
            st.info("No data for this strategy.")
            continue

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df.get("Open"), high=df.get("High"),
            low=df.get("Low"), close=df.get("Close"), name="Candlestick"
        ))
        buys = df[df["Signal"] == "BUY"]
        sells = df[df["Signal"] == "SELL"]
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
                st.line_chart(df["RSI"], height=180)
        fig.update_layout(title=f"{tk} Price with {name} Signals", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Alpaca (alpaca-py) helpers
# =========================================================
def get_alpaca_client():
    try:
        from alpaca.trading.client import TradingClient
    except Exception as e:
        raise RuntimeError("alpaca-py not installed. Add 'alpaca-py' to requirements.txt") from e
    cfg = st.secrets.get("alpaca", {})
    key = cfg.get("key")
    sec = cfg.get("secret")
    paper = bool(cfg.get("paper", True))
    if not key or not sec:
        raise RuntimeError("Missing Alpaca keys in secrets.")
    return TradingClient(key, sec, paper=paper)

def alpaca_account_info(client):
    try:
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": acct.status,
        }
    except Exception as e:
        st.error(f"Alpaca account error: {e}")
        return None

def alpaca_get_position_value(client, symbol):
    try:
        pos = client.get_position(symbol)
        return float(pos.market_value)
    except Exception:
        return 0.0  # no position

def alpaca_market_order(client, symbol, side, qty=None, notional=None, tif="day"):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    kwargs = {
        "symbol": symbol,
        "side": OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        "time_in_force": TimeInForce.DAY if tif == "day" else TimeInForce.GTC,
    }
    if notional is not None:
        kwargs["notional"] = float(notional)
    elif qty is not None:
        kwargs["qty"] = int(qty)
    else:
        raise ValueError("Provide qty or notional for order sizing.")
    req = MarketOrderRequest(**kwargs)
    return client.submit_order(req)

# =========================================================
# Execution Panel (Paper Trading)
# =========================================================
st.header("Execution (Paper Trading)")

colA, colB = st.columns([1,1])
with colA:
    arm = st.toggle("Arm Autotrader (safety switch)", value=False, help="Must be ON to send orders.")
with colB:
    use_notional = st.toggle("Size by Notional $ (vs. shares)", value=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    notional_usd = st.number_input("Notional per trade ($)", min_value=100.0, value=1000.0, step=100.0)
with col2:
    qty_shares = st.number_input("OR Shares per trade", min_value=0, value=0, step=1)
with col3:
    max_pos_usd = st.number_input("Max position per symbol ($ cap)", min_value=0.0, value=5000.0, step=500.0)

# Duplicate protection per latest bar key
if "last_exec_key" not in st.session_state:
    st.session_state["last_exec_key"] = {}

# Show latest actionable signals
if exec_candidates:
    st.write("Latest actionable signals:")
    for c in exec_candidates:
        st.write(f"- **{c['symbol']}**: {c['signal']} @ ${c['price']:.2f}")
else:
    st.info("No actionable signals on the latest bar.")

# Connect to Alpaca (if secrets exist)
client = None
try:
    client = get_alpaca_client()
    acct = alpaca_account_info(client)
    if acct:
        st.caption(
            f"Alpaca (paper) — Equity: ${acct['equity']:,.2f} | Cash: ${acct['cash']:,.2f} | "
            f"BP: ${acct['buying_power']:,.2f} | Status: {acct['status']}"
        )
except Exception as e:
    st.info(f"Alpaca disabled: {e}")

can_execute = arm and (client is not None) and len(exec_candidates) > 0

if st.button("Execute Latest Signal Now", disabled=not can_execute,
             help="Sends market order(s) on the newest bar signals."):
    if not arm:
        st.warning("Autotrader is not armed.")
    elif client is None:
        st.error("Broker not connected.")
    else:
        executed_any = False
        for c in exec_candidates:
            key = c["ts_key"]
            # skip duplicates for the same latest bar
            if st.session_state["last_exec_key"].get(key):
                st.info(f"Skipped duplicate: {key}")
                continue

            # cap by current position market value
            current_val = alpaca_get_position_value(client, c["symbol"])
            if max_pos_usd > 0 and current_val >= max_pos_usd:
                st.warning(
                    f"Skipped {c['symbol']}: position ${current_val:,.2f} is at/over cap ${max_pos_usd:,.2f}."
                )
                continue

            # determine sizing
            try:
                if use_notional:
                    alpaca_market_order(client, c["symbol"], c["signal"], notional=notional_usd)
                else:
                    if qty_shares <= 0:
                        st.warning("Shares per trade is 0. Set shares or switch to notional.")
                        continue
                    alpaca_market_order(client, c["symbol"], c["signal"], qty=qty_shares)
                st.success(f"Order sent: {c['symbol']} {c['signal']}")
                st.session_state["last_exec_key"][key] = True
                executed_any = True
            except Exception as e:
                st.error(f"Order error for {c['symbol']}: {e}")

        if not executed_any:
            st.info("No orders sent.")
