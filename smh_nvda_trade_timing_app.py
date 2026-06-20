# smh_nvda_trade_timing_app.py
# Streamlit app for SMH + NVDA trade timing, position sizing, buy/sell levels
# Educational tool only. Not financial advice.

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="SMH + NVDA Trade Timing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f7f9fb;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e8eef5;
        padding: 14px 16px;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 10px;
    }
    .big-signal {
        font-size: 1.6rem;
        font-weight: 800;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0 12px 0;
    }
    .buy {background: #eaf7ef; border: 1px solid #b9e3c5;}
    .watch {background: #fff8e6; border: 1px solid #f3d37a;}
    .sell {background: #fdecec; border: 1px solid #edb6b6;}
    .hold {background: #edf2ff; border: 1px solid #bccbff;}
    .small-note {font-size: 0.88rem; color: #5b6775;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=300)
def load_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna()
    return data


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["RSI14"] = rsi(out["Close"])
    out["ATR14"] = atr(out)
    out["ATR_PCT"] = out["ATR14"] / out["Close"] * 100
    out["52W_HIGH_PROXY"] = out["Close"].rolling(126, min_periods=20).max()
    out["DIST_FROM_HIGH"] = (out["Close"] / out["52W_HIGH_PROXY"] - 1) * 100
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    return out


def dollar_to_shares(dollars: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, math.floor(dollars / price))


def classify_signal(price, ema20, ema50, rsi14, atr_pct, dist_high):
    score = 0
    reasons = []

    if price > ema20:
        score += 2
        reasons.append("price above EMA20")
    else:
        reasons.append("price below EMA20")

    if ema20 > ema50:
        score += 2
        reasons.append("EMA20 above EMA50")
    else:
        reasons.append("EMA20 below EMA50")

    if 42 <= rsi14 <= 62:
        score += 2
        reasons.append("RSI in healthy buy/hold zone")
    elif rsi14 < 35:
        score += 1
        reasons.append("RSI oversold, wait for rebound")
    elif rsi14 > 70:
        reasons.append("RSI hot, avoid chasing")
    else:
        score += 1
        reasons.append("RSI acceptable")

    if atr_pct < 4.5:
        score += 1
        reasons.append("volatility manageable")
    else:
        reasons.append("volatility elevated")

    if -12 <= dist_high <= -3:
        score += 2
        reasons.append("meaningful pullback from recent high")
    elif dist_high > -3:
        reasons.append("near recent high")
    else:
        score += 1
        reasons.append("deep pullback, confirm trend first")

    if score >= 7:
        label = "BUY / ADD"
        css = "buy"
    elif score >= 5:
        label = "WATCH / SMALL BUY"
        css = "watch"
    elif rsi14 > 70 or price < ema50:
        label = "REDUCE / WAIT"
        css = "sell"
    else:
        label = "HOLD"
        css = "hold"

    return score, label, css, reasons


def position_plan(
    ticker: str,
    price: float,
    avg_cost: float,
    current_shares: int,
    account_value: float,
    cash_available: float,
    smh_target_pct: float,
    nvda_target_pct: float,
    cash_target_pct: float,
):
    current_value = current_shares * price
    pnl_pct = 0 if avg_cost <= 0 or current_shares <= 0 else (price / avg_cost - 1) * 100

    if ticker == "SMH":
        base_target = account_value * smh_target_pct
        max_target = account_value * 0.80
        # Core + pullback add plan
        add_5 = 0.05
        add_10 = 0.10
        add_15 = 0.15
        buy_levels = [
            ("First add", avg_cost * (1 - add_5) if avg_cost > 0 else price * 0.95, min(500, cash_available)),
            ("Second add", avg_cost * (1 - add_10) if avg_cost > 0 else price * 0.90, min(500, cash_available)),
            ("Third add", avg_cost * (1 - add_15) if avg_cost > 0 else price * 0.85, min(1000, cash_available)),
        ]
        sell_levels = [
            ("Trim 20%", avg_cost * 1.15 if avg_cost > 0 else price * 1.15, 0.20),
            ("Trim 20%", avg_cost * 1.25 if avg_cost > 0 else price * 1.25, 0.20),
            ("Trim 20%", avg_cost * 1.35 if avg_cost > 0 else price * 1.35, 0.20),
        ]
        stop_level = avg_cost * 0.88 if avg_cost > 0 else price * 0.88
        note = "SMH is core position: buy pullbacks, trim into strength, avoid full exit unless trend breaks."
    else:
        base_target = account_value * nvda_target_pct
        max_target = account_value * 0.25
        buy_levels = [
            ("Tactical buy", price * 0.98, min(base_target - current_value, cash_available) if current_value < base_target else 0),
        ]
        sell_levels = [
            ("Sell 50%", avg_cost * 1.10 if avg_cost > 0 else price * 1.10, 0.50),
            ("Sell remaining 50%", avg_cost * 1.20 if avg_cost > 0 else price * 1.20, 0.50),
        ]
        stop_level = avg_cost * 0.93 if avg_cost > 0 else price * 0.93
        note = "NVDA is tactical position: smaller size, faster profit-taking, strict stop-loss."

    target_gap = max(0, min(base_target, max_target) - current_value)
    suggested_buy_dollars = min(target_gap, cash_available)
    suggested_buy_shares = dollar_to_shares(suggested_buy_dollars, price)

    return {
        "current_value": current_value,
        "pnl_pct": pnl_pct,
        "base_target": base_target,
        "max_target": max_target,
        "suggested_buy_dollars": suggested_buy_dollars,
        "suggested_buy_shares": suggested_buy_shares,
        "buy_levels": buy_levels,
        "sell_levels": sell_levels,
        "stop_level": stop_level,
        "note": note,
    }


def make_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{ticker} Price with EMA20 / EMA50",
        xaxis_rangeslider_visible=False,
    )
    return fig


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Account Settings")

account_value = st.sidebar.number_input("Total short-term account value ($)", min_value=1000.0, value=10000.0, step=500.0)
cash_available = st.sidebar.number_input("Current cash available ($)", min_value=0.0, value=2000.0, step=100.0)

st.sidebar.subheader("Target Allocation")
smh_target_pct = st.sidebar.slider("SMH target %", 0, 90, 60) / 100
nvda_target_pct = st.sidebar.slider("NVDA tactical %", 0, 50, 20) / 100
cash_target_pct = max(0, 1 - smh_target_pct - nvda_target_pct)
st.sidebar.caption(f"Implied cash target: {cash_target_pct:.0%}")

st.sidebar.subheader("Your Current Positions")
smh_shares = st.sidebar.number_input("SMH shares", min_value=0, value=0, step=1)
smh_avg_cost = st.sidebar.number_input("SMH average cost ($)", min_value=0.0, value=0.0, step=1.0)

nvda_shares = st.sidebar.number_input("NVDA shares", min_value=0, value=0, step=1)
nvda_avg_cost = st.sidebar.number_input("NVDA average cost ($)", min_value=0.0, value=0.0, step=1.0)

period = st.sidebar.selectbox("Chart period", ["3mo", "6mo", "1y"], index=1)
refresh = st.sidebar.button("Refresh data")

st.title("📈 SMH + NVDA Trade Timing App")
st.caption("Purpose: show buy points, buy amount/shares, sell target price, sell amount/shares, and stop-loss levels. Educational use only.")

if refresh:
    st.cache_data.clear()

tickers = {
    "SMH": {"shares": smh_shares, "avg_cost": smh_avg_cost},
    "NVDA": {"shares": nvda_shares, "avg_cost": nvda_avg_cost},
}

summary_rows = []

for ticker, pos in tickers.items():
    try:
        raw = load_data(ticker, period=period, interval="1d")
        df = add_indicators(raw)
        latest = df.iloc[-1]

        price = float(latest["Close"])
        ema20 = float(latest["EMA20"])
        ema50 = float(latest["EMA50"])
        rsi14 = float(latest["RSI14"])
        atr_pct = float(latest["ATR_PCT"])
        dist_high = float(latest["DIST_FROM_HIGH"])

        score, label, css, reasons = classify_signal(price, ema20, ema50, rsi14, atr_pct, dist_high)

        plan = position_plan(
            ticker=ticker,
            price=price,
            avg_cost=pos["avg_cost"],
            current_shares=pos["shares"],
            account_value=account_value,
            cash_available=cash_available,
            smh_target_pct=smh_target_pct,
            nvda_target_pct=nvda_target_pct,
            cash_target_pct=cash_target_pct,
        )

        st.markdown(f"## {ticker}")
        st.markdown(f"<div class='big-signal {css}'>{label} — Score {score}/9</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${price:,.2f}")
        c2.metric("RSI14", f"{rsi14:.1f}")
        c3.metric("ATR %", f"{atr_pct:.2f}%")
        c4.metric("Distance from High", f"{dist_high:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Current Value", f"${plan['current_value']:,.0f}")
        c6.metric("P/L vs Avg Cost", f"{plan['pnl_pct']:.1f}%")
        c7.metric("Suggested Buy $", f"${plan['suggested_buy_dollars']:,.0f}")
        c8.metric("Suggested Buy Shares", f"{plan['suggested_buy_shares']}")

        with st.expander(f"{ticker} trading plan details", expanded=True):
            st.write(plan["note"])
            st.markdown("### Buy / Add Plan")
            buy_rows = []
            for name, level_price, amount in plan["buy_levels"]:
                shares = dollar_to_shares(max(0, amount), level_price)
                buy_rows.append({
                    "Action": name,
                    "Trigger Price": f"${level_price:,.2f}",
                    "Buy Amount": f"${max(0, amount):,.0f}",
                    "Approx Shares": shares,
                })
            st.dataframe(pd.DataFrame(buy_rows), use_container_width=True, hide_index=True)

            st.markdown("### Sell / Trim Plan")
            sell_rows = []
            for name, level_price, pct_to_sell in plan["sell_levels"]:
                shares_to_sell = math.floor(pos["shares"] * pct_to_sell)
                dollars_to_sell = shares_to_sell * level_price
                sell_rows.append({
                    "Action": name,
                    "Target Price": f"${level_price:,.2f}",
                    "Sell Shares": shares_to_sell,
                    "Approx Sell Value": f"${dollars_to_sell:,.0f}",
                })
            st.dataframe(pd.DataFrame(sell_rows), use_container_width=True, hide_index=True)

            st.markdown("### Risk Control")
            st.write(f"Stop-loss reference level: **${plan['stop_level']:,.2f}**")
            st.write("Reasons:", ", ".join(reasons))

        st.plotly_chart(make_chart(df.tail(120), ticker), use_container_width=True, key=f"chart_{ticker}")

        summary_rows.append({
            "Ticker": ticker,
            "Signal": label,
            "Score": score,
            "Price": price,
            "Suggested Buy $": plan["suggested_buy_dollars"],
            "Suggested Buy Shares": plan["suggested_buy_shares"],
            "Stop Level": plan["stop_level"],
        })

    except Exception as e:
        st.error(f"Could not load {ticker}: {e}")

st.markdown("---")
st.subheader("Portfolio Action Summary")
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    ### Thomas Rule Set Used in This App

    **Default allocation:** SMH 60%, NVDA 20%, Cash 20%.

    **SMH:** core position. Add gradually on pullbacks. Trim into strength.  
    **NVDA:** tactical position. Smaller size. Take profit faster. Stop-loss tighter.

    This app is for educational decision support only. It does not place trades and does not guarantee profit.
    """
)
