import pickle

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta
import yfinance as yf


st.set_page_config(page_title="Stock Market Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return pickle.load(open("xgb.pkl", "rb"))


@st.cache_data(ttl=900)
def load_stock_data(symbol: str, period: str):
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        return df

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    macd = ta.trend.MACD(close)

    df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["SMA50"] = close.rolling(window=50).mean()
    df["SMA100"] = close.rolling(window=100).mean()
    df.dropna(inplace=True)
    return df


def to_scalar(value):
    if hasattr(value, "iloc"):
        value = value.iloc[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return float(value)


def build_prediction_frame(df, model):
    feature_names = list(getattr(model, "feature_names_in_", ["Close AAPL", "Volume AAPL", "RSI ", "MACD "]))
    values = {
        feature_names[0]: to_scalar(df["Close"].iloc[-1]),
        feature_names[1]: to_scalar(df["Volume"].iloc[-1]),
        feature_names[2]: to_scalar(df["RSI"].iloc[-1]),
        feature_names[3]: to_scalar(df["MACD"].iloc[-1]),
    }
    return pd.DataFrame([values], columns=feature_names)


def build_analysis(latest_row, previous_row):
    close = to_scalar(latest_row["Close"])
    sma50 = to_scalar(latest_row["SMA50"])
    sma100 = to_scalar(latest_row["SMA100"])
    rsi = to_scalar(latest_row["RSI"])
    macd = to_scalar(latest_row["MACD"])
    macd_signal = to_scalar(latest_row["MACD_Signal"])
    previous_close = to_scalar(previous_row["Close"])

    if close > sma50 > sma100:
        trend_summary = "Price is above both moving averages, which suggests a bullish medium-term trend."
    elif close < sma50 < sma100:
        trend_summary = "Price is below both moving averages, which suggests a bearish medium-term trend."
    elif sma50 > sma100:
        trend_summary = "MA50 is above MA100, so the trend is still leaning bullish."
    else:
        trend_summary = "MA50 is below MA100, so the trend is still leaning bearish."

    if rsi > 70:
        rsi_summary = "RSI is over 70, which can indicate an overbought market."
    elif rsi < 30:
        rsi_summary = "RSI is below 30, which can indicate an oversold market."
    else:
        rsi_summary = "RSI is in a neutral range."

    if macd > macd_signal:
        macd_summary = "MACD is above its signal line, supporting bullish momentum."
    else:
        macd_summary = "MACD is below its signal line, showing weaker momentum."

    price_change = close - previous_close
    price_change_pct = (price_change / previous_close) * 100

    return {
        "trend": trend_summary,
        "rsi": rsi_summary,
        "macd": macd_summary,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
    }


model = load_model()

st.title("📈 Stock Market Dashboard")
st.caption("Price action, technical indicators, moving averages, and a lightweight model signal.")

with st.sidebar:
    st.header("Market Settings")
    symbol = st.text_input("Stock symbol", value="AAPL").upper().strip() or "AAPL"
    period = st.selectbox("History window", ["6mo", "1y", "2y", "5y"], index=1)

df = load_stock_data(symbol, period)

if df.empty:
    st.error(f"No data found for {symbol}. Try another ticker symbol.")
    st.stop()

latest = build_prediction_frame(df, model)
prediction = model.predict(latest)
signal = "BUY" if prediction[0] == 1 else "SELL"

latest_row = df.iloc[-1]
previous_row = df.iloc[-2] if len(df) > 1 else latest_row
analysis = build_analysis(latest_row, previous_row)

current_price = float(latest_row["Close"])
rsi = float(latest_row["RSI"])
macd = float(latest_row["MACD"])
ma50 = float(latest_row["SMA50"])
ma100 = float(latest_row["SMA100"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stock", symbol)
col2.metric("Current Price", f"${current_price:.2f}", f"{analysis['price_change']:.2f} ({analysis['price_change_pct']:.2f}%)")
col3.metric("MA50", f"${ma50:.2f}")
col4.metric("MA100", f"${ma100:.2f}")

signal_col, rsi_col, macd_col = st.columns(3)
signal_col.metric("Model Signal", signal)
rsi_col.metric("RSI", f"{rsi:.2f}")
macd_col.metric("MACD", f"{macd:.2f}")

if signal == "BUY":
    st.success(f"Prediction: {signal}")
else:
    st.error(f"Prediction: {signal}")

st.subheader("Price Trend with MA50 and MA100")
price_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
price_chart.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4", width=2)), row=1, col=1)
price_chart.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="MA50", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
price_chart.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="MA100", line=dict(color="#2ca02c", width=2)), row=1, col=1)
price_chart.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(100, 149, 237, 0.45)"), row=2, col=1)
price_chart.update_layout(height=650, legend=dict(orientation="h"), margin=dict(l=20, r=20, t=40, b=20))
price_chart.update_yaxes(title_text="Price", row=1, col=1)
price_chart.update_yaxes(title_text="Volume", row=2, col=1)
st.plotly_chart(price_chart, use_container_width=True)

st.subheader("Momentum Indicators")
indicator_chart = go.Figure()
indicator_chart.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#9467bd", width=2)))
indicator_chart.add_hline(y=70, line_dash="dash", line_color="red")
indicator_chart.add_hline(y=30, line_dash="dash", line_color="green")
indicator_chart.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="RSI")
st.plotly_chart(indicator_chart, use_container_width=True)

macd_chart = go.Figure()
macd_chart.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#17becf", width=2)))
macd_chart.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#e377c2", width=2)))
macd_chart.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="MACD")
st.plotly_chart(macd_chart, use_container_width=True)

st.subheader("Quick Analysis")
st.info(f"{analysis['trend']} {analysis['rsi']} {analysis['macd']}")

st.subheader("Latest Indicators")
st.dataframe(df[["Close", "Volume", "RSI", "MACD", "MACD_Signal", "SMA50", "SMA100"]].tail())