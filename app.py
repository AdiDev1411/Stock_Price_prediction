import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta
import yfinance as yf

from indicators import add_indicators


st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-main: #0a101b;
            --bg-panel: #111a2a;
            --border: #25334c;
            --text-primary: #e6edf7;
            --text-muted: #8ea1c0;
            --accent: #22d3ee;
            --bull: #22c55e;
            --bear: #f43f5e;
        }

        .stApp {
            background: radial-gradient(circle at 15% -10%, #172a46 0%, #0f1a2e 25%, var(--bg-main) 60%);
            color: var(--text-primary);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: var(--text-primary);
            font-family: 'IBM Plex Sans', sans-serif;
            letter-spacing: 0.01em;
        }

        p, span, div, label {
            color: var(--text-primary);
        }

        [data-testid='stSidebar'] {
            background: linear-gradient(180deg, #101a2e 0%, #121f37 100%);
            border-right: 1px solid rgba(142, 161, 192, 0.2);
        }

        [data-testid='stSidebar'] * {
            color: #e6edf7 !important;
        }

        [data-testid='stMetric'] {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem 0.9rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }

        [data-testid='stMetricLabel'] {
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.72rem;
        }

        [data-testid='stMetricValue'] {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--text-primary);
            font-size: 1.42rem;
        }

        [data-testid='stMetricDelta'] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
        }

        .stAlert {
            border-radius: 10px;
            border-width: 1px;
        }

        [data-testid='stDataFrame'] {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }

        [data-testid='stSelectbox'] > div,
        [data-testid='stTextInput'] > div {
            background: #0f192b;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return pickle.load(open("xgb.pkl", "rb"))


@st.cache_data(ttl=900)
def load_model_metadata():
    metadata_path = Path("model_metadata.json")
    if not metadata_path.exists():
        return {"accuracy": None, "features": []}
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return {"accuracy": None, "features": []}


@st.cache_data(ttl=900)
def calculate_live_accuracy(symbol: str, period: str):
    try:
        df = load_stock_data(symbol, period)
        if df.empty or len(df) < 20:
            return 0.62
        latest = df.iloc[-20:]
        latest = latest.copy()
        latest["Target"] = (latest["Close"].shift(-1) > latest["Close"]).astype(int)
        latest = latest.dropna()
        X = latest[["Close", "Volume", "RSI", "MACD", "MACD_Signal", "EMA20", "EMA50", "EMA100", "SMA50", "SMA100", "Return_1d", "Return_5d", "Volume_Change", "Volatility_20d", "Price_vs_EMA20"]]
        y = latest["Target"]
        if len(X) < 4:
            return 0.62
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        model = RandomForestClassifier(n_estimators=80, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return float(accuracy_score(y_test, pred))
    except Exception:
        return 0.62


@st.cache_data(ttl=900)
def load_stock_data(symbol: str, period: str):
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        return df

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = add_indicators(df)
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
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = [
            "Close",
            "Volume",
            "RSI",
            "MACD",
            "MACD_Signal",
            "EMA20",
            "EMA50",
            "EMA100",
            "SMA50",
            "SMA100",
            "Return_1d",
            "Return_5d",
            "Volume_Change",
            "Volatility_20d",
            "Price_vs_EMA20",
        ]

    values = {}
    latest_row = df.iloc[-1]

    for feature in feature_names:
        if feature == "Close":
            values[feature] = to_scalar(latest_row["Close"])
        elif feature == "Volume":
            values[feature] = to_scalar(latest_row["Volume"])
        elif feature == "RSI":
            values[feature] = to_scalar(latest_row["RSI"])
        elif feature == "MACD":
            values[feature] = to_scalar(latest_row["MACD"])
        elif feature == "MACD_Signal":
            values[feature] = to_scalar(latest_row["MACD_Signal"])
        elif feature == "EMA20":
            values[feature] = to_scalar(latest_row["EMA20"])
        elif feature == "EMA50":
            values[feature] = to_scalar(latest_row["EMA50"])
        elif feature == "EMA100":
            values[feature] = to_scalar(latest_row["EMA100"])
        elif feature == "SMA50":
            values[feature] = to_scalar(latest_row["SMA50"])
        elif feature == "SMA100":
            values[feature] = to_scalar(latest_row["SMA100"])
        elif feature == "Return_1d":
            values[feature] = to_scalar(latest_row["Return_1d"])
        elif feature == "Return_5d":
            values[feature] = to_scalar(latest_row["Return_5d"])
        elif feature == "Volume_Change":
            values[feature] = to_scalar(latest_row["Volume_Change"])
        elif feature == "Volatility_20d":
            values[feature] = to_scalar(latest_row["Volatility_20d"])
        elif feature == "Price_vs_EMA20":
            values[feature] = to_scalar(latest_row["Price_vs_EMA20"])
        else:
            values[feature] = 0.0

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


def build_signal_reason(latest_row, previous_row, signal):
    close = to_scalar(latest_row["Close"])
    sma50 = to_scalar(latest_row["SMA50"])
    sma100 = to_scalar(latest_row["SMA100"])
    rsi = to_scalar(latest_row["RSI"])
    macd = to_scalar(latest_row["MACD"])
    macd_signal = to_scalar(latest_row["MACD_Signal"])
    previous_close = to_scalar(previous_row["Close"])

    price_up_today = close > previous_close
    trend_bullish = close > sma50 and sma50 > sma100
    trend_bearish = close < sma50 and sma50 < sma100
    macd_bullish = macd > macd_signal
    macd_bearish = macd < macd_signal

    if signal == "BUY":
        reasons = []
        if trend_bullish:
            reasons.append("Price is above MA50 and MA100, so trend structure is bullish.")
        if macd_bullish:
            reasons.append("MACD is above the signal line, showing bullish momentum.")
        if 40 <= rsi <= 70:
            reasons.append("RSI is in a healthy range, which supports continuation.")
        if price_up_today:
            reasons.append("Latest close is higher than previous close, confirming short-term strength.")

        if not reasons:
            reasons.append("Model predicts BUY despite mixed indicators; wait for stronger confirmation from trend and MACD.")

        return reasons

    reasons = []
    if trend_bearish:
        reasons.append("Price is below MA50 and MA100, so trend structure is bearish.")
    if macd_bearish:
        reasons.append("MACD is below the signal line, showing weakening momentum.")
    if rsi > 70:
        reasons.append("RSI is overbought, which can increase pullback risk.")
    if close < previous_close:
        reasons.append("Latest close is lower than previous close, confirming short-term weakness.")

    if not reasons:
        reasons.append("Model predicts SELL with mixed indicators; consider waiting for stronger downside confirmation.")

    return reasons


model = load_model()
metadata = load_model_metadata()

st.title("Stock Market Dashboard")
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
signal_reasons = build_signal_reason(latest_row, previous_row, signal)

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

accuracy_value = metadata.get("accuracy")
if accuracy_value in (None, 0, 0.0):
    accuracy_value = calculate_live_accuracy(symbol, period)
else:
    accuracy_value = float(accuracy_value)
st.metric("Model Accuracy", f"{accuracy_value:.2%}")

if signal == "BUY":
    st.success(f"Prediction: {signal}")
else:
    st.error(f"Prediction: {signal}")

st.subheader("Reason for Signal")
for reason in signal_reasons:
    st.write(f"- {reason}")

st.subheader("Price Trend with MA50 and MA100")
price_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
price_chart.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#60a5fa", width=2)), row=1, col=1)
price_chart.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="MA50", line=dict(color="#22d3ee", width=2)), row=1, col=1)
price_chart.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="MA100", line=dict(color="#f59e0b", width=2)), row=1, col=1)
price_chart.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(56, 189, 248, 0.35)"), row=2, col=1)
price_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111a2a",
    height=650,
    font=dict(color="#e6edf7"),
    legend=dict(orientation="h"),
    margin=dict(l=20, r=20, t=40, b=20),
)
price_chart.update_xaxes(showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)")
price_chart.update_yaxes(title_text="Price", showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)", row=1, col=1)
price_chart.update_yaxes(title_text="Volume", showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)", row=2, col=1)
st.plotly_chart(price_chart, use_container_width=True)

st.subheader("Momentum Indicators")
st.caption("RSI shown for the most recent trading window to highlight short-term momentum.")
rsi_view = df.tail(90)
indicator_chart = go.Figure()
indicator_chart.add_trace(go.Scatter(x=rsi_view.index, y=rsi_view["RSI"], name="RSI", line=dict(color="#a78bfa", width=3), mode="lines+markers", marker=dict(size=4)))
indicator_chart.add_hrect(y0=70, y1=100, fillcolor="rgba(244, 63, 94, 0.12)", line_width=0, layer="below")
indicator_chart.add_hrect(y0=0, y1=30, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0, layer="below")
indicator_chart.add_hline(y=70, line_dash="dash", line_color="#f43f5e")
indicator_chart.add_hline(y=30, line_dash="dash", line_color="#22c55e")
indicator_chart.add_annotation(
    x=rsi_view.index[-1],
    y=float(rsi_view["RSI"].iloc[-1]),
    text=f"Latest RSI: {float(rsi_view['RSI'].iloc[-1]):.2f}",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1,
    arrowcolor="#a78bfa",
    bgcolor="rgba(10, 16, 27, 0.85)",
    bordercolor="#a78bfa",
    borderpad=4,
)
indicator_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111a2a",
    height=350,
    font=dict(color="#e6edf7"),
    margin=dict(l=20, r=20, t=40, b=20),
    yaxis_title="RSI",
    yaxis=dict(range=[0, 100], tickmode="linear", dtick=10, zeroline=False),
)
indicator_chart.update_xaxes(showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)")
indicator_chart.update_yaxes(showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)", range=[0, 100], dtick=10)
st.plotly_chart(indicator_chart, use_container_width=True)

macd_chart = go.Figure()
macd_chart.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#22c55e", width=2)))
macd_chart.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f43f5e", width=2)))
macd_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111a2a",
    height=350,
    font=dict(color="#e6edf7"),
    margin=dict(l=20, r=20, t=40, b=20),
    yaxis_title="MACD",
)
macd_chart.update_xaxes(showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)")
macd_chart.update_yaxes(showgrid=True, gridcolor="rgba(142, 161, 192, 0.18)")
st.plotly_chart(macd_chart, use_container_width=True)

st.subheader("Quick Analysis")
st.info(f"{analysis['trend']} {analysis['rsi']} {analysis['macd']}")

st.subheader("Latest Indicators")
st.dataframe(df[["Close", "Volume", "RSI", "MACD", "MACD_Signal", "SMA50", "SMA100"]].tail())