import streamlit as st
import yfinance as yf
import ta
import pickle

# --------------------
# Load trained model
# --------------------
model = pickle.load(open("xgb.pkl", "rb"))

# --------------------
# Download stock data
# --------------------
df = yf.download("AAPL", period="3mo")

# Convert Close column to Series
close = df["Close"].squeeze()

# --------------------
# Calculate indicators
# --------------------
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MACD"] = ta.trend.MACD(close).macd()

# Remove NaN rows
df.dropna(inplace=True)

# --------------------
# Features for prediction
# --------------------
latest = df[["Close", "Volume", "RSI", "MACD"]].iloc[-1:]

# Predict
prediction = model.predict(latest)

signal = "BUY" if prediction[0] == 1 else "SELL"

# --------------------
# Current values
# --------------------
current_price = float(close.iloc[-1])
rsi = float(df["RSI"].iloc[-1])
macd = float(df["MACD"].iloc[-1])

# --------------------
# Streamlit UI
# --------------------
st.title("📈 Real-Time Stock Market Prediction")

st.metric("Stock", "AAPL")
st.metric("Current Price", f"${current_price:.2f}")
st.metric("RSI", f"{rsi:.2f}")
st.metric("MACD", f"{macd:.2f}")

# Prediction
if signal == "BUY":
    st.success(f"Prediction: {signal}")
else:
    st.error(f"Prediction: {signal}")

# Show latest row
st.subheader("Latest Indicators")
st.dataframe(
    df[["Close", "Volume", "RSI", "MACD"]].tail()
)