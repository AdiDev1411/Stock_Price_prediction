import pickle
import yfinance as yf
import ta

# Load model
model = pickle.load(open("xgb.pkl", "rb"))

# Download latest data
df = yf.download("AAPL", period="3mo")

# Indicators
close = df['Close'].squeeze()

df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
df['MACD'] = ta.trend.MACD(close).macd()

df.dropna(inplace=True)

# Latest row
latest = df[['Close','Volume','RSI','MACD']].iloc[-1:]

# Predict
prediction = model.predict(latest)

if prediction[0] == 1:
    print("BUY")
else:
    print("SELL")