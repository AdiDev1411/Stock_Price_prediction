import yfinance as yf
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

# Download stock data
df = yf.download("AAPL", period="2y")

# Convert Close column to Series
close = df['Close'].squeeze()

# Technical indicators
df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
df['MACD'] = ta.trend.MACD(close).macd()

# Remove NaN values
df.dropna(inplace=True)

# Create target
df['Target'] = (close.shift(-1) > close).astype(int)

# Features
features = ['Close', 'Volume', 'RSI', 'MACD']

X = df[features]
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("xgb.pkl", "wb"))

print("Model trained successfully!")