import json
import pickle
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from indicators import add_indicators
from sentiment import get_global_news_sentiment

# Download stock data
df = yf.download("AAPL", period="2y")
if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

# Add technical indicators and derived features
df = add_indicators(df)

# Create target
close = df["Close"].squeeze()
df["Target"] = (close.shift(-1) > close).astype(int)

# Add global news sentiment feature
news_sentiment = get_global_news_sentiment()
df["News_Sentiment"] = news_sentiment

# Features
features = [
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
    "News_Sentiment",
]

# Keep rows with complete feature set
df = df.dropna(subset=features + ["Target"])
X = df[features]
y = df["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model and metadata
pickle.dump(model, open("xgb.pkl", "wb"))
metadata = {
    "accuracy": float(accuracy),
    "features": features,
}
with open("model_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle)

print(f"Model trained successfully with accuracy: {accuracy:.2%}")