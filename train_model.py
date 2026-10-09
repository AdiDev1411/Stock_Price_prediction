import json
import pickle
from pathlib import Path

import yfinance as yf
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from indicators import add_indicators

symbol = "AAPL"
horizon_days = 20
confidence_threshold = 0.75
df = yf.download(symbol, period="10y", progress=False)
if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

df = add_indicators(df)

close = df["Close"].squeeze()
future_close = close.shift(-horizon_days)
df["Target"] = (future_close > close).where(future_close.notna())

features = [
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
    "EMA20_vs_EMA50",
    "EMA50_vs_EMA100",
    "Return_10d",
    "RSI_Change",
]

df = df.dropna(subset=features + ["Target"])
X = df[features]
y = df["Target"].astype(int)

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test).max(axis=1)
selected = probabilities >= confidence_threshold
overall_accuracy = accuracy_score(y_test, predictions)
accuracy = accuracy_score(y_test[selected], predictions[selected])
coverage = float(selected.mean())

with Path("xgb.pkl").open("wb") as handle:
    pickle.dump(model, handle)
metadata = {
    "symbol": symbol,
    "period": "10y",
    "target": f"{horizon_days}-trading-day direction",
    "confidence_threshold": confidence_threshold,
    "accuracy": float(accuracy),
    "overall_accuracy": float(overall_accuracy),
    "coverage": coverage,
    "features": features,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
}
with open("model_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle)

print(f"Model trained successfully with selective accuracy: {accuracy:.2%} ({coverage:.2%} coverage)")