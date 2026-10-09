import pickle
import yfinance as yf

from indicators import add_indicators

# Load model
model = pickle.load(open("xgb.pkl", "rb"))

# Download enough history for the model's 100-day indicators.
df = yf.download("AAPL", period="1y", progress=False)

if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

df = add_indicators(df)

feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    raise RuntimeError("The model does not contain feature names.")

latest = df.dropna(subset=feature_names)[feature_names].iloc[-1:]

# Predict
prediction = model.predict(latest)

if prediction[0] == 1:
    print("BUY")
else:
    print("SELL")