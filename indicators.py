import pandas as pd


def add_indicators(df):
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    df = df.copy()
    df["RSI"] = rsi.fillna(50)
    df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA100"] = close.ewm(span=100, adjust=False).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA100"] = close.rolling(100).mean()
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Volume_Change"] = volume.pct_change(1)
    df["Volatility_20d"] = close.pct_change().rolling(20).std()
    df["Price_vs_EMA20"] = close / df["EMA20"] - 1
    df["EMA20_vs_EMA50"] = df["EMA20"] / df["EMA50"] - 1
    df["EMA50_vs_EMA100"] = df["EMA50"] / df["EMA100"] - 1
    df["Return_10d"] = close.pct_change(10)
    df["RSI_Change"] = df["RSI"].diff(5)

    df = df.dropna(
        subset=[
            "Close",
            "Volume",
            "RSI",
            "MACD",
            "MACD_Signal",
            "EMA20",
            "EMA50",
            "EMA100",
        ]
    )
    return df