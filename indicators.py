import ta

def add_indicators(df):

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    df['MACD'] = ta.trend.MACD(df['Close']).macd()

    df['EMA20'] = df['Close'].ewm(span=20).mean()

    df['SMA50'] = df['Close'].rolling(50).mean()

    df.dropna(inplace=True)

    return df