import yfinance as yf
from indicators import add_indicators


df = yf.download('AAPL', period='2y', progress=False)
if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

result = add_indicators(df)
print('rows', len(result))
print('rsi min', float(result['RSI'].min()))
print('rsi max', float(result['RSI'].max()))
print('rsi tail')
print(result['RSI'].tail(10).to_string())
