import yfinance as yf
from indicators import add_indicators


df = yf.download('AAPL', period='2y', progress=False)
if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

result = add_indicators(df)
print(result.head())
print('rows', len(result))
