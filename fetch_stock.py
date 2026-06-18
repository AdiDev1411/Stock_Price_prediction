import yfinance as yf

def get_stock_data(symbol):
    df = yf.download(symbol, period='2y')
    df.reset_index(inplace=True)
    return df

if __name__=="__main__":
    df = get_stock_data("AAPL")
    print(df.head())