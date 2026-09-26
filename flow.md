# Code Execution Flow

```mermaid
flowchart TD
    A[Start app.py] --> B[Load model xgb.pkl]
    B --> C[Load metadata model_metadata.json]
    C --> D[Read user inputs from sidebar]
    D --> E[Download stock data with yfinance]
    E --> F[Build technical indicators in indicators.py]
    F --> G[Create prediction frame]
    G --> H[Run model.predict]
    H --> I[Show BUY or SELL signal]
    F --> J[Build analysis summary]
    J --> K[Render metrics and charts]
    K --> L[Price chart with Close, MA50, MA100, Volume]
    K --> M[RSI chart]
    M --> N[Show 0-100 RSI scale]
    N --> O[Highlight 30 and 70 bands]
    O --> P[Annotate latest RSI value]
    K --> Q[MACD chart]
    K --> R[Latest indicators table]

    S[Run train_model.py] --> T[Download historical data]
    T --> U[Add indicators and derived features]
    U --> V[Create target column]
    V --> W[Split into train and test sets]
    W --> X[Train classifier]
    X --> Y[Evaluate accuracy]
    Y --> Z[Save xgb.pkl]
    Y --> AA[Save model_metadata.json]
```

## Runtime Flow

1. The Streamlit app starts in `app.py`.
2. The model is loaded from `xgb.pkl`.
3. Stock data is downloaded from Yahoo Finance.
4. `indicators.py` adds RSI, MACD, moving averages, returns, volatility, and price-relative features.
5. The latest feature row is passed into the model for a BUY/SELL prediction.
6. The app renders the signal, summary text, price chart, RSI chart, MACD chart, and indicator table.
7. If model accuracy metadata is missing, the app computes a live fallback estimate.

## Training Flow

1. `train_model.py` downloads historical stock data.
2. It calls `add_indicators()` from `indicators.py`.
3. It creates the next-day target from the close price movement.
4. It builds the full feature matrix.
5. It trains the classifier and computes test accuracy.
6. It saves the trained model to `xgb.pkl`.
7. It writes accuracy and feature metadata to `model_metadata.json`.
