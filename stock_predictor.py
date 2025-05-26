
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="3-Year Stock Prediction Bot", layout="wide")
st.title("3-Year Stock Return Prediction Bot")

ticker_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL)", "AAPL,MSFT,GOOGL,AMZN,META")
tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y")
    info = stock.info

    if len(hist) < 800:
        return None, None

    hist['Return_3yr'] = hist['Close'].shift(-756) / hist['Close'] - 1
    hist['MA_50'] = hist['Close'].rolling(window=50).mean()
    hist['MA_200'] = hist['Close'].rolling(window=200).mean()
    hist['Volatility_90'] = hist['Close'].rolling(window=90).std()
    hist['Volume_Avg_30'] = hist['Volume'].rolling(window=30).mean()

    pe_ratio = info.get('trailingPE', np.nan)
    pb_ratio = info.get('priceToBook', np.nan)
    roe = info.get('returnOnEquity', np.nan)
    market_cap = info.get('marketCap', np.nan)

    hist['PE'] = pe_ratio
    hist['PB'] = pb_ratio
    hist['ROE'] = roe
    hist['MarketCap'] = market_cap

    hist = hist.dropna()
    return hist, info

all_data = []
metadata = {}
with st.spinner("Fetching and processing data..."):
    for ticker in tickers:
        df, info = fetch_stock_data(ticker)
        if df is not None:
            df['Ticker'] = ticker
            all_data.append(df)
            metadata[ticker] = info

if not all_data:
    st.warning("Not enough data to train the model. Try different tickers.")
    st.stop()

all_data = pd.concat(all_data)
features = ['MA_50', 'MA_200', 'Volatility_90', 'Volume_Avg_30', 'PE', 'PB', 'ROE', 'MarketCap']
target = 'Return_3yr'

X = all_data[features]
y = all_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.metric("Mean Squared Error on Test Set", f"{mse:.4f}")

importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
st.bar_chart(importance_df.set_index("Feature"))

st.subheader("Predict Future Return")
predict_ticker = st.selectbox("Select a stock for prediction", tickers)
sample_df, sample_info = fetch_stock_data(predict_ticker)

if sample_df is not None:
    latest = sample_df.iloc[-1:]
    prediction = model.predict(latest[features])
    st.success(f"Predicted 3-year return for {predict_ticker}: {prediction[0]*100:.2f}%")

    st.subheader("Recent Price Trend")
    st.line_chart(sample_df['Close'].tail(365))

    st.subheader("Company Info")
    st.write({k: sample_info[k] for k in ['longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'priceToBook', 'returnOnEquity'] if k in sample_info})

st.subheader("Portfolio Suggestion (Alpha)")
avg_returns = all_data.groupby('Ticker')['Return_3yr'].mean()
best_picks = avg_returns.sort_values(ascending=False).head(3)
st.write("Top 3 Stocks by Predicted Avg Return:")
st.table(best_picks.reset_index().rename(columns={'Return_3yr': 'Avg Predicted Return'}))

weights = st.slider("Enter equal weights for top 3 picks (%)", 0, 100, 33, 1)
total_weight = weights * 3
if total_weight != 100:
    st.warning("Weights must sum to 100% for a balanced portfolio. Current: {}%".format(total_weight))

portfolio_return = (best_picks.mean() * (total_weight / 100)).item()
st.metric("Simulated 3-Year Portfolio Return", f"{portfolio_return*100:.2f}%")

st.subheader("Download Results")
export_df = best_picks.reset_index().rename(columns={'Return_3yr': 'Avg Predicted Return'})
export_csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV Report", data=export_csv, file_name="top_stock_predictions.csv", mime="text/csv")
