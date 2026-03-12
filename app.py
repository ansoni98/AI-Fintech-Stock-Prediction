import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

st.title("AI Stock Prediction & Investment Analytics Dashboard")

# Company search (name instead of ticker)
company_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
}

company = st.selectbox("Search Company", list(company_dict.keys()))
ticker = company_dict[company]

if st.button("Run Analysis"):

    # ---------------------------
    # 1 Fetch Historical Data
    # ---------------------------

    data = yf.download(ticker, start="2015-01-01")

    st.subheader("Historical Stock Data")
    st.dataframe(data.tail())

    # ---------------------------
    # 2 Data Preprocessing
    # ---------------------------

    data = data[["Close"]]
    data.dropna(inplace=True)

    forecast_days = 30
    data["Prediction"] = data["Close"].shift(-forecast_days)

    X = np.array(data.drop(["Prediction"], axis=1))
    X = X[:-forecast_days]

    y = np.array(data["Prediction"])
    y = y[:-forecast_days]

    # ---------------------------
    # 3 Train Test Split
    # ---------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # 4 Machine Learning Models
    # ---------------------------

    lr = LinearRegression()
    rf = RandomForestRegressor()
    svr = SVR()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svr.fit(X_train, y_train)

    # ---------------------------
    # 5 Model Accuracy
    # ---------------------------

    lr_acc = r2_score(y_test, lr.predict(X_test))
    rf_acc = r2_score(y_test, rf.predict(X_test))
    svr_acc = r2_score(y_test, svr.predict(X_test))

    acc_df = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest", "SVR"],
            "Accuracy": [lr_acc, rf_acc, svr_acc],
        }
    )

    st.subheader("Model Accuracy Comparison")

    fig, ax = plt.subplots()
    ax.bar(acc_df["Model"], acc_df["Accuracy"])
    ax.set_ylabel("R2 Score")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

    # ---------------------------
    # 6 Moving Averages
    # ---------------------------

    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    st.subheader("Moving Average Trend")

    fig, ax = plt.subplots()
    ax.plot(data.index, data["Close"], label="Price")
    ax.plot(data.index, data["MA50"], label="MA50")
    ax.plot(data.index, data["MA200"], label="MA200")
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # 7 Volatility Analysis
    # ---------------------------

    data["Returns"] = data["Close"].pct_change()

    volatility = data["Returns"].std()

    st.subheader("Volatility Analysis")

    st.write("Stock Volatility:", volatility)

    fig, ax = plt.subplots()
    ax.hist(data["Returns"].dropna(), bins=50)
    ax.set_title("Return Distribution")
    st.pyplot(fig)

    # ---------------------------
    # 8 Prediction Models
    # ---------------------------

    X_future = X[-forecast_days:]

    lr_pred = lr.predict(X_future)
    rf_pred = rf.predict(X_future)
    svr_pred = svr.predict(X_future)

    avg_prediction = (lr_pred + rf_pred + svr_pred) / 3

    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1)[1:]

    pred_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Price": avg_prediction,
        }
    )

    st.subheader("Future Price Prediction")
    st.dataframe(pred_df)

    # ---------------------------
    # 9 Prediction Visualization
    # ---------------------------

    fig, ax = plt.subplots()

    ax.plot(data.index[-200:], data["Close"].tail(200), label="Historical Price")

    ax.plot(future_dates, avg_prediction, linestyle="dashed", label="Predicted Price")

    ax.legend()

    ax.set_title("Stock Price Prediction")

    st.pyplot(fig)

    # ---------------------------
    # 10 Portfolio Simulation
    # ---------------------------

    st.subheader("Portfolio Simulation")

    investment = st.number_input("Investment Amount (₹)", 1000)

    current_price = data["Close"].iloc[-1]

    predicted_price = avg_prediction[-1]

    future_value = investment * (predicted_price / current_price)

    st.write("Current Price:", current_price)

    st.write("Predicted Price (30 days):", predicted_price)

    st.write("Estimated Portfolio Value:", future_value)

    # Portfolio chart

    portfolio_values = investment * (avg_prediction / current_price)

    fig, ax = plt.subplots()

    ax.plot(future_dates, portfolio_values)

    ax.set_title("Projected Portfolio Growth")

    st.pyplot(fig)
