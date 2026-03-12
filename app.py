import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

st.title("AI Stock Prediction & Investment Analytics Dashboard")

# ---------------------------
# Company Search
# ---------------------------

company_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS"
}

company = st.selectbox("Search Company", list(company_dict.keys()))
ticker = company_dict[company]

if st.button("Run Analysis"):

    # ---------------------------
    # Fetch Historical Data
    # ---------------------------

    data = yf.download(ticker, start="2015-01-01")

    data.index = pd.to_datetime(data.index).date

    st.subheader("Historical Stock Data")

    st.dataframe(data.tail(200))

    st.subheader("Historical Price Chart")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(data.index, data["Close"])

    ax.set_title("Historical Closing Price")

    ax.set_xlabel("Date")

    ax.set_ylabel("Price")

    st.pyplot(fig)

    # ---------------------------
    # Preprocessing
    # ---------------------------

    df = data[["Close"]].copy()

    forecast_days = 30

    df["Prediction"] = df["Close"].shift(-forecast_days)

    df.dropna(inplace=True)

    X = np.array(df[["Close"]])
    y = np.array(df["Prediction"])

    # ---------------------------
    # Train Test Split
    # ---------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    st.subheader("Train vs Test Split")

    fig, ax = plt.subplots()

    sizes = [len(X_train), len(X_test)]

    labels = ["Training Data", "Testing Data"]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%')

    ax.set_title("Dataset Split")

    st.pyplot(fig)

    # ---------------------------
    # ML Models
    # ---------------------------

    lr = LinearRegression()
    rf = RandomForestRegressor()
    svr = SVR()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svr.fit(X_train, y_train)

    # ---------------------------
    # Accuracy Metrics
    # ---------------------------

    lr_pred_test = lr.predict(X_test)
    rf_pred_test = rf.predict(X_test)
    svr_pred_test = svr.predict(X_test)

    acc_df = pd.DataFrame({
        "Model":["Linear Regression","Random Forest","SVR"],
        "R2":[
            r2_score(y_test, lr_pred_test),
            r2_score(y_test, rf_pred_test),
            r2_score(y_test, svr_pred_test)
        ],
        "MAE":[
            mean_absolute_error(y_test, lr_pred_test),
            mean_absolute_error(y_test, rf_pred_test),
            mean_absolute_error(y_test, svr_pred_test)
        ]
    })

    st.subheader("Model Performance Comparison")

    fig, ax = plt.subplots()

    ax.bar(acc_df["Model"], acc_df["R2"])

    ax.set_ylabel("R² Score")

    ax.set_title("Model Accuracy Comparison")

    st.pyplot(fig)

    st.dataframe(acc_df)

    # ---------------------------
    # Moving Averages
    # ---------------------------

    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    st.subheader("Moving Average Indicator")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(data.index, data["Close"], label="Price")

    ax.plot(data.index, data["MA50"], label="MA50")

    ax.plot(data.index, data["MA200"], label="MA200")

    ax.legend()

    ax.set_title("Moving Average Trend")

    st.pyplot(fig)

    # ---------------------------
    # Volatility
    # ---------------------------

    data["Returns"] = data["Close"].pct_change()

    st.subheader("Volatility Distribution")

    fig, ax = plt.subplots()

    ax.hist(data["Returns"].dropna(), bins=50)

    ax.set_title("Return Volatility")

    st.pyplot(fig)

    # ---------------------------
    # Backtesting
    # ---------------------------

    st.subheader("Backtesting: Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(y_test, label="Actual Price")

    ax.plot(lr_pred_test, label="Predicted Price")

    ax.legend()

    ax.set_title("Backtesting Result")

    st.pyplot(fig)

    # ---------------------------
    # Future Prediction
    # ---------------------------

    X_future = np.array(data[["Close"]].tail(forecast_days))

    lr_future = lr.predict(X_future)
    rf_future = rf.predict(X_future)
    svr_future = svr.predict(X_future)

    avg_prediction = (lr_future + rf_future + svr_future) / 3

    future_dates = pd.date_range(
        start=pd.to_datetime(data.index[-1]),
        periods=forecast_days+1
    )[1:]

    future_dates = future_dates.date

    pred_df = pd.DataFrame({
        "Date":future_dates,
        "Predicted Price":avg_prediction
    })

    st.subheader("Future Price Prediction")

    st.dataframe(pred_df)

    # ---------------------------
    # Prediction Visualization
    # ---------------------------

    st.subheader("Historical vs Predicted Trend")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(data.index[-200:], data["Close"].tail(200), label="Historical")

    ax.plot(future_dates, avg_prediction, linestyle="dashed", label="Prediction")

    ax.legend()

    ax.set_xlabel("Date")

    ax.set_ylabel("Price")

    st.pyplot(fig)

    # ---------------------------
    # Portfolio Simulation
    # ---------------------------

    st.subheader("Portfolio Simulation")

    investment = st.number_input("Investment Amount (₹)", min_value=1000)

    current_price = float(data["Close"].iloc[-1])

    predicted_price = float(avg_prediction[-1])

    future_value = investment * (predicted_price/current_price)

    st.write("Current Price:", round(current_price,2))

    st.write("Predicted Price:", round(predicted_price,2))

    st.write("Estimated Portfolio Value:", round(future_value,2))

    portfolio_values = investment * (avg_prediction/current_price)

    fig, ax = plt.subplots()

    ax.plot(future_dates, portfolio_values)

    ax.set_title("Projected Portfolio Growth")

    ax.set_xlabel("Date")

    ax.set_ylabel("Portfolio Value")

    st.pyplot(fig)
