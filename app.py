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

# -------------------------
# Company search
# -------------------------

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

    # -------------------------
    # Fetch Data
    # -------------------------

    data = yf.download(ticker, start="2010-01-01")

    data.index = data.index.date  # remove time

    st.subheader("Historical Stock Data")

    st.dataframe(data)

    st.subheader("Historical Price Chart")

    st.line_chart(data["Close"])

    # -------------------------
    # Preprocessing
    # -------------------------

    df = data[["Close"]].copy()

    forecast_days = 30

    df["Prediction"] = df["Close"].shift(-forecast_days)

    X = np.array(df.drop(["Prediction"], axis=1))
    X = X[:-forecast_days]

    y = np.array(df["Prediction"])
    y = y[:-forecast_days]

    # -------------------------
    # Train Test Split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    st.subheader("Train/Test Split")

    train_size = len(X_train)
    test_size = len(X_test)

    fig, ax = plt.subplots()

    ax.pie(
        [train_size, test_size],
        labels=["Training Data", "Testing Data"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # -------------------------
    # ML Models
    # -------------------------

    lr = LinearRegression()
    rf = RandomForestRegressor()
    svr = SVR()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svr.fit(X_train, y_train)

    # -------------------------
    # Model Accuracy
    # -------------------------

    lr_acc = r2_score(y_test, lr.predict(X_test))
    rf_acc = r2_score(y_test, rf.predict(X_test))
    svr_acc = r2_score(y_test, svr.predict(X_test))

    acc_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "SVR"],
        "Accuracy": [lr_acc, rf_acc, svr_acc]
    })

    st.subheader("Model Accuracy Comparison")

    fig, ax = plt.subplots()

    ax.bar(acc_df["Model"], acc_df["Accuracy"])

    ax.set_ylabel("R² Score")

    st.pyplot(fig)

    # -------------------------
    # Moving Average
    # -------------------------

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    st.subheader("Moving Average Trend")

    fig, ax = plt.subplots()

    ax.plot(df.index, df["Close"], label="Price")

    ax.plot(df.index, df["MA50"], label="MA50")

    ax.plot(df.index, df["MA200"], label="MA200")

    ax.legend()

    st.pyplot(fig)

    # -------------------------
    # Volatility
    # -------------------------

    df["Returns"] = df["Close"].pct_change()

    st.subheader("Volatility Distribution")

    fig, ax = plt.subplots()

    ax.hist(df["Returns"].dropna(), bins=50)

    st.pyplot(fig)

    # -------------------------
    # Backtesting
    # -------------------------

    st.subheader("Backtesting (Actual vs Predicted)")

    predictions = lr.predict(X_test)

    fig, ax = plt.subplots()

    ax.plot(y_test, label="Actual")

    ax.plot(predictions, label="Predicted")

    ax.legend()

    st.pyplot(fig)

    # -------------------------
    # Future Prediction
    # -------------------------

    X_future = X[-forecast_days:]

    lr_pred = lr.predict(X_future)
    rf_pred = rf.predict(X_future)
    svr_pred = svr.predict(X_future)

    avg_prediction = (lr_pred + rf_pred + svr_pred) / 3

    future_dates = pd.date_range(
        start=pd.to_datetime(df.index[-1]),
        periods=forecast_days + 1
    )[1:]

    future_dates = future_dates.date

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": avg_prediction
    })

    st.subheader("Future Price Prediction")

    st.dataframe(pred_df)

    # -------------------------
    # Prediction Visualization
    # -------------------------

    st.subheader("Prediction Trend")

    fig, ax = plt.subplots()

    ax.plot(df.index[-200:], df["Close"].tail(200), label="Historical")

    ax.plot(future_dates, avg_prediction, linestyle="dashed", label="Prediction")

    ax.legend()

    st.pyplot(fig)

    # -------------------------
    # Portfolio Simulation
    # -------------------------

    st.subheader("Portfolio Simulation")

    investment = st.number_input("Investment Amount (₹)", min_value=1000)

    current_price = float(df["Close"].iloc[-1])

    predicted_price = float(avg_prediction[-1])

    future_value = investment * (predicted_price / current_price)

    st.write("Current Price:", current_price)

    st.write("Predicted Price (30 Days):", predicted_price)

    st.write("Estimated Portfolio Value:", round(future_value, 2))

    portfolio_values = investment * (avg_prediction / current_price)

    fig, ax = plt.subplots()

    ax.plot(future_dates, portfolio_values)

    ax.set_title("Projected Portfolio Growth")

    st.pyplot(fig)
