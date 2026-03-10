import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("AI FinTech Stock Price Prediction")

stock = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

if st.button("Predict Stock Price"):

    data = yf.download(stock, start="2015-01-01")

    st.subheader("Historical Stock Data")
    st.write(data)

    data = data[['Close']]

    forecast_days = 30
    data['Prediction'] = data['Close'].shift(-forecast_days)

    X = np.array(data.drop(['Prediction'], axis=1))
    X = X[:-forecast_days]

    y = np.array(data['Prediction'])
    y = y[:-forecast_days]

    model = LinearRegression()
    model.fit(X, y)

    forecast = model.predict(X[-forecast_days:])

future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1)[1:]

prediction_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": forecast
})

# remove time (00:00:00)
prediction_df["Date"] = prediction_df["Date"].dt.date

st.subheader("Future Price Prediction")
st.write(prediction_df)

fig, ax = plt.subplots()

# historical prices
ax.plot(data.index, data['Close'], label="Historical Price")

# predicted prices
ax.plot(future_dates, forecast, label="Predicted Price", linestyle="dashed")

ax.set_title("Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

from sklearn.metrics import r2_score

pred = model.predict(X_test)
accuracy = r2_score(y_test, pred)

st.write("Model Accuracy:", accuracy)
