import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("AI FinTech Stock Price Prediction")

stock = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

data = yf.download(stock, start="2015-01-01")

st.subheader("Stock Data")
st.write(data.tail())

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

st.subheader("Predicted Future Prices")
st.write(forecast)

st.subheader("Stock Price Chart")
st.line_chart(data['Close'])
