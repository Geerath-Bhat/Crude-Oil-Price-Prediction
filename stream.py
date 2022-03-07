#importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import time
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

#setting the page fevicon details
st.set_page_config(page_title ="P80 Team 2",
                    page_icon="@*@")

#Title and other text details
st.title('WTI Crude Oil Price Forecasting')
st.write('This project enables you to generate time series prediction of WTI Crude Oil Price. It is developed by: ')
st.write('Mentors: Vinod and Deepika')
st.write('Team P80-2: Geerath,Joshua,Raghav,Akanksha and Ankita')

#Drop down menu - Select no.of days for prediction
df = pd.DataFrame({
    'first column': [1, 2, 3, 4,5,6,7,8]
    })
nobs = st.selectbox(
    'How many days of Oil Price would you like to predict?',
     df['first column'])
'You selected: ', nobs

#File uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)

#Model pre-processing
df = df.set_index(['Date'])
df = df.astype(float)
df_differenced = df.diff().dropna()

#VAR Model
model = VAR(df_differenced)
x = model.select_order(maxlags=12)

#Train the VAR Model of Selected Order(2) as per model analysis already performed
model_fitted = model.fit(2)

#Forecast VAR model using statsmodels / get the lag order
lag_order = model_fitted.k_ar

#Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]

#Compute the Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')

# Invert the transformation to get the real forecast
def invert_transformation(df, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df[col].iloc[-1]-df[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc
df_results = invert_transformation(df, df_forecast, second_diff=True)        

#Create DataFrame for prediction
forecast_price = df_results['Price_forecast']
forecast_price = pd.DataFrame(forecast_price)
forecast_price = forecast_price.reset_index()
forecast_price = forecast_price.drop(columns=['Date'])
st.write(forecast_price)
df_train_last_plot = df.tail(64)
till_today = df_train_last_plot[['Price']]

#Plot the forecast as a graph
st.line_chart(data=forecast_price, width=0, height=0, use_container_width=True)
fig,ax = plt.subplots(figsize=(20,8))
plt.plot(np.arange(64), till_today.values)
plt.plot(np.arange(64, 64+nobs), forecast_price.values)
plt.show()
st.pyplot(fig)