# app.py - Streamlit Retail Forecasting using Facebook Prophet

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, MetaData
import datetime

st.set_page_config(page_title="Retail Forecasting (Prophet)", layout="wide")
st.title("📈 Retail Sales Forecasting with Prophet")

# Connect to PostgreSQL if available
engine = None
if "DATABASE_URL" in st.secrets:
    DATABASE_URL = st.secrets["DATABASE_URL"]
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    forecast_table = Table("sales_forecasts", metadata,
        Column("timestamp", String),
        Column("store_id", String),
        Column("item_id", String),
        Column("forecast_days", Integer),
        Column("date", String),
        Column("predicted_sales", Integer)
    )
    metadata.create_all(engine)

# Upload dataset
uploaded_file = st.file_uploader("📤 Upload your full_sales_dataset.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])

    # UI filters
    store_options = df['Store_ID'].unique()
    item_options = df['Item_ID'].unique()
    store_id = st.selectbox("Select Store", store_options)
    item_id = st.selectbox("Select Item", item_options)
    forecast_days = st.slider("Days to Forecast", min_value=7, max_value=90, value=30, step=7)

    # Filter and prepare data
    df_filtered = df[(df['Store_ID'] == store_id) & (df['Item_ID'] == item_id)].copy()
    df_filtered = df_filtered[['Date', 'Sales']].rename(columns={"Date": "ds", "Sales": "y"})
    df_filtered.sort_values("ds", inplace=True)

    # Train Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(df_filtered)

    # Forecast
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Extract forecast
    forecast_df = forecast[['ds', 'yhat']].tail(forecast_days)
    forecast_df.columns = ['Date', 'Predicted_Sales']
    forecast_df['Predicted_Sales'] = forecast_df['Predicted_Sales'].round().astype(int)

    # Plot
    fig = model.plot(forecast)
    st.pyplot(fig)

    # Table
    st.subheader("📅 Forecasted Sales")
    st.dataframe(forecast_df)

    # Download
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

    # Optional: Log to PostgreSQL
    if engine is not None:
        with engine.connect() as conn:
            for _, row in forecast_df.iterrows():
                conn.execute(forecast_table.insert().values(
                    timestamp=str(datetime.datetime.now()),
                    store_id=store_id,
                    item_id=item_id,
                    forecast_days=forecast_days,
                    date=str(row['Date'].date()),
                    predicted_sales=int(row['Predicted_Sales'])
                ))
        st.success("✅ Forecast logged to PostgreSQL database.")
else:
    st.info("Upload a sales dataset to begin.")
