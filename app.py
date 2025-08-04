# app.py - Streamlit Retail Sales Forecasting with PostgreSQL Logging

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, MetaData
import datetime

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("📈 Retail Sales Forecasting with LSTM")

# Connect to PostgreSQL if secrets are available
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

# Upload sales dataset
uploaded_file = st.file_uploader("📤 Upload your full_sales_dataset.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])

    # UI filters
    store_options = df['Store_ID'].unique()
    item_options = df['Item_ID'].unique()
    store_id = st.selectbox("Select Store", store_options)
    item_id = st.selectbox("Select Item", item_options)
    forecast_days = st.slider("Days to Forecast", min_value=7, max_value=90, value=30, step=7)

    # Filter data
    df_filtered = df[(df['Store_ID'] == store_id) & (df['Item_ID'] == item_id)].copy()
    df_filtered.sort_values('Date', inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)

    features = ['Sales', 'Price', 'Promo']
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_filtered[features])
    df_scaled = pd.DataFrame(scaled_values, columns=features)

    sequence_length = 30
    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled.iloc[i:i+sequence_length].values)
        y.append(df_scaled['Sales'].iloc[i+sequence_length])
    X, y = np.array(X), np.array(y)

    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Predict next N days
    recent_seq = df_scaled[features].iloc[-sequence_length:].values
    preds = []
    for _ in range(forecast_days):
        input_seq = np.expand_dims(recent_seq, axis=0)
        pred = model.predict(input_seq)[0][0]
        preds.append(pred)
        new_row = np.array([pred, recent_seq[-1][1], recent_seq[-1][2]])
        recent_seq = np.vstack((recent_seq[1:], new_row))

    # Inverse scale predictions
    forecast_sales = scaler.inverse_transform(np.column_stack([preds, 
                                        np.repeat(recent_seq[-1][1], forecast_days),
                                        np.repeat(recent_seq[-1][2], forecast_days)]))[:, 0]

    forecast_dates = pd.date_range(start=df_filtered['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Sales': forecast_sales.astype(int)
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_filtered['Date'], df_filtered['Sales'], label="Actual Sales")
    ax.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], label="Forecasted Sales", linestyle='--')
    ax.set_title(f"Forecast for {store_id} - {item_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Show forecast table
    st.subheader("📅 Forecasted Sales")
    st.dataframe(forecast_df)

    # Download forecast
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

    # Optional: Log forecasts to PostgreSQL
    if engine is not None:
        with engine.connect() as conn:
            for i, row in forecast_df.iterrows():
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
