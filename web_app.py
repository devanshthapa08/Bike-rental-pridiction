# web_app.py
import streamlit as st
import requests

st.title("🚲 Bike Rental Prediction")

st.markdown("Enter the feature values and click **Predict**")

season = st.selectbox("Season", [1,2,3,4])
mnth = st.slider("Month", 1, 12, 1)
hr = st.slider("Hour (0-23)", 0, 23, 12)
weekday = st.slider("Weekday (0=Sun)", 0, 6, 1)
workingday = st.selectbox("Working day", [0,1])
holiday = st.selectbox("Holiday", [0,1])
weathersit = st.selectbox("Weather situation", [1,2,3,4])

temp = st.number_input("Temperature (normalized 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
atemp = st.number_input("Feels-like temp (normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
hum = st.number_input("Humidity (normalized 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
windspeed = st.number_input("Windspeed (normalized 0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
yr = st.selectbox("Year (0=2011, 1=2012)", [0,1])

if st.button("Predict Rentals"):
    payload = {
        "season": int(season),
        "mnth": int(mnth),
        "hr": int(hr),
        "weekday": int(weekday),
        "workingday": int(workingday),
        "holiday": int(holiday),
        "weathersit": int(weathersit),
        "temp": float(temp),
        "atemp": float(atemp),
        "hum": float(hum),
        "windspeed": float(windspeed),
        "yr": int(yr)
    }
    try:
        res = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
        if res.status_code == 200:
            st.success(f"Predicted rentals: {res.json()['predicted_rentals']:.0f}")
        else:
            st.error(f"API Error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
