# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Allow calls from Streamlit (local)

# Load model artifacts
model = pickle.load(open("bike_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# same categorical/numerical split used during training
categorical = ['season','weathersit','mnth','weekday','workingday','yr','holiday']
numerical = ['temp','atemp','hum','windspeed']  # scaler expects these
required_fields = ['season','mnth','hr','weekday','workingday','holiday','weathersit','temp','atemp','hum','windspeed','yr']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Validate basic input presence
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build dataframe from single example
        input_df = pd.DataFrame([data])

        # ensure numeric types
        for col in ['temp','atemp','hum','windspeed','hr','mnth','weekday','yr','season','weathersit','workingday','holiday']:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    pass

        # One-hot encode categorical (same method as training)
        input_df = pd.get_dummies(input_df, columns=[c for c in categorical if c in input_df.columns], drop_first=True)

        # Scale numerical columns (scaler expects the 4 columns)
        input_df[numerical] = scaler.transform(input_df[numerical])

        # Reindex columns to match training features (fill missing with 0)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        pred = model.predict(input_df)[0]
        return jsonify({"predicted_rentals": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
