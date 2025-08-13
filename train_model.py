# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1) Load
df = pd.read_csv("BikeRentalData.csv")
df = df.dropna()

# 2) Drop unwanted cols if they exist
for c in ["instant","dteday","casual","registered"]:
    if c in df.columns:
        df = df.drop(columns=c)

# 3) Define categorical and numerical (keep hr numeric)
categorical = ['season','weathersit','mnth','weekday','workingday','yr','holiday']
numerical = ['temp','atemp','hum','windspeed']

# 4) One-hot encode categorical (drop_first to avoid multicollinearity like before)
df = pd.get_dummies(df, columns=[c for c in categorical if c in df.columns], drop_first=True)

# 5) Scale numerical features
scaler = StandardScaler()
df[numerical] = scaler.fit_transform(df[numerical])

# 6) Prepare X,y
X = df.drop('cnt', axis=1)
y = df['cnt'].values

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8) Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 9) Evaluate
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

# 10) Save model, scaler, feature columns
pickle.dump(rf, open("bike_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("feature_columns.pkl", "wb"))
print("Saved: bike_model.pkl, scaler.pkl, feature_columns.pkl")
