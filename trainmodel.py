import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ==========================
# LOAD DATASET
# ==========================
file_path = "hyderabad_uber_trip_dataset_40k.csv"
df = pd.read_csv(file_path)
df = df.dropna()

# ==========================
# FEATURE ENGINEERING
# ==========================
df["date"]    = pd.to_datetime(df["date"], format="%d-%m-%Y")
df["Hour"]    = df["time"].str.split(":").str[0].astype(int)
df["Day"]     = df["date"].dt.day
df["Month"]   = df["date"].dt.month
df["Weekday"] = df["date"].dt.weekday
df["Year"]    = df["date"].dt.year

# ==========================
# LABEL ENCODING
# ==========================
le_pickup  = LabelEncoder()
le_drop    = LabelEncoder()
le_weather = LabelEncoder()
le_demand  = LabelEncoder()

df["pickup_location"]       = le_pickup.fit_transform(df["pickup_location"])
df["dropoff_location"]      = le_drop.fit_transform(df["dropoff_location"])
df["weather_condition"]     = le_weather.fit_transform(df["weather_condition"])
df["trip_frequency_bucket"] = le_demand.fit_transform(df["trip_frequency_bucket"])

# ==========================
# FEATURE SELECTION
# ==========================
X = df[[
    "Hour", "Day", "Month", "Weekday", "Year",
    "trip_distance_km", "trip_duration_min",
    "surge_multiplier", "driver_utilization_percent",
    "pickup_location", "dropoff_location", "weather_condition"
]]

# ==========================
# SCALING
# ==========================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# FARE MODEL — RandomForestRegressor
# ==========================
y_fare = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_fare, test_size=0.2, random_state=42
)

fare_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
fare_model.fit(X_train, y_train)
r2 = r2_score(y_test, fare_model.predict(X_test))
print(f"Fare Model R² Score: {round(r2, 4)}")

# ==========================
# DEMAND MODEL — RandomForestClassifier
# ==========================
y_demand = df["trip_frequency_bucket"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y_demand, test_size=0.2, random_state=42
)

demand_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
demand_model.fit(X_train2, y_train2)
accuracy = accuracy_score(y_test2, demand_model.predict(X_test2))
print(f"Demand Model Accuracy: {round(accuracy, 4)}")

# ==========================
# SAVE PICKLE FILES
# ==========================
os.makedirs("models", exist_ok=True)

with open("models/fare_model.pkl",    "wb") as f: pickle.dump(fare_model,   f)
with open("models/demand_model.pkl",  "wb") as f: pickle.dump(demand_model, f)
with open("models/scaler.pkl",        "wb") as f: pickle.dump(scaler,       f)
with open("models/le_pickup.pkl",     "wb") as f: pickle.dump(le_pickup,    f)
with open("models/le_drop.pkl",       "wb") as f: pickle.dump(le_drop,      f)
with open("models/le_weather.pkl",    "wb") as f: pickle.dump(le_weather,   f)
with open("models/le_demand.pkl",     "wb") as f: pickle.dump(le_demand,    f)

# Save class names for UI dropdowns
classes = {
    "pickup_locations":   le_pickup.classes_.tolist(),
    "dropoff_locations":  le_drop.classes_.tolist(),
    "weather_conditions": le_weather.classes_.tolist(),
    "demand_classes":     le_demand.classes_.tolist()
}
with open("models/classes.json", "w") as f:
    json.dump(classes, f)

print("✅ All models saved to /models/")