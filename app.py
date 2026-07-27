import pickle
import json
import math
from flask import Flask, request, jsonify, render_template
from datetime import datetime

app = Flask(__name__)

# ==========================
# LOAD PICKLE FILES ONCE
# ==========================
with open("models/fare_model.pkl",    "rb") as f: fare_model   = pickle.load(f)
with open("models/demand_model.pkl",  "rb") as f: demand_model = pickle.load(f)
with open("models/scaler.pkl",        "rb") as f: scaler       = pickle.load(f)
with open("models/le_pickup.pkl",     "rb") as f: le_pickup    = pickle.load(f)
with open("models/le_drop.pkl",       "rb") as f: le_drop      = pickle.load(f)
with open("models/le_weather.pkl",    "rb") as f: le_weather   = pickle.load(f)
with open("models/le_demand.pkl",     "rb") as f: le_demand    = pickle.load(f)

with open("models/classes.json", "r") as f:
    classes = json.load(f)

print("✅ Models loaded successfully!")

# ==========================
# AREA COORDINATES
# ==========================
COORDS = {
    "Ameerpet":      (17.4375, 78.4483),
    "Banjara Hills": (17.4138, 78.4480),
    "Begumpet":      (17.4439, 78.4636),
    "Charminar":     (17.3616, 78.4747),
    "Gachibowli":    (17.4401, 78.3489),
    "Hitech City":   (17.4504, 78.3808),
    "Jubilee Hills": (17.4310, 78.4072),
    "Kukatpally":    (17.4849, 78.4138),
    "LB Nagar":      (17.3486, 78.5523),
    "Madhapur":      (17.4478, 78.3918),
    "Miyapur":       (17.4956, 78.3566),
    "Secunderabad":  (17.4399, 78.4983)
}

def calc_distance(pickup, dropoff):
    lat1, lon1 = COORDS[pickup]
    lat2, lon2 = COORDS[dropoff]
    dlat = abs(lat1 - lat2) * 111
    dlon = abs(lon1 - lon2) * 111 * 0.9
    straight = math.sqrt(dlat**2 + dlon**2)
    return round(max(2.0, straight * 1.4), 2)

def calc_duration(distance_km):
    return round((distance_km / 25) * 60, 1)

def calc_surge(hour, weekday):
    surge = 1.0
    if hour in [8, 9, 10]:       surge += 0.6
    elif hour in [17, 18, 19]:   surge += 0.8
    elif hour in [12, 13]:       surge += 0.2
    elif hour in [0, 1, 2]:      surge += 0.4
    if weekday >= 5:              surge += 0.3
    return round(min(surge, 2.8), 1)

def calc_utilization(hour, weekday):
    if hour in [8, 9, 10, 17, 18, 19]:   util = 85
    elif hour in [12, 13, 20, 21]:        util = 70
    elif hour in [0, 1, 2, 3]:            util = 50
    else:                                  util = 60
    if weekday >= 5:                       util += 10
    return min(util, 95)

# ==========================
# ROUTES
# ==========================
@app.route("/")
def index():
    return render_template("index.html", classes=classes)


@app.route("/route-info", methods=["POST"])
def route_info():
    try:
        data    = request.get_json()
        pickup  = data["pickup_location"]
        dropoff = data["dropoff_location"]

        if pickup == dropoff:
            return jsonify({"error": "Pickup and dropoff cannot be the same location!"}), 400

        distance = calc_distance(pickup, dropoff)
        duration = calc_duration(distance)

        now     = datetime.now()
        hour    = now.hour
        weekday = now.weekday()

        surge = calc_surge(hour, weekday)
        util  = calc_utilization(hour, weekday)

        return jsonify({
            "distance": distance,
            "duration": duration,
            "hour":     hour,
            "day":      now.day,
            "month":    now.month,
            "weekday":  weekday,
            "year":     now.year,
            "surge":    surge,
            "util":     util
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data["pickup_location"] == data["dropoff_location"]:
            return jsonify({"error": "Pickup and dropoff cannot be the same location!"}), 400

        pickup_id  = le_pickup.transform([data["pickup_location"]])[0]
        dropoff_id = le_drop.transform([data["dropoff_location"]])[0]
        weather_id = le_weather.transform([data["weather_condition"]])[0]

        features = [[
            data["hour"], data["day"], data["month"], data["weekday"], data["year"],
            data["trip_distance_km"], data["trip_duration_min"],
            data["surge_multiplier"], data["driver_utilization_percent"],
            pickup_id, dropoff_id, weather_id
        ]]

        X_scaled       = scaler.transform(features)
        fare           = round(float(fare_model.predict(X_scaled)[0]), 2)
        demand_id_pred = demand_model.predict(X_scaled)[0]
        demand         = le_demand.inverse_transform([demand_id_pred])[0]

        return jsonify({"fare": fare, "demand": demand})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)