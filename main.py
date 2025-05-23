from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

#uvicorn main:app --reload
#

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}




# Load trained model
model = joblib.load("model/hotel_price_model.pkl")
model2 = joblib.load("model/hotel_base_price_model.pkl")

# Define request schema
class BookingRequest(BaseModel):
    base_price: int
    room_type: int
    is_weekend: int
    is_summer: int
    is_booked: int
    loyalty: int
    is_special_offer: int
    is_local_event: int
    is_bad_weather: int
    is_holiday: int
    traveler_type: int

# Rule-based logic
def calculate_final_price(base_price, room_type, is_weekend, is_summer, is_booked, loyalty, is_special_offer, is_local_event, is_bad_weather, is_holiday, traveler_type):
    price = base_price

    if room_type == 1:
        price += 1000
    elif room_type == 0:
        price += 500

    if is_weekend:
        price *= 1.05
    if is_summer:
        price *= 1.15
    if is_booked:
        price *= 1.18
    if loyalty:
        price *= 0.8
    if is_special_offer:
        price *= 0.93
    if is_local_event:
        price *= 1.10
    if is_bad_weather:
        price *= 0.9
    if is_holiday:
        price *= 1.08
    if traveler_type == 1:  # Business
        price *= 1.12
    elif traveler_type == 0:  # Leisure
        price *= 0.95

    return round(price)


@app.post("/predict")
def predict_price(data: BookingRequest):
    features = pd.DataFrame([data.dict()])
    ml_prediction = round(model.predict(features)[0])
    rule_based_price = calculate_final_price(**data.dict())

    return {
        "base_price": data.base_price,
        #"rule_based_price": rule_based_price,
        "ml_predicted_price": ml_prediction,
       #"match": rule_based_price == ml_prediction
    }


class BookingInput(BaseModel):
    star_rating: int
    room_type: int
    place: int
    num_adults: int
    weather: int
    is_weekend: int
    local_events: int
    occupancy_rate: float
    season: int
    month: int
    day_of_week: int
    stay_duration: int
    is_cancelled: int


@app.post("/predict2")
def predict_base_price(data: BookingInput):
    input_df = pd.DataFrame([data.dict()])

    input_df.rename(columns={
        "room_type": "Room_Type",
        "place": "Place",
        "weather": "Weather",
        "local_events": "Local_Events",
        "season": "Season"
    }, inplace=True)
    prediction = model2.predict(input_df)[0]
    return {
        "predicted_price": round(prediction)
    }

model3 = joblib.load('model/hotel_price_model3.pkl')


class BookingInputWithTrends(BaseModel):
    star_rating: int
    room_type: int
    place: int
    num_adults: int
    weather: int
    is_weekend: int
    local_events: int
    occupancy_rate: float
    season: int
    month: int
    day_of_week: int
    stay_duration: int
    is_cancelled: int
    googletrends: float

@app.post("/predict3")
def predict_base_price(data:BookingInputWithTrends):
    input_df = pd.DataFrame([data.dict()])
    print(input_df)
    input_df.rename(columns={
        "room_type": "Room_Type",
        "place": "Place",
        "weather": "Weather",
        "local_events": "Local_Events",
        "season": "Season",
    }, inplace=True)
    prediction = model3.predict(input_df)[0]
    return {
        "predicted_price": round(prediction)
    }

# @app.route('/predict3', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
 
#         # Expected input columns in the correct order
#         input_df = pd.DataFrame([{
            # 'star_rating': data['star_rating'],
            # 'Room_Type': data['Room_Type'],
            # 'Place': data['Place'],
            # 'num_adults': data['num_adults'],
            # 'Weather': data['Weather'],
            # 'is_weekend': data['is_weekend'],
            # 'Local_Events': data['Local_Events'],
            # 'occupancy_rate': data['occupancy_rate'],
            # 'Season': data['Season'],
            # 'month': data['month'],
            # 'day_of_week': data['day_of_week'],
            # 'stay_duration': data['stay_duration'],
            # 'is_cancelled': data['is_cancelled'],
            # 'googletrends': data['googletrends']
#         }])
 
#         prediction = model3.predict(input_df)[0]
 
#         return jsonify({
#             'predicted_difference': round(prediction, 2)
#         })
 
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400