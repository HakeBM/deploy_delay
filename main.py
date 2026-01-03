from fastapi import FastAPI
import joblib
import pandas as pd
from schemas import FlightInput

app= FastAPI(tittle="Predictor Delay API",
description= "Predictor of delays on flights whit ML",
version= "0.0.1"
)
model= joblib.load('predictor_delay.pkl')
THRESHOLD =0.4
@app.get("/")
def home():
  return{"status":"API EN FUNCIONAMIENTO 🎉🎊"}
@app.post("/predict")
def predict_delay(flight : FlightInput):
  x= pd.DataFrame([flight.dict()])
  prob= model.predict_proba(x)[0][1]
  predict= int(prob>=THRESHOLD)
  return{
"delay _prediction": predict, 
"delay_probability": round(float(prob),3),
"treshold_used" : THRESHOLD
}