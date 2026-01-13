from fastapi import FastAPI,UploadFile,File,HTTPException, Depends
from fastapi.response import JSONResponse
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import io 
import logging
from contextlib import asynccontextmanager
from schemas import FlightInput,PredictionFlight, BatchResponse

model_in_use = "predictor_delay.pkl"
THRESHOLD =0.4
req_colum= [
  "airline","destination","origin","day_of_week","hour","distance_km"
]
app= FastAPI(tittle="Predictor Delay API", 
description= "Predictor of delays on flights whit ML",
version= "0.0.1"
)
model= joblib.load(model_in_use)
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
@app.post ("batch/predict")
def predict_batch (file: UploadFile=(File...)):
