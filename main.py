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
req_colum= {
  "airline","destination","origin","day_of_week","hour","distance_km"
}
app= FastAPI(tittle="Predictor Delay API", 
description= "Predictor of delays on flights whit ML",
version= "0.0.1"
)
model= joblib.load(model_in_use)
@app.get("/")
def home():
  return{"status":"API IS WORK 🎉🎊"}

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(df)[:, 1]
    df = df.copy()
    df["delay_prediction"] = (probs >= THRESHOLD).astype(int)
    df["delay_probability"] = probs.round(2)
    return df
  
@app.post("/predict")
def predict_delay(flight : FlightInput):
  df= pd.DataFrame([flight.dict()])
  result= predict_df(df).iloc[0]
  return{
"delay _prediction": int(result.delay_prediction), 
"delay_probability": float(result.delay_probability),
}
@app.post ("/batch-predict")
def predict_batch (file: UploadFile=File(...)):
  if not file.filename.endswith(".csv"):
    raise HTTPException (400,"El archivo debe ser csv")
  df=pd.read_csv(file.file)

  if not req_colum.isssubset(df.columns):
    raise HTTPException (
      400, f"El archivo debe contener las columas: {req_colum}"
    )
  return predict_df(df).to_dict(orient="records")