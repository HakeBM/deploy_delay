from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import joblib
import pandas as pd
from schemas import FlightInput

MODEL_PATH = "predictor_delay.pkl"
THRESHOLD = 0.4
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

REQUIRED_COLUMNS = {
    "airline",
    "origin",
    "destination",
    "day_of_week",
    "hour",
    "distance_km",
}

NUMERIC_COLUMNS = {
    "day_of_week",
    "hour",
    "distance_km",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("PredictorDelayAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Cargando modelo ML...")
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado correctamente")
        yield
    except Exception as e:
        logger.exception("Error al cargar el modelo")
        raise RuntimeError("No se pudo cargar el modelo") from e
    finally:
        logger.info("Cerrando aplicación")

app = FastAPI(
    title="Predictor Delay API",
    description="Predictor of delays on flights with ML",
    version="1.0.0",
    lifespan=lifespan,
)

def validate_dataframe(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columnas faltantes: {sorted(missing)}",
        )

    if df.empty:
        raise HTTPException(400, "El archivo CSV está vacío")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise HTTPException(400, "Existen valores nulos en el archivo")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(
                400, f"La columna '{col}' debe ser numérica"
            )

def predict_df(df: pd.DataFrame, model) -> pd.DataFrame:
    probs = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["delay_prediction"] = (probs >= THRESHOLD).astype(int)
    result["delay_probability"] = probs.round(3)
    return result

@app.get("/")
def home():
    return {"status": "API IS WORKING 🎉"}

@app.post("/predict")
def predict_delay(flight: FlightInput):
    logger.info("Predicción individual recibida")
    df = pd.DataFrame([flight.model_dump()])
    result = predict_df(df, app.state.model).iloc[0]

    return {
        "delay_prediction": int(result.delay_prediction),
        "delay_probability": float(result.delay_probability),
    }

@app.post("/batch-predict")
async def predict_batch(file: UploadFile = File(...)):
    logger.info(f"Archivo recibido: {file.filename}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser .csv")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            413, "El archivo excede el tamaño máximo de 1 MB"
        )

    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception:
        raise HTTPException(400, "No se pudo leer el archivo CSV")

    validate_dataframe(df)

    predictions = predict_df(df, app.state.model)
    logger.info(f"Predicción en lote completada ({len(df)} registros)")

    return JSONResponse(
        content=predictions.to_dict(orient="records")
    )
