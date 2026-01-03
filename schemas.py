from pydantic import BaseModel, Field

class FlightInput(BaseModel):
airline: str= Field(..., example="AA")
origin: str= Field(..., example="SFO")
destination: str= Field(..., example="LAX")
distance_km: float= Field(..., example="4000")
day_of_week: int= Field(..., ge=0, le=6 example="0")
hour: int= Field(..., ge=0, le=23 example="18")
