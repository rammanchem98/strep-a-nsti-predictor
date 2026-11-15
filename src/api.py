"""FastAPI service for the served pipeline. Trained on SYNTHETIC data; not for clinical use."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import config

STATE: dict = {}


def load_model():
    if "pipe" not in STATE:
        STATE["pipe"] = joblib.load(config.SERVED_MODEL)
        STATE["meta"] = json.loads(config.SERVED_META.read_text())
    return STATE["pipe"], STATE["meta"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="NSTI GAS predictor (SYNTHETIC)", version="0.1.0", lifespan=lifespan,
              description=config.DISCLAIMER)

Binary = Literal[0, 1]


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "AGE": 45, "DIABETES": 0, "UPPER_ARM_ARRIVAL": 1, "LOWER_ARM_ARRIVAL": 0,
        "ANOGENITAL_ARRIVAL": 0, "SURGERY_BEFORE": 0, "creatinine_preop": 180.0, "hgb_preop": 12.1}})

    AGE: int = Field(..., ge=18, le=100, description="Age in years")
    DIABETES: Binary
    UPPER_ARM_ARRIVAL: Binary
    LOWER_ARM_ARRIVAL: Binary
    ANOGENITAL_ARRIVAL: Binary
    SURGERY_BEFORE: Binary = Field(..., description="Recent surgery before presentation")
    creatinine_preop: Optional[float] = Field(None, ge=20, le=2000, description="umol/L; null -> imputed")
    hgb_preop: Optional[float] = Field(None, ge=3, le=22, description="g/dL; null -> imputed")

    @model_validator(mode="after")
    def one_site(self):
        if self.UPPER_ARM_ARRIVAL + self.LOWER_ARM_ARRIVAL + self.ANOGENITAL_ARRIVAL > 1:
            raise ValueError("arrival sites are mutually exclusive (at most one may be 1)")
        return self


class Prediction(BaseModel):
    probability_gas: float
    gas_flag: bool
    threshold: float
    model: str
    data: str
    disclaimer: str


@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "model": meta["served_model"], "data": "SYNTHETIC",
            "disclaimer": config.DISCLAIMER}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    pipe, meta = load_model()
    X = pd.DataFrame([patient.model_dump()])[config.FEATURES].astype(float)
    p = float(pipe.predict_proba(X)[0, 1])
    return Prediction(probability_gas=round(p, 4), gas_flag=p >= config.THRESHOLD,
                      threshold=config.THRESHOLD, model=meta["served_model"], data="SYNTHETIC",
                      disclaimer=config.DISCLAIMER)
