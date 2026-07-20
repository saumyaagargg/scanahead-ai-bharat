"""
ScanAhead backend API.

Wraps the similarity engine (engine.py) and the explanation layer
(reasoning.py) into a single REST endpoint the frontend dashboard can
call. Also simulates the "cold storage" vs "pre-fetched" retrieval time
difference, so the demo can visually show the speed-up.

Run locally with:
    uvicorn api:app --reload --port 8000

Then POST to http://localhost:8000/predict
"""

import random
import time

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import ScanAheadEngine
from reasoning import explain_prediction

app = FastAPI(title="ScanAhead API")

# Allow the React dev server (and any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and fit the engine ONCE at startup (not per-request)
df = pd.read_csv("../data/opd_visits.csv")
engine = ScanAheadEngine().fit(df)


class VisitRequest(BaseModel):
    department: str
    diagnosis_category: str
    visit_type: str = "new"
    prior_modality: str = "none"
    days_since_last_scan: int = 0


def simulate_cold_retrieval_ms() -> int:
    """Simulates slow retrieval from archived/cold storage (3-8 seconds)."""
    return random.randint(3000, 8000)


def simulate_prefetched_retrieval_ms() -> int:
    """Simulates near-instant retrieval since the scan was already pre-fetched."""
    return random.randint(80, 250)


@app.get("/")
def health_check():
    return {"status": "ScanAhead API is running", "visits_loaded": len(df)}


@app.post("/predict")
def predict(visit: VisitRequest):
    new_visit = visit.model_dump()

    result = engine.predict(new_visit, k=7)
    top_prediction = result["predictions"][0]

    explanation = explain_prediction(new_visit, top_prediction, result["neighbors"])

    return {
        "input_visit": new_visit,
        "predictions": result["predictions"],
        "top_explanation": explanation,
        "supporting_visits": result["neighbors"][:5],
        "retrieval_comparison": {
            "cold_storage_ms": simulate_cold_retrieval_ms(),
            "prefetched_ms": simulate_prefetched_retrieval_ms(),
        },
    }


@app.get("/departments")
def get_departments():
    """Helper endpoint so the frontend can populate dropdowns from real data."""
    return {
        "departments": sorted(df["department"].unique().tolist()),
        "diagnosis_by_department": {
            dept: sorted(df[df["department"] == dept]["diagnosis_category"].unique().tolist())
            for dept in df["department"].unique()
        },
    }