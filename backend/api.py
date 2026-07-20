"""
ScanAhead Backend API
"""

import random
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.engine import ScanAheadEngine
from backend.reasoning import explain_prediction
from pacs.orthanc_client import OrthancClient

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="ScanAhead AI Bharat",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "opd_visits.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Initialize AI Engine
# --------------------------------------------------

engine = ScanAheadEngine()
engine.fit(df)

# --------------------------------------------------
# Initialize Orthanc
# --------------------------------------------------

orthanc = OrthancClient()

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class VisitRequest(BaseModel):
    department: str
    diagnosis_category: str
    visit_type: str = "new"
    prior_modality: str = "none"
    days_since_last_scan: int = 0


# --------------------------------------------------
# Demo Timing Functions
# --------------------------------------------------

def simulate_cold_retrieval_ms():
    return random.randint(3000, 8000)


def simulate_prefetched_retrieval_ms():
    return random.randint(80, 250)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ScanAhead AI Bharat Backend Running",
        "records_loaded": len(df),
        "orthanc_connected": orthanc.latest_study() is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/departments")
def departments():

    return {
        "departments": sorted(df["department"].unique()),

        "diagnosis_by_department": {

            dept: sorted(
                df[df["department"] == dept]["diagnosis_category"].unique()
            )

            for dept in sorted(df["department"].unique())

        }

    }


@app.post("/predict")
def predict(visit: VisitRequest):

    visit_dict = visit.model_dump()

    result = engine.predict(
        visit_dict,
        k=7
    )

    top_prediction = result["predictions"][0]

    explanation = explain_prediction(
        visit_dict,
        top_prediction,
        result["neighbors"]
    )

    # ---------------------------------------
    # Latest PACS Study
    # ---------------------------------------

    try:
        latest_study = orthanc.latest_study()
    except Exception as e:
        latest_study = {
            "error": str(e)
        }

    # ---------------------------------------
    # Response
    # ---------------------------------------

    return {

        "input_visit": visit_dict,

        "predictions": result["predictions"],

        "top_prediction": top_prediction,

        "top_explanation": explanation,

        "supporting_visits": result["neighbors"][:5],

        "retrieval_comparison": {

            "cold_storage_ms": simulate_cold_retrieval_ms(),

            "prefetched_ms": simulate_prefetched_retrieval_ms()

        },

        "pacs": latest_study

    }