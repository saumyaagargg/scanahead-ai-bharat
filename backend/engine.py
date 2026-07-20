"""
ScanAhead Similarity Engine
"""

from collections import Counter
from pathlib import Path

import faiss
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = [
    "department",
    "diagnosis_category",
    "visit_type",
    "prior_modality",
]

NUMERIC_COLS = [
    "days_since_last_scan",
]


class ScanAheadEngine:

    def __init__(self):

        self.encoder = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_COLS,
                ),
                (
                    "num",
                    StandardScaler(),
                    NUMERIC_COLS,
                ),
            ]
        )

        self.index = None
        self.visits_df = None

    def fit(self, df):

        self.visits_df = df.reset_index(drop=True)

        X = self.encoder.fit_transform(
            self.visits_df[
                CATEGORICAL_COLS + NUMERIC_COLS
            ]
        )

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = X.astype("float32")

        dimension = X.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(X)

        return self

    def predict(self, visit, k=7):

        if self.index is None:
            raise RuntimeError("Engine has not been fitted.")

        query = pd.DataFrame([visit])

        X = self.encoder.transform(
            query[
                CATEGORICAL_COLS + NUMERIC_COLS
            ]
        )

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = X.astype("float32")

        distances, indices = self.index.search(X, k)

        distances = distances[0]
        indices = indices[0]

        neighbors = self.visits_df.iloc[indices].copy()

        neighbors["distance"] = distances

        counter = Counter(
            neighbors["scan_accessed_next"]
        )

        total_votes = sum(counter.values())

        predictions = []

        for scan, votes in counter.most_common():

            predictions.append(
                {
                    "scan": scan,
                    "votes": votes,
                    "confidence": round(votes / total_votes, 2),
                }
            )

        return {

            "predictions": predictions,

            "neighbors": neighbors[
                [
                    "visit_id",
                    "patient_id",
                    "department",
                    "diagnosis_category",
                    "scan_accessed_next",
                    "distance",
                ]
            ].to_dict(
                orient="records"
            ),
        }


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_PATH = BASE_DIR / "data" / "opd_visits.csv"

    df = pd.read_csv(DATA_PATH)

    engine = ScanAheadEngine()

    engine.fit(df)

    sample = {

        "department": "Cardiology",

        "diagnosis_category": "chest_pain",

        "visit_type": "new",

        "prior_modality": "none",

        "days_since_last_scan": 0,
    }

    result = engine.predict(sample)

    print("\nPredictions\n")

    for prediction in result["predictions"]:

        print(prediction)

    print("\nNearest Neighbours\n")

    for neighbour in result["neighbors"][:5]:

        print(neighbour)