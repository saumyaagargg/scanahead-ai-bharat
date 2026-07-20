"""
ScanAhead core engine: cross-patient, few-shot similarity matching.

Instead of training a model per patient or per hospital (infeasible with
limited data), we encode every historical visit as a feature vector and
build a similarity index over ALL visits. For a new visit, we retrieve
the k most similar historical visits (regardless of which patient they
belong to) and predict the scan(s) most commonly accessed next among
them. This sidesteps the data-scarcity problem: a brand-new patient with
zero history can still get a useful prediction, because we're matching
against similar *situations*, not this patient's own (sparse) past.
"""

from collections import Counter

import faiss
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = ["department", "diagnosis_category", "visit_type", "prior_modality"]
NUMERIC_COLS = ["days_since_last_scan"]


class ScanAheadEngine:
    def __init__(self):
        self.encoder = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
                ("num", StandardScaler(), NUMERIC_COLS),
            ]
        )
        self.index = None
        self.visits_df = None
        self.feature_matrix = None

    def fit(self, df: pd.DataFrame):
        """Build the similarity index over all historical visits."""
        self.visits_df = df.reset_index(drop=True)
        X = self.encoder.fit_transform(
            self.visits_df[CATEGORICAL_COLS + NUMERIC_COLS]
        ).astype("float32")

        # Convert sparse matrix (from OneHotEncoder) to dense for FAISS
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.feature_matrix = X
        dim = X.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(X)
        return self

    def predict(self, new_visit: dict, k: int = 7):
        """
        new_visit: dict with keys department, diagnosis_category,
        visit_type, prior_modality, days_since_last_scan.

        Returns predicted scans ranked by how often they appeared among
        the k most similar historical visits, plus the neighbor visits
        used (for explainability / the LLM reasoning layer).
        """
        if self.index is None:
            raise RuntimeError("Call .fit() before .predict()")

        row = pd.DataFrame([new_visit])
        x = self.encoder.transform(row[CATEGORICAL_COLS + NUMERIC_COLS]).astype("float32")
        if hasattr(x, "toarray"):
            x = x.toarray()

        distances, neighbor_idx = self.index.search(x, k)
        neighbor_idx = neighbor_idx[0]
        distances = distances[0]

        neighbors = self.visits_df.iloc[neighbor_idx].copy()
        neighbors["distance"] = distances

        scan_counts = Counter(neighbors["scan_accessed_next"])
        total = sum(scan_counts.values())
        ranked = [
            {"scan": scan, "confidence": round(count / total, 2), "votes": count}
            for scan, count in scan_counts.most_common()
        ]

        return {
            "predictions": ranked,
            "neighbors": neighbors[
                ["visit_id", "patient_id", "department", "diagnosis_category",
                 "scan_accessed_next", "distance"]
            ].to_dict(orient="records"),
        }


if __name__ == "__main__":
    df = pd.read_csv("../data/opd_visits.csv")
    engine = ScanAheadEngine().fit(df)

    # Example: a brand-new patient, Cardiology, chest pain, no history at all
    new_visit = {
        "department": "Cardiology",
        "diagnosis_category": "chest_pain",
        "visit_type": "new",
        "prior_modality": "none",
        "days_since_last_scan": 0,
    }

    result = engine.predict(new_visit, k=7)
    print("Predicted scans to pre-fetch:")
    for p in result["predictions"]:
        print(f"  {p['scan']:20s} confidence={p['confidence']:.2f} ({p['votes']} similar visits)")

    print("\nBased on these similar historical visits:")
    for n in result["neighbors"][:5]:
        print(f"  {n['visit_id']} | {n['department']} | {n['diagnosis_category']} -> {n['scan_accessed_next']}")