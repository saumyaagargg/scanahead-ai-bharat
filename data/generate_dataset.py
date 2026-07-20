"""
Generates a synthetic OPD (Out-Patient Department) visit dataset for ScanAhead.

Each row = one patient visit. Fields are chosen to mimic realistic
OPD registration data available WITHOUT deep EHR integration:
department, diagnosis category, visit type, referring doctor, and
prior scan history. The target label ("scan_accessed_next") simulates
which scan the doctor ended up pulling from PACS during that visit.

Correlations between department/diagnosis and scan type are built in
deliberately (with noise) so that similarity-based few-shot matching
has real signal to learn from, without needing a large training set
per patient (the actual data-scarcity problem this project solves).
"""

import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

DEPARTMENTS = [
    "Cardiology", "Orthopedics", "Pulmonology",
    "Neurology", "Gastroenterology", "General Medicine",
]

DIAGNOSIS_BY_DEPT = {
    "Cardiology": ["chest_pain", "palpitations", "post_angioplasty_followup"],
    "Orthopedics": ["fracture_suspected", "joint_pain", "post_surgery_followup"],
    "Pulmonology": ["breathing_difficulty", "chronic_cough", "tb_followup"],
    "Neurology": ["severe_headache", "seizure_evaluation", "post_stroke_followup"],
    "Gastroenterology": ["abdominal_pain", "jaundice", "post_surgery_followup"],
    "General Medicine": ["fever_evaluation", "general_checkup", "diabetes_followup"],
}

# Most likely next scan per department, with some noise/alternatives
SCAN_PROBABILITY_BY_DEPT = {
    "Cardiology": {"chest_xray": 0.4, "ct_angio": 0.3, "ecg_only": 0.2, "none": 0.1},
    "Orthopedics": {"xray_limb": 0.6, "ct_bone": 0.15, "mri_joint": 0.15, "none": 0.1},
    "Pulmonology": {"chest_xray": 0.45, "ct_chest": 0.35, "none": 0.2},
    "Neurology": {"ct_head": 0.35, "mri_brain": 0.45, "none": 0.2},
    "Gastroenterology": {"ultrasound_abdomen": 0.5, "ct_abdomen": 0.3, "none": 0.2},
    "General Medicine": {"xray_chest": 0.2, "none": 0.65, "ultrasound_abdomen": 0.15},
}

VISIT_TYPES = ["new", "follow_up"]
DOCTORS = [f"Dr_{fake.last_name()}" for _ in range(12)]
PRIOR_MODALITIES = ["none", "xray", "ct", "ultrasound", "mri"]


def sample_scan(department):
    options = SCAN_PROBABILITY_BY_DEPT[department]
    return random.choices(list(options.keys()), weights=list(options.values()))[0]


def generate_dataset(n_visits=1500, n_patients=400):
    rows = []
    patient_ids = [f"P{1000+i}" for i in range(n_patients)]

    for visit_idx in range(n_visits):
        patient_id = random.choice(patient_ids)
        department = random.choice(DEPARTMENTS)
        diagnosis = random.choice(DIAGNOSIS_BY_DEPT[department])
        visit_type = random.choices(VISIT_TYPES, weights=[0.55, 0.45])[0]
        doctor = random.choice(DOCTORS)

        days_since_last_scan = (
            0 if visit_type == "new" else random.choice([0, 7, 14, 30, 60, 90, 180])
        )
        prior_modality = (
            "none" if visit_type == "new" else random.choice(PRIOR_MODALITIES)
        )

        scan_accessed_next = sample_scan(department)

        rows.append({
            "visit_id": f"V{10000+visit_idx}",
            "patient_id": patient_id,
            "department": department,
            "diagnosis_category": diagnosis,
            "visit_type": visit_type,
            "referring_doctor": doctor,
            "days_since_last_scan": days_since_last_scan,
            "prior_modality": prior_modality,
            "scan_accessed_next": scan_accessed_next,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "opd_visits.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic visits -> {out_path}")
    print(df.head(10).to_string(index=False))