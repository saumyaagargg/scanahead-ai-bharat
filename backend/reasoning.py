"""
ScanAhead reasoning layer.

Takes the raw output of the similarity engine (predicted scans + the
similar historical visits that support them) and turns it into a short,
plain-language explanation a doctor or hospital staff member can quickly
read and trust. This does NOT do the prediction itself (that's engine.py)
-- it only explains a prediction that's already been made, which keeps
the core AI logic auditable and separate from the language-generation step.

Two modes:
  1. TEMPLATE MODE (default, no API key needed, works offline/free) --
     builds the explanation from a simple sentence template using the
     actual numbers from the prediction. Good enough for the hackathon
     demo and doesn't depend on any paid API.
  2. LLM MODE (optional, needs ANTHROPIC_API_KEY) -- asks Claude to
     write a more natural-sounding explanation. Use this if you get API
     access; otherwise template mode covers you completely.
"""

import os


def explain_prediction_template(new_visit: dict, prediction: dict, neighbors: list) -> str:
    """Free, no-API-key explanation using a sentence template."""
    supporting = [
        n for n in neighbors if n["scan_accessed_next"] == prediction["scan"]
    ]
    depts = {n["department"] for n in supporting}
    diagnoses = {n["diagnosis_category"] for n in supporting}

    dept_str = ", ".join(sorted(depts)) if depts else new_visit["department"]
    diag_str = ", ".join(d.replace("_", " ") for d in sorted(diagnoses)) or new_visit["diagnosis_category"].replace("_", " ")

    return (
        f"Pre-fetching {prediction['scan'].replace('_', ' ')} — "
        f"{prediction['votes']} of the {len(neighbors)} most similar past visits "
        f"({dept_str}, {diag_str}) needed this scan "
        f"({prediction['confidence']*100:.0f}% confidence)."
    )


def explain_prediction_llm(new_visit: dict, prediction: dict, neighbors: list) -> str:
    """Optional: needs ANTHROPIC_API_KEY set. Falls back to template mode if not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return explain_prediction_template(new_visit, prediction, neighbors)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    supporting = [
        n for n in neighbors if n["scan_accessed_next"] == prediction["scan"]
    ]

    system_prompt = """You are ScanAhead's explanation assistant for hospital staff.
Given a predicted scan to pre-fetch and the similar past visits that support
the prediction, write ONE short sentence (under 30 words) explaining why this
scan is being pre-fetched. Be concrete, mention the department/diagnosis
pattern and roughly how many similar cases support it. Do not use technical
ML terms like "k-nearest neighbors" or "feature vector" -- write for a
hospital administrator, not a data scientist."""

    user_prompt = f"""New patient visit:
Department: {new_visit['department']}
Diagnosis: {new_visit['diagnosis_category']}
Visit type: {new_visit['visit_type']}

Predicted scan to pre-fetch: {prediction['scan']}
Confidence: {prediction['confidence']*100:.0f}% ({prediction['votes']} of the closest similar past visits)

Supporting similar past visits:
{chr(10).join(f"- {n['department']} / {n['diagnosis_category']} -> {n['scan_accessed_next']}" for n in supporting[:5])}

Write the one-sentence explanation."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


# Default entry point used by the rest of the app. Automatically uses the
# LLM if a key is available, otherwise the free template -- so the app
# never breaks due to missing API access.
def explain_prediction(new_visit: dict, prediction: dict, neighbors: list) -> str:
    return explain_prediction_llm(new_visit, prediction, neighbors)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import pandas as pd
    from engine import ScanAheadEngine

    df = pd.read_csv("../data/opd_visits.csv")
    engine = ScanAheadEngine().fit(df)

    new_visit = {
        "department": "Cardiology",
        "diagnosis_category": "chest_pain",
        "visit_type": "new",
        "prior_modality": "none",
        "days_since_last_scan": 0,
    }

    result = engine.predict(new_visit, k=7)
    top_prediction = result["predictions"][0]

    explanation = explain_prediction(new_visit, top_prediction, result["neighbors"])
    print(f"Prediction: {top_prediction['scan']} ({top_prediction['confidence']*100:.0f}% confidence)")
    print(f"Explanation: {explanation}")