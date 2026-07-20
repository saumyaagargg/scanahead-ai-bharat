import time
import requests


class OrthancClient:

    def __init__(self, url="http://localhost:8042"):
        self.url = url

    def get_all_studies(self):
        r = requests.get(f"{self.url}/studies")
        r.raise_for_status()
        return r.json()

    def get_study_details(self, study_id):
        r = requests.get(f"{self.url}/studies/{study_id}")
        r.raise_for_status()
        return r.json()

    def latest_study(self):

        start = time.perf_counter()

        studies = self.get_all_studies()

        if not studies:
            return None

        details = self.get_study_details(studies[0])

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "study_id": studies[0],
            "patient_name": details["PatientMainDicomTags"].get("PatientName", ""),
            "patient_id": details["PatientMainDicomTags"].get("PatientID", ""),
            "study_description": details["MainDicomTags"].get("StudyDescription", ""),
            "study_date": details["MainDicomTags"].get("StudyDate", ""),
            "series_count": len(details["Series"]),
            "retrieval_time_ms": elapsed_ms,
        }