# ScanAhead AI Bharat

AI-powered medical scan prefetching system that predicts the most likely diagnostic scan for a patient using similarity search and integrates with a real PACS server (Orthanc).

---

## Live Demo

Frontend (Vercel)

https://scanahead-ai-bharat.vercel.app

Backend (Render)

https://scanahead-ai-bharat.onrender.com

Swagger (Local)

http://127.0.0.1:8010/docs

---

## Problem Statement

Hospitals often waste valuable time waiting for medical scans to be retrieved after a doctor requests them.

ScanAhead AI Bharat predicts the next likely scan before it is requested and connects to a PACS server to retrieve study information, helping reduce delays in clinical workflows.

---

## Features

- AI-based scan prediction
- Similarity Search using FAISS
- Orthanc PACS Integration
- Explainable AI predictions
- FastAPI REST Backend
- React Frontend
- Swagger API Documentation
- Medical Imaging Workflow
- Real-time PACS study metadata retrieval
- REST API integration with Orthanc
- Cross-patient similarity matching using historical hospital visits

---

## Tech Stack

### Backend

- Python
- FastAPI
- FAISS
- Pandas
- Scikit-learn

### Frontend

- React
- Vite

### Medical Imaging

- Orthanc PACS
- DICOM

---

## System Architecture

```
Patient Visit
      │
      ▼
React Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
FAISS Similarity Engine
      │
      ▼
AI Prediction
      │
      ▼
Reasoning Layer
      │
      ▼
Orthanc PACS
      │
      ▼
Study Metadata
```

---

## API Endpoints

### GET /

Health Check

### GET /departments

Returns all departments and diagnosis categories.

### POST /predict

Predicts the most likely scan based on similar historical patient visits and retrieves the latest PACS study metadata.

Example Request

```json
{
  "department": "Cardiology",
  "diagnosis_category": "chest_pain",
  "visit_type": "new",
  "prior_modality": "none",
  "days_since_last_scan": 0
}
```

---

## Run Backend

```bash
python -m uvicorn backend.api:app --reload --port 8010
```

Swagger UI

```
http://127.0.0.1:8010/docs
```

---

## Project Structure

```
backend/
frontend/
pacs/
data/
README.md
requirements.txt
```

---

## Future Improvements

- DICOM Image Preview
- Real PACS Prefetching
- Docker Deployment
- C-ECHO
- C-FIND
- C-MOVE
- C-STORE
- Multi-study visualization
- AI-assisted scan prioritization

---

## Author

**SAUMYAA GARG**

---

## Acknowledgements

- FastAPI
- FAISS
- Orthanc PACS
- React
- Scikit-learn