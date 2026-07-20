import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiActivity,
  FiClock,
  FiZap,
  FiDatabase,
  FiCheckCircle,
} from "react-icons/fi";
import "./App.css";

const API_BASE = "http://localhost:8000"; // update to your deployed backend URL

function App() {
  const [departments, setDepartments] = useState([]);
  const [diagnosisByDept, setDiagnosisByDept] = useState({});
  const [department, setDepartment] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [visitType, setVisitType] = useState("new");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [coldMs, setColdMs] = useState(0);
  const [fastMs, setFastMs] = useState(0);
  const [raceDone, setRaceDone] = useState(false);
  const raceTimeouts = useRef([]);

  useEffect(() => {
    async function fetchDepartments() {
      try {
        const res = await axios.get(`${API_BASE}/departments`);
        setDepartments(res.data.departments);
        setDiagnosisByDept(res.data.diagnosis_by_department);
        if (res.data.departments.length > 0) {
          const firstDept = res.data.departments[0];
          setDepartment(firstDept);
          setDiagnosis(res.data.diagnosis_by_department[firstDept]?.[0] || "");
        }
      } catch (err) {
        setError("Could not reach the ScanAhead backend. Make sure the API server is running.");
      }
    }
    fetchDepartments();
  }, []);

  useEffect(() => {
    if (department && diagnosisByDept[department]) {
      setDiagnosis(diagnosisByDept[department][0] || "");
    }
  }, [department, diagnosisByDept]);

  function runRetrievalRace(coldTargetMs, fastTargetMs) {
    raceTimeouts.current.forEach(clearTimeout);
    raceTimeouts.current = [];
    setRaceDone(false);
    setColdMs(0);
    setFastMs(0);

    const duration = 1400;
    const steps = 40;
    const interval = duration / steps;

    for (let i = 1; i <= steps; i++) {
      const t = setTimeout(() => {
        const progress = i / steps;
        setColdMs(Math.round(Math.min(progress, 1) * coldTargetMs));
        const fastProgress = Math.min(
          (i * interval) / Math.max((fastTargetMs / coldTargetMs) * duration, interval),
          1
        );
        setFastMs(Math.round(fastProgress * fastTargetMs));
        if (i === steps) setRaceDone(true);
      }, i * interval);
      raceTimeouts.current.push(t);
    }
  }

  async function handlePredict() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        department,
        diagnosis_category: diagnosis,
        visit_type: visitType,
        prior_modality: "none",
        days_since_last_scan: 0,
      });

      setResult(res.data);
      const { cold_storage_ms, prefetched_ms } = res.data.retrieval_comparison;
      runRetrievalRace(cold_storage_ms, prefetched_ms);
    } catch (err) {
      setError("Prediction failed. Check that the backend is running and reachable.");
    } finally {
      setLoading(false);
    }
  }

  const topPrediction = result?.predictions?.[0];
  const speedup =
    result && topPrediction
      ? Math.max(
          1,
          Math.round(
            result.retrieval_comparison.cold_storage_ms /
              result.retrieval_comparison.prefetched_ms
          )
        )
      : null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="wordmark">
          <FiActivity className="wordmark-icon" />
          <div>
            <h1>ScanAhead AI</h1>
            <p className="tagline">AI-Powered Intelligent Medical Scan Pre-Fetching</p>
            <p className="subtitle">Reducing patient wait times through predictive scan retrieval</p>
          </div>
        </div>
      </header>

      <main className="layout">
        <motion.section
          className="card form-card"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h2>Patient Information</h2>

          <label htmlFor="department">Department</label>
          <select
            id="department"
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
          >
            {departments.map((dept) => (
              <option key={dept} value={dept}>
                {dept}
              </option>
            ))}
          </select>

          <label htmlFor="diagnosis">Diagnosis</label>
          <select
            id="diagnosis"
            value={diagnosis}
            onChange={(e) => setDiagnosis(e.target.value)}
          >
            {(diagnosisByDept[department] || []).map((diag) => (
              <option key={diag} value={diag}>
                {diag.replace(/_/g, " ")}
              </option>
            ))}
          </select>

          <label htmlFor="visitType">Visit type</label>
          <select
            id="visitType"
            value={visitType}
            onChange={(e) => setVisitType(e.target.value)}
          >
            <option value="new">New</option>
            <option value="follow_up">Follow-up</option>
          </select>

          <motion.button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading || !department}
            whileTap={{ scale: 0.97 }}
          >
            {loading ? "Predicting..." : "Predict scan"}
          </motion.button>

          {error && <p className="error-text">{error}</p>}
        </motion.section>

        <motion.section
          className="card result-card"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <AnimatePresence mode="wait">
            {!result && !loading && (
              <motion.div
                key="empty"
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <FiDatabase className="empty-icon" />
                <p>
                  Select patient information and click
                  <strong> Predict Scan </strong>
                  to generate an AI-powered scan recommendation and retrieval comparison.
                </p>
              </motion.div>
            )}

            {result && topPrediction && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35 }}
              >
                <div className="predicted-badge">
                  <FiCheckCircle />
                  {topPrediction.scan.replace(/_/g, " ")}
                </div>

                <p className="confidence-label">
                  Confidence Score: <strong>{Math.round(topPrediction.confidence * 100)}%</strong>
                  <br />
                  Based on {topPrediction.votes} similar historical patient records.
                </p>

                <div className="confidence-bar-track">
                  <motion.div
                    className="confidence-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${topPrediction.confidence * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>

                <p className="explanation">{result.top_explanation}</p>

                <h3 className="race-title">
                  <FiClock /> Retrieval Performance Comparison
                </h3>

                <div className="lane cold">
                  <div className="lane-label">
                    <span className="lane-name">Cold storage (current)</span>
                    <span className="lane-time">{coldMs} ms</span>
                  </div>
                  <div className="track">
                    <motion.div
                      className="fill fill-cold"
                      animate={{
                        width: `${Math.min(
                          (coldMs / result.retrieval_comparison.cold_storage_ms) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="lane fast">
                  <div className="lane-label">
                    <span className="lane-name">ScanAhead pre-fetched</span>
                    <span className="lane-time">{fastMs} ms</span>
                  </div>
                  <div className="track">
                    <motion.div
                      className="fill fill-fast"
                      animate={{
                        width: `${Math.min(
                          (fastMs / result.retrieval_comparison.prefetched_ms) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <AnimatePresence>
                  {raceDone && speedup && (
                    <motion.div
                      className="speedup"
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                    >
                      <FiZap />
                      Average Retrieval Time Reduced by <strong>{speedup}&times;</strong>
                      <br />
                      Predicted scans are available before clinician request.
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>
      </main>
    </div>
  );
}

export default App;