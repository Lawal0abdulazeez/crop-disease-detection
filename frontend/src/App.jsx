import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, getModelInfo, predictImage, API_BASE } from "./api";
import { getDiseaseInfo } from "./diseaseInfo";

export default function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({ status: "error", model_loaded: false, device: null })
      );
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  const onFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please select a JPEG or PNG image.");
      return;
    }
    setError(null);
    setResult(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  const runPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const disease = result ? getDiseaseInfo(result.predicted_class) : null;
  const confPct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <div className="app">
      <header>
        <h1>Crop Disease Detection</h1>
        <p>
          Upload a leaf photo to identify disease with deep learning and get
          practical treatment guidance.
        </p>
        <div className="status-bar">
          <span
            className={`badge ${
              health?.status === "ok" ? "ok" : "err"
            }`}
          >
            <span className="dot" />
            API {health?.status === "ok" ? "online" : "offline"}
          </span>
          <span
            className={`badge ${
              health?.model_loaded ? "ok" : "err"
            }`}
          >
            <span className="dot" />
            Model {health?.model_loaded ? "loaded" : "not loaded"}
          </span>
          {health?.device && (
            <span className="badge">Device: {health.device}</span>
          )}
          {modelInfo && (
            <span className="badge">
              {modelInfo.model_name} · {modelInfo.num_classes} classes
            </span>
          )}
        </div>
      </header>

      <div className="grid two">
        <section className="card">
          <h2>Upload leaf image</h2>
          <div
            className={`dropzone ${dragOver ? "dragover" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg,image/webp"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <strong>Click to browse</strong> or drag & drop
            <p>JPEG / PNG · leaf close-up works best</p>
          </div>

          {preview && (
            <div className="preview">
              <img src={preview} alt="Leaf preview" />
            </div>
          )}

          <button
            className="btn"
            disabled={!file || loading}
            onClick={runPredict}
          >
            {loading ? (
              <>
                <span className="spinner" /> Analyzing…
              </>
            ) : (
              "Detect disease"
            )}
          </button>
          {file && (
            <button className="btn secondary" onClick={clear} disabled={loading}>
              Clear
            </button>
          )}

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="card">
          <h2>Results</h2>
          {!result && !loading && (
            <p className="placeholder">
              Results appear here after you upload an image and run detection.
            </p>
          )}
          {loading && <p className="placeholder">Running model…</p>}
          {result && (
            <div className="result-main">
              <div className="result-class">{result.predicted_class}</div>
              <div>
                <div className="confidence-label">
                  Confidence · {confPct}%
                  {result.above_threshold ? " · above threshold" : " · low confidence"}
                </div>
                <div className="confidence-bar">
                  <div style={{ width: `${confPct}%` }} />
                </div>
              </div>

              <ul className="topk">
                {result.top_k?.map((item) => (
                  <li key={item.class_index}>
                    <span>{item.class_name}</span>
                    <span>{(item.confidence * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>

              {disease && (
                <div className="treatment">
                  <h3>{disease.title}</h3>
                  <p>{disease.summary}</p>
                  <ul>
                    {disease.treatment.map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      </div>

      <footer>
        API base: <code>{API_BASE}</code>
        {" · "}
        EfficientNet · FastAPI · React
      </footer>
    </div>
  );
}
