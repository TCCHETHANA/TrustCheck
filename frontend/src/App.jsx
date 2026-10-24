import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import {
  Activity, Upload, ShieldCheck, ScanSearch, LoaderCircle,
  AlertCircle, CheckCircle2, FileImage, Download, RefreshCw,
  LayoutDashboard, ChartNoAxesCombined, GitCompareArrows,
  Home, Menu, X, ArrowRight,
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

const VARIATIONS = {
  original: "Original",
  brightness_increased: "Brightness increased",
  contrast_increased: "Contrast increased",
  gaussian_noise: "Gaussian noise",
};

const VARIATION_ORDER = Object.keys(VARIATIONS);

const NAV_ITEMS = [
  { id: "home", label: "Home", icon: Home },
  { id: "experimentA", label: "Experiment A", icon: ChartNoAxesCombined },
  { id: "experimentB", label: "Experiment B · CheXpert", icon: Activity },
  { id: "comparison", label: "Comparison", icon: GitCompareArrows },
  { id: "analysis", label: "Image Analysis", icon: ScanSearch },
];

function formatPercent(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : "—";
}

function parseCSV(text) {
  const lines = String(text || "").trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const parseLine = (line) => {
    const values = [];
    let current = "";
    let quoted = false;

    for (let i = 0; i < line.length; i++) {
      const ch = line[i];

      if (ch === '"' && line[i + 1] === '"' && quoted) {
        current += '"';
        i++;
      } else if (ch === '"') {
        quoted = !quoted;
      } else if (ch === "," && !quoted) {
        values.push(current);
        current = "";
      } else {
        current += ch;
      }
    }

    values.push(current);
    return values;
  };

  const headers = parseLine(lines[0]).map((h) => h.trim());

  return lines.slice(1).filter(Boolean).map((line) => {
    const values = parseLine(line);
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""])
    );
  });
}

function pick(row, candidates) {
  for (const key of candidates) {
    if (row?.[key] !== undefined && row[key] !== "") {
      return row[key];
    }
  }
  return "—";
}

function readable(value) {
  return String(value ?? "—").replaceAll("_", " ");
}

function getAccuracy(evaluation, variation) {
  const value = evaluation?.metrics_by_variation?.[variation]?.accuracy;
  return typeof value === "number" ? value : null;
}

function MetricCard({ label, value, note }) {
  return (
    <div className="evaluation-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}

function SectionTitle({
  icon: Icon = Activity,
  title,
  subtitle,
  action,
}) {
  return (
    <div className="section-heading">
      <div className="heading-icon">
        <Icon size={19} />
      </div>
      <div className="section-heading-copy">
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {action}
    </div>
  );
}

function PageIntro({ eyebrow, title, subtitle }) {
  return (
    <section className="page-intro">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </section>
  );
}

function ChartPanel({ title, subtitle, children }) {
  return (
    <section className="panel chart-panel">
      <div className="chart-heading">
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix) {
    return (
      <p className="muted">
        Confusion matrix data is not available in this API response.
      </p>
    );
  }

  return (
    <div className="table-wrapper">
      <table className="robustness-table confusion-table">
        <thead>
          <tr>
            <th>Actual / Predicted</th>
            {labels.map((label) => <th key={label}>{label}</th>)}
          </tr>
        </thead>
        <tbody>
          {labels.map((actual) => (
            <tr key={actual}>
              <td className="variation-cell">{actual}</td>
              {labels.map((predicted) => {
                const value = matrix?.[actual]?.[predicted];
                return (
                  <td key={predicted} className="matrix-value">
                    <strong>{value ?? "—"}</strong>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VariationAccuracyChart({ evaluation }) {
  const data = VARIATION_ORDER.map((key) => ({
    variation: VARIATIONS[key],
    accuracy: getAccuracy(evaluation, key) === null
      ? null
      : +(getAccuracy(evaluation, key) * 100).toFixed(2),
  }));

  return (
    <ResponsiveContainer width="100%" height={290}>
      <BarChart
        data={data}
        margin={{ top: 10, right: 12, left: 0, bottom: 48 }}
      >
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="variation"
          angle={-18}
          textAnchor="end"
          interval={0}
          height={65}
          tick={{ fontSize: 11 }}
        />
        <YAxis domain={[0, 100]} unit="%" />
        <Tooltip
          formatter={(value) => [
            value == null ? "Unavailable" : `${value}%`,
            "Accuracy",
          ]}
        />
        <Bar
          dataKey="accuracy"
          name="Accuracy"
          fill="#087f8c"
          radius={[7, 7, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ClassPerformanceChart({ evaluation }) {
  const metrics =
    evaluation?.original_model_metrics?.class_metrics ??
    evaluation?.original_model_metrics?.per_class_metrics ??
    evaluation?.class_metrics ??
    {};

  const data = Object.entries(metrics).map(([name, values]) => ({
    className: name,
    Precision: typeof values?.precision === "number"
      ? +(values.precision * 100).toFixed(2) : null,
    Recall: typeof values?.recall === "number"
      ? +(values.recall * 100).toFixed(2) : null,
    "F1-score": typeof (values?.f1_score ?? values?.["f1-score"]) === "number"
      ? +((values.f1_score ?? values["f1-score"]) * 100).toFixed(2)
      : null,
  }));

  if (!data.length) {
    return (
      <p className="muted">
        Class-wise precision, recall and F1-score are not included in the
        summary returned by the backend.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={290}>
      <BarChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="className" tick={{ fontSize: 11 }} />
        <YAxis domain={[0, 100]} unit="%" />
        <Tooltip formatter={(value) => [
          value == null ? "Unavailable" : `${value}%`,
        ]} />
        <Legend />
        <Bar dataKey="Precision" fill="#087f8c" radius={[5, 5, 0, 0]} />
        <Bar dataKey="Recall" fill="#3977d5" radius={[5, 5, 0, 0]} />
        <Bar dataKey="F1-score" fill="#86a96f" radius={[5, 5, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ResultsTable({ rows }) {
  if (!rows?.length) {
    return (
      <div className="empty-inline">
        Detailed CSV rows could not be loaded. Use the CSV download to inspect
        all image-level records.
      </div>
    );
  }

  const columns = [
    {
      label: "Image",
      keys: ["filename", "relative_path", "image_path", "image", "path"],
    },
    {
      label: "Variation",
      keys: ["variation", "condition", "transformation"],
    },
    {
      label: "Actual label",
      keys: ["actual_label", "actual", "true_label", "ground_truth", "label"],
    },
    {
      label: "Predicted label",
      keys: ["predicted_label", "prediction", "predicted", "predicted_class"],
    },
    {
      label: "Correct",
      keys: ["is_correct", "correct", "correct_prediction"],
    },
    {
      label: "Score",
      keys: ["pneumonia_score", "score", "probability"],
    },
  ];

  return (
    <div className="table-wrapper">
      <table className="robustness-table">
        <thead>
          <tr>
            {columns.map((column) => <th key={column.label}>{column.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((row, index) => (
            <tr key={`${pick(row, columns[0].keys)}-${index}`}>
              {columns.map((column) => (
                <td key={column.label}>
                  {readable(pick(row, column.keys))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="table-footnote">
        Previewing {Math.min(rows.length, 12)} of {rows.length} CSV rows.
        Download the CSV for the complete results.
      </p>
    </div>
  );
}

function EvaluationPage({
  title,
  description,
  evaluation,
  loading,
  error,
  onRefresh,
  onDownload,
  rows,
  labels,
}) {
  const metrics = evaluation?.original_model_metrics ?? {};
  const consistency = evaluation?.prediction_consistency?.rate;
  const heatmapSimilarity = evaluation?.mean_perturbed_heatmap_similarity;

  return (
    <section className="panel evaluation-panel">
      <SectionTitle
        icon={ChartNoAxesCombined}
        title={title}
        subtitle={description}
        action={
          <div className="evaluation-actions">
            <button
              className="secondary-button"
              onClick={onRefresh}
              disabled={loading}
            >
              <RefreshCw size={16} />
              Refresh
            </button>
            <button className="secondary-button" onClick={onDownload}>
              <Download size={16} />
              Download CSV
            </button>
          </div>
        }
      />

      {loading && !evaluation && (
        <div className="evaluation-message">
          <LoaderCircle className="spin" size={20} />
          Loading evaluation results…
        </div>
      )}

      {error && (
        <div className="error-message">
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      {evaluation && (
        <>
          <div className="evaluation-stats">
            <MetricCard
              label="Images evaluated"
              value={evaluation.images_evaluated ?? "—"}
              note="Evaluation sample size"
            />
            <MetricCard
              label="Original accuracy"
              value={formatPercent(metrics.accuracy)}
              note="Unperturbed images"
            />
            <MetricCard
              label="Prediction consistency"
              value={formatPercent(consistency)}
              note="Across image variations"
            />
            <MetricCard
              label="Mean heatmap similarity"
              value={formatPercent(heatmapSimilarity)}
              note="Perturbed vs original"
            />
          </div>

          <div className="evaluation-subsection">
            <h3>Accuracy across image variations</h3>
            <p>
              Compare the model's accuracy on the original images and three
              transformed versions.
            </p>
            <VariationAccuracyChart evaluation={evaluation} />
          </div>

          <div className="evaluation-subsection">
            <h3>Class-wise performance</h3>
            <p>
              Precision, recall and F1-score, when available in the evaluation
              summary.
            </p>
            <ClassPerformanceChart evaluation={evaluation} />
          </div>

          <div className="evaluation-subsection">
            <h3>Confusion matrix</h3>
            <p>
              Rows represent actual labels; columns represent predicted labels.
            </p>
            <ConfusionMatrix
              matrix={metrics.confusion_matrix}
              labels={labels}
            />
          </div>

          <div className="evaluation-subsection">
            <h3>Image-level results</h3>
            <p>
              A preview of the detailed CSV generated by the evaluation.
            </p>
            <ResultsTable rows={rows} />
          </div>

          <div className="stability-note">
            These are research evaluation results, not clinical validation.
            Grad-CAM similarity measures visual similarity and does not
            establish explanation correctness.
          </div>
        </>
      )}
    </section>
  );
}

function ComparisonMetric({ title, a, b }) {
  return (
    <div className="panel comparison-metric">
      <span>{title}</span>
      <div className="comparison-values">
        <div>
          <small>Experiment A</small>
          <strong>{formatPercent(a)}</strong>
        </div>
        <div>
          <small>Experiment B</small>
          <strong>{formatPercent(b)}</strong>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState("home");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const [evaluation, setEvaluation] = useState(null);
  const [evaluationRows, setEvaluationRows] = useState([]);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  const [chexpertEvaluation, setChexpertEvaluation] = useState(null);
  const [chexpertRows, setChexpertRows] = useState([]);
  const [chexpertLoading, setChexpertLoading] = useState(false);
  const [chexpertError, setChexpertError] = useState("");

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadEvaluation = async () => {
    setEvaluationLoading(true);
    setEvaluationError("");

    try {
      const response = await axios.get(`${API}/evaluation-report`, {
        timeout: 15000,
      });

      setEvaluation(response.data.summary ?? response.data);
    } catch (err) {
      setEvaluationError(
        err.response?.data?.detail ??
        "Could not load Experiment A results. Check that FastAPI is running."
      );
    } finally {
      setEvaluationLoading(false);
    }

    try {
      const response = await axios.get(`${API}/evaluation-report/csv`, {
        responseType: "text",
        timeout: 30000,
      });
      setEvaluationRows(parseCSV(response.data));
    } catch {
      setEvaluationRows([]);
    }
  };

  const loadChexpertEvaluation = async () => {
    setChexpertLoading(true);
    setChexpertError("");

    try {
      const response = await axios.get(`${API}/evaluation-report/chexpert`, {
        timeout: 15000,
      });

      setChexpertEvaluation(response.data.summary ?? response.data);
    } catch (err) {
      setChexpertError(
        err.response?.data?.detail ??
        "Could not load CheXpert results. Check that FastAPI is running."
      );
    } finally {
      setChexpertLoading(false);
    }

    try {
      const response = await axios.get(
        `${API}/evaluation-report/chexpert/csv`,
        { responseType: "text", timeout: 30000 }
      );
      setChexpertRows(parseCSV(response.data));
    } catch {
      setChexpertRows([]);
    }
  };

  useEffect(() => {
    loadEvaluation();
    loadChexpertEvaluation();
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const downloadCSV = async (endpoint, filename, setError) => {
    setError("");

    try {
      const response = await axios.get(`${API}${endpoint}`, {
        responseType: "blob",
        timeout: 30000,
      });

      const blobURL = URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = blobURL;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(blobURL);
    } catch (err) {
      setError(
        err.response?.data?.detail ??
        `Could not download ${filename}.`
      );
    }
  };

  const goTo = (id) => {
    setActivePage(id);
    setMobileNavOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleFile = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;

    if (!selected.type.startsWith("image/")) {
      setError("Please select an image file.");
      event.target.value = "";
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResults(null);
    setError("");
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Upload a chest X-ray image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API}/gradcam-test`, formData, {
        timeout: 180000,
      });

      setResults(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ??
        "Could not connect to the API. Check that FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const predictions = results?.predictions ?? {};
  const original = predictions.original;
  const similarities = results?.heatmap_cosine_similarity_to_original ?? {};
  const overlays = results?.heatmap_overlays ?? {};
  const overlayEntries = Object.entries(overlays);

  const getLabel = (item) =>
    item?.prediction ?? item?.label ?? "Unavailable";

  const getScoreNumber = (item) => {
    const value = item?.pneumonia_score ?? item?.score;
    return typeof value === "number" && Number.isFinite(value)
      ? value
      : null;
  };

  const getScore = (item) => formatPercent(getScoreNumber(item));

  const getSimilarity = (key) =>
    formatPercent(
      typeof similarities[key] === "number" ? similarities[key] : null
    );

  const allPredictionsMatch = Boolean(
    original?.prediction &&
    Object.keys(predictions).length > 0 &&
    Object.values(predictions).every(
      (item) => item?.prediction === original.prediction
    )
  );

  const imageVariationData = VARIATION_ORDER.map((key) => {
    const score = getScoreNumber(predictions[key]);

    return {
      variation: VARIATIONS[key],
      score: score === null ? null : +(score * 100).toFixed(2),
      label: getLabel(predictions[key]),
    };
  });

  const comparisonData = useMemo(() => {
    return VARIATION_ORDER.map((key) => {
      const a = getAccuracy(evaluation, key);
      const b = getAccuracy(chexpertEvaluation, key);

      return {
        variation: VARIATIONS[key],
        "Experiment A": a === null ? null : +(a * 100).toFixed(2),
        "Experiment B": b === null ? null : +(b * 100).toFixed(2),
      };
    });
  }, [evaluation, chexpertEvaluation]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="brand brand-button"
          onClick={() => goTo("home")}
        >
          <span className="brand-icon"><Activity size={23} /></span>
          <span>TrustCheck<span className="brand-dot">.</span></span>
        </button>

        <div className="research-tag">
          AI EXPLAINABILITY &amp; ROBUSTNESS
        </div>

        <button
          className="mobile-menu-button"
          aria-label="Toggle navigation"
          onClick={() => setMobileNavOpen((value) => !value)}
        >
          {mobileNavOpen ? <X size={21} /> : <Menu size={21} />}
        </button>
      </header>

      <div className="app-layout">
        <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
          <p className="nav-caption">WORKSPACE</p>

          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activePage === id ? "active" : ""}`}
              onClick={() => goTo(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {activePage === id && <span className="nav-active-dot" />}
            </button>
          ))}

          <div className="sidebar-note">
            <ShieldCheck size={18} />
            <span>Research prototype<br />Not for clinical use</span>
          </div>
        </aside>

        <main className="main-content">
          {activePage === "home" && (
            <>
              <section className="hero hero-left">
                <div className="eyebrow">
                  <ShieldCheck size={16} />
                  TRUSTCHECK RESEARCH DASHBOARD
                </div>

                <h1>Look beyond the <span>prediction.</span></h1>

                <p>
                  Evaluate how chest X-ray predictions and Grad-CAM
                  explanations respond to image changes across two
                  evaluation datasets.
                </p>

                <div className="hero-actions">
                  <button
                    className="analyze-button"
                    onClick={() => goTo("experimentA")}
                  >
                    Explore experiments <ArrowRight size={17} />
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => goTo("analysis")}
                  >
                    Analyze an image <ScanSearch size={17} />
                  </button>
                </div>
              </section>

              <section className="home-metrics">
                <MetricCard
                  label="Experiment A sample"
                  value={evaluation?.images_evaluated ?? (evaluationLoading ? "…" : "—")}
                  note="TrustCheck evaluation set"
                />
                <MetricCard
                  label="Experiment A accuracy"
                  value={formatPercent(evaluation?.original_model_metrics?.accuracy)}
                  note="Original images"
                />
                <MetricCard
                  label="Experiment B sample"
                  value={chexpertEvaluation?.images_evaluated ?? (chexpertLoading ? "…" : "—")}
                  note="Selected CheXpert sample"
                />
                <MetricCard
                  label="Experiment B accuracy"
                  value={formatPercent(chexpertEvaluation?.original_model_metrics?.accuracy)}
                  note="Original images"
                />
              </section>

              <div className="page-heading">
                <div>
                  <span className="eyebrow">YOUR WORKSPACE</span>
                  <h2>Explore the project</h2>
                  <p>Each page focuses on one part of the evaluation workflow.</p>
                </div>
              </div>

              <div className="navigation-cards">
                {NAV_ITEMS.filter((item) => item.id !== "home").map(
                  ({ id, label, icon: Icon }, index) => (
                    <button
                      className="navigation-card"
                      key={id}
                      onClick={() => goTo(id)}
                    >
                      <span className={`nav-card-icon tint-${index + 1}`}>
                        <Icon size={22} />
                      </span>
                      <span className="navigation-card-title">{label}</span>
                      <span className="navigation-card-copy">
                        {{
                          experimentA: "Dataset metrics, robustness charts, class-wise results and confusion matrix.",
                          experimentB: "The complete CheXpert evaluation using the same report structure.",
                          comparison: "Compare accuracy, consistency and explanation stability across datasets.",
                          analysis: "Upload an image and inspect predictions across four image variations.",
                        }[id]}
                      </span>
                      <span className="navigation-card-link">
                        Open page <ArrowRight size={15} />
                      </span>
                    </button>
                  )
                )}
              </div>

              <section className="panel research-note">
                <SectionTitle
                  icon={ShieldCheck}
                  title="How to read these results"
                  subtitle="Important context for interpreting the research outputs."
                />
                <p>
                  Accuracy describes performance on the selected evaluation
                  samples. Prediction consistency measures whether predictions
                  stay the same under image changes. Grad-CAM similarity
                  measures visual-map similarity, not explanation correctness.
                  These experiments are research-only and do not establish
                  clinical reliability.
                </p>
              </section>
            </>
          )}

          {activePage === "experimentA" && (
            <>
              <PageIntro
                eyebrow="EXPERIMENT A"
                title="TrustCheck dataset evaluation"
                subtitle="Full evaluation metrics, robustness across four image conditions, class-wise performance and image-level results."
              />
              <EvaluationPage
                title="Experiment A — Dataset Evaluation"
                description="TrustCheck evaluation sample · original and perturbed images"
                evaluation={evaluation}
                loading={evaluationLoading}
                error={evaluationError}
                onRefresh={loadEvaluation}
                onDownload={() => downloadCSV(
                  "/evaluation-report/csv",
                  "trustcheck_robustness_results.csv",
                  setEvaluationError
                )}
                rows={evaluationRows}
                labels={["NORMAL", "PNEUMONIA"]}
              />
            </>
          )}

          {activePage === "experimentB" && (
            <>
              <PageIntro
                eyebrow="EXPERIMENT B"
                title="CheXpert evaluation"
                subtitle="The same depth of evaluation as Experiment A, using the selected CheXpert sample and its own class labels."
              />
              <EvaluationPage
                title="Experiment B — CheXpert"
                description="Selected, balanced 100-image sample · external dataset evaluation"
                evaluation={chexpertEvaluation}
                loading={chexpertLoading}
                error={chexpertError}
                onRefresh={loadChexpertEvaluation}
                onDownload={() => downloadCSV(
                  "/evaluation-report/chexpert/csv",
                  "chexpert_robustness_results.csv",
                  setChexpertError
                )}
                rows={chexpertRows}
                labels={["NORMAL_LIKE", "PNEUMONIA"]}
              />
              <p className="small-note">
                NORMAL_LIKE represents the selected CheXpert No Finding
                category; it does not guarantee that an individual is healthy.
              </p>
            </>
          )}

          {activePage === "comparison" && (
            <>
              <PageIntro
                eyebrow="CROSS-EXPERIMENT ANALYSIS"
                title="Compare the experiments"
                subtitle="Compare the two evaluation samples without treating them as directly interchangeable clinical benchmarks."
              />

              <section className="panel evaluation-panel">
                <SectionTitle
                  icon={GitCompareArrows}
                  title="Accuracy across image variations"
                  subtitle="Accuracy percentage for each condition in both evaluation samples."
                />

                {evaluationLoading || chexpertLoading ? (
                  <div className="evaluation-message">
                    <LoaderCircle className="spin" size={20} />
                    Loading both reports…
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      data={comparisonData}
                      margin={{ top: 10, right: 15, left: 0, bottom: 45 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="variation"
                        angle={-15}
                        textAnchor="end"
                        interval={0}
                        height={60}
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis domain={[0, 100]} unit="%" />
                      <Tooltip formatter={(value) => [
                        value == null ? "Unavailable" : `${value}%`,
                      ]} />
                      <Legend />
                      <Bar
                        dataKey="Experiment A"
                        fill="#087f8c"
                        radius={[5, 5, 0, 0]}
                      />
                      <Bar
                        dataKey="Experiment B"
                        fill="#3977d5"
                        radius={[5, 5, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </section>

              <div className="comparison-metrics-grid">
                <ComparisonMetric
                  title="Original accuracy"
                  a={evaluation?.original_model_metrics?.accuracy}
                  b={chexpertEvaluation?.original_model_metrics?.accuracy}
                />
                <ComparisonMetric
                  title="Prediction consistency"
                  a={evaluation?.prediction_consistency?.rate}
                  b={chexpertEvaluation?.prediction_consistency?.rate}
                />
                <ComparisonMetric
                  title="Mean heatmap similarity"
                  a={evaluation?.mean_perturbed_heatmap_similarity}
                  b={chexpertEvaluation?.mean_perturbed_heatmap_similarity}
                />
              </div>

              <section className="panel evaluation-panel">
                <SectionTitle
                  icon={LayoutDashboard}
                  title="Metric comparison table"
                  subtitle="A compact view of the reported evaluation metrics."
                />

                <div className="table-wrapper">
                  <table className="robustness-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Experiment A</th>
                        <th>Experiment B · CheXpert</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Images evaluated</td>
                        <td>{evaluation?.images_evaluated ?? "—"}</td>
                        <td>{chexpertEvaluation?.images_evaluated ?? "—"}</td>
                      </tr>
                      <tr>
                        <td>Original accuracy</td>
                        <td>{formatPercent(evaluation?.original_model_metrics?.accuracy)}</td>
                        <td>{formatPercent(chexpertEvaluation?.original_model_metrics?.accuracy)}</td>
                      </tr>
                      <tr>
                        <td>Prediction consistency</td>
                        <td>{formatPercent(evaluation?.prediction_consistency?.rate)}</td>
                        <td>{formatPercent(chexpertEvaluation?.prediction_consistency?.rate)}</td>
                      </tr>
                      <tr>
                        <td>Mean perturbed heatmap similarity</td>
                        <td>{formatPercent(evaluation?.mean_perturbed_heatmap_similarity)}</td>
                        <td>{formatPercent(chexpertEvaluation?.mean_perturbed_heatmap_similarity)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="stability-note">
                  The samples differ in source and composition. Compare these
                  figures as results for their respective evaluation samples,
                  not as a controlled head-to-head clinical benchmark.
                </div>
              </section>
            </>
          )}

          {activePage === "analysis" && (
            <>
              <PageIntro
                eyebrow="SINGLE-IMAGE ANALYSIS"
                title="Upload and analyze an X-ray"
                subtitle="Inspect the model prediction, pneumonia score, robustness across four image variations, and Grad-CAM overlays."
              />

              <section className="workspace analysis-workspace">
                <div className="panel upload-panel">
                  <SectionTitle
                    icon={Upload}
                    title="Upload X-ray"
                    subtitle="Select a PNG or JPEG image to begin."
                  />

                  <label className="dropzone">
                    {preview ? (
                      <img
                        className="xray-preview"
                        src={preview}
                        alt="Uploaded X-ray preview"
                      />
                    ) : (
                      <>
                        <div className="upload-symbol"><Upload size={27} /></div>
                        <strong>Choose an image</strong>
                        <span>Click here to browse your files</span>
                        <small>PNG, JPG or JPEG</small>
                      </>
                    )}
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg"
                      onChange={handleFile}
                      hidden
                    />
                  </label>

                  {file && (
                    <p className="file-name">
                      <FileImage size={15} /> {file.name}
                    </p>
                  )}

                  <button
                    className="analyze-button"
                    onClick={analyzeImage}
                    disabled={!file || loading}
                  >
                    {loading ? (
                      <>
                        <LoaderCircle className="spin" size={18} />
                        Analyzing…
                      </>
                    ) : (
                      <>
                        <ScanSearch size={18} />
                        Analyze image
                      </>
                    )}
                  </button>

                  {error && (
                    <div className="error-message">
                      <AlertCircle size={17} />
                      <span>{String(error)}</span>
                    </div>
                  )}
                </div>

                <div className="panel result-panel">
                  <SectionTitle
                    icon={Activity}
                    title="Analysis overview"
                    subtitle="Prediction and robustness summary."
                  />

                  {!results ? (
                    <div className="empty-state">
                      <div className="empty-symbol">
                        <ScanSearch size={30} />
                      </div>
                      <strong>Your results will appear here</strong>
                      <p>
                        Upload an X-ray and run analysis to compare the model
                        outputs.
                      </p>
                    </div>
                  ) : (
                    <div className="result-summary">
                      <span className="result-label">MODEL PREDICTION</span>
                      <h3>{getLabel(original)}</h3>

                      <div className="metric">
                        <span>Pneumonia score</span>
                        <strong>{getScore(original)}</strong>
                      </div>
                      <div className="metric">
                        <span>Variations returned</span>
                        <strong>{Object.keys(predictions).length} / 4</strong>
                      </div>
                      <div className="metric">
                        <span>Prediction consistency</span>
                        <strong>
                          {allPredictionsMatch ? "Consistent" : "Varies / unavailable"}
                        </strong>
                      </div>

                      <p className="disclaimer">
                        Research output only. The score is not a calibrated
                        probability or medical diagnosis.
                      </p>
                    </div>
                  )}
                </div>
              </section>

              {results && (
                <>
                  <section className="panel evaluation-panel">
                    <SectionTitle
                      icon={GitCompareArrows}
                      title="Four-variation comparison graph"
                      subtitle="Pneumonia score for each version of the uploaded image. Missing scores are shown as unavailable."
                    />

                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={imageVariationData}
                        margin={{ top: 10, right: 12, left: 0, bottom: 50 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="variation"
                          angle={-18}
                          textAnchor="end"
                          interval={0}
                          height={65}
                          tick={{ fontSize: 11 }}
                        />
                        <YAxis domain={[0, 100]} unit="%" />
                        <Tooltip
                          formatter={(value, name, props) => [
                            props.payload.label === "Unavailable"
                              ? "Unavailable"
                              : `${value}%`,
                            "Pneumonia score",
                          ]}
                        />
                        <Bar
                          dataKey="score"
                          name="Pneumonia score"
                          fill="#087f8c"
                          radius={[6, 6, 0, 0]}
                        >
                          {imageVariationData.map((item) => (
                            <Cell
                              key={item.variation}
                              fill={item.variation === "Original" ? "#087f8c" : "#3977d5"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>

                    <p className="table-footnote">
                      A score change does not by itself establish a class
                      change or clinical significance.
                    </p>
                  </section>

                  <section className="panel evaluation-panel">
                    <SectionTitle
                      icon={LayoutDashboard}
                      title="Image-level results table"
                      subtitle="Prediction and explanation similarity for each variation."
                    />

                    <div className="table-wrapper">
                      <table className="robustness-table">
                        <thead>
                          <tr>
                            <th>Image variation</th>
                            <th>Prediction</th>
                            <th>Pneumonia score</th>
                            <th>Heatmap similarity</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {VARIATION_ORDER.map((key) => {
                            const item = predictions[key];
                            const matches = Boolean(
                              original?.prediction &&
                              item?.prediction === original.prediction
                            );

                            return (
                              <tr key={key}>
                                <td className="variation-cell">{VARIATIONS[key]}</td>
                                <td className="prediction-value">{getLabel(item)}</td>
                                <td className="number-cell">{getScore(item)}</td>
                                <td className="number-cell">{getSimilarity(key)}</td>
                                <td>
                                  {key === "original" ? (
                                    <span className="status-badge baseline">Baseline</span>
                                  ) : matches ? (
                                    <span className="status-badge consistent">
                                      <CheckCircle2 size={14} /> Consistent
                                    </span>
                                  ) : (
                                    <span className="status-badge different">
                                      <AlertCircle size={14} /> Different / unavailable
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    <div className="stability-note">
                      A single uploaded image has no ground-truth label by
                      default, so its predictions cannot produce a valid
                      confusion matrix. The matrices below are aggregate
                      results from the labeled evaluation datasets.
                    </div>
                  </section>

                  <div className="chart-grid reference-matrices">
                    <ChartPanel
                      title="Experiment A reference confusion matrix"
                      subtitle="Aggregate TrustCheck evaluation results, not this single image."
                    >
                      <ConfusionMatrix
                        matrix={evaluation?.original_model_metrics?.confusion_matrix}
                        labels={["NORMAL", "PNEUMONIA"]}
                      />
                    </ChartPanel>

                    <ChartPanel
                      title="Experiment B reference confusion matrix"
                      subtitle="Aggregate CheXpert results, not this single image."
                    >
                      <ConfusionMatrix
                        matrix={chexpertEvaluation?.original_model_metrics?.confusion_matrix}
                        labels={["NORMAL_LIKE", "PNEUMONIA"]}
                      />
                    </ChartPanel>
                  </div>

                  <section className="panel explanation-panel">
                    <SectionTitle
                      icon={ScanSearch}
                      title="Grad-CAM explanation stability"
                      subtitle="Overlays returned by the API for each image variation."
                    />

                    {overlayEntries.length ? (
                      <div className="heatmap-grid">
                        {overlayEntries.map(([name, data]) => (
                          <div className="heatmap-card" key={name}>
                            <h3>{VARIATIONS[name] ?? name.replaceAll("_", " ")}</h3>
                            <img
                              src={data}
                              alt={`${name} Grad-CAM overlay`}
                            />

                            <div className="heatmap-details">
                              <span>Prediction</span>
                              <strong>{getLabel(predictions[name])}</strong>
                            </div>
                            <div className="heatmap-details">
                              <span>Pneumonia score</span>
                              <strong>{getScore(predictions[name])}</strong>
                            </div>
                            <div className="heatmap-details">
                              <span>Heatmap similarity</span>
                              <strong>{getSimilarity(name)}</strong>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">
                        No heatmap overlays were returned by the API.
                      </p>
                    )}

                    <div className="stability-note">
                      Higher similarity means the heatmaps are more alike
                      according to this metric; it does not mean the
                      explanation is correct or clinically reliable.
                    </div>
                  </section>
                </>
              )}
            </>
          )}

          <footer>
            <span>TrustCheck · MedAI-nEXUS 2026</span>
            <span>For research and educational use only</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;