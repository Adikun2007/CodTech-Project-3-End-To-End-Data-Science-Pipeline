<div align="center">

<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" alt="Live"/>
<img src="https://img.shields.io/badge/Model-XGBoost-blue?style=for-the-badge" alt="XGBoost"/>
<img src="https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge" alt="Render"/>
<img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge" alt="Flask"/>

# 🛡️ FraudShield

### End-to-End Credit Card Fraud Detection Pipeline

*Train → Tune → Deploy → Detect — all in one project.*

**[🌐 Live Demo](https://fraudshield-1vjs.onrender.com/)** &nbsp;·&nbsp; **[📂 GitHub Repo](https://github.com/Adikun2007/CodTech-Project-3-End-To-End-Data-Science-Pipeline)**

</div>

---

## 📌 Overview

FraudShield is a full end-to-end machine learning project that detects credit card fraud in real time. It covers every stage from raw data ingestion and class-imbalance handling, through XGBoost training and precision-tuned threshold selection, to a deployed Flask REST API with an interactive web UI.

| Detail | Value |
|---|---|
| Dataset | Kaggle Credit Card Fraud (284,807 transactions) |
| Model | XGBoost Classifier |
| Best Threshold | **0.9677** |
| Deployment | Render (free tier) |
| API status | `/health` → `{"status": "ok"}` |

---

## 🗂️ Project Structure

```
CodTech-Project-3-End-To-End-Data-Science-Pipeline/
│
├── main.py                   # Full ML pipeline (train, tune, save)
├── app.py                    # Flask API + Web UI server
│
├── fraud_detection_model.pkl # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
├── best_threshold.pkl        # Optimal decision threshold (0.9677)
│
├── templates/
│   └── index.html            # Web UI
│
└── requirements.txt
```

---

## 🔁 ML Pipeline

> Raw data → Clean → Scale → SMOTE → Train → Tune → Threshold → Deploy

```
creditcard.csv
      │
      ▼
 EDA & Cleaning ──► Feature Scaling ──► SMOTE Oversampling
                    (Time, Amount)       (balance minority class)
                                              │
                                              ▼
                              Train/Test Split (StratifiedKFold 80/20)
                                              │
                                              ▼
                                   XGBoost Classifier
                                   RandomizedSearchCV
                                              │
                                              ▼
                             Precision / Recall / ROC-AUC Eval
                                              │
                                              ▼
                                  Custom Threshold Tuning
                                     best = 0.9677
                                              │
                                              ▼
                          Save .pkl artifacts (model, scaler, threshold)
                                              │
                                              ▼
                                  Flask REST API on Render
                                   FRAUD DETECTED / LEGITIMATE
```

**Key design decisions:**
- `StandardScaler` is applied only to `Time` and `Amount` — the 28 PCA features (`V1`–`V28`) are already scaled in the dataset.
- SMOTE is applied only on the training set to prevent data leakage.
- Threshold is tuned on the validation set to maximise F1 for the minority fraud class, not hardcoded at 0.5.

---

## 📡 API Reference

Base URL: `https://fraudshield-1vjs.onrender.com`

### `GET /health`
Check if the API is running.

```bash
curl https://fraudshield-1vjs.onrender.com/health
```

```json
{
  "status": "ok",
  "threshold": 0.9677
}
```

---

### `POST /predict`
Predict whether a transaction is fraudulent.

**Request body** (JSON):
```json
{
  "Time": 406.0,
  "V1": -2.312, "V2": 1.951, "V3": -1.609,
  "...",
  "V28": -0.143,
  "Amount": 239.93
}
```

**Response**:
```json
{
  "fraud": true,
  "probability": 0.9993,
  "threshold_used": 0.9677,
  "verdict": "FRAUD DETECTED"
}
```

| Field | Description |
|---|---|
| `fraud` | `true` if fraud probability ≥ threshold |
| `probability` | Raw fraud score from the model (0–1) |
| `threshold_used` | The saved best threshold |
| `verdict` | Human-readable result string |

---

### `GET /sample`
Returns a real fraudulent transaction from the dataset — useful for testing.

```bash
curl https://fraudshield-1vjs.onrender.com/sample
```

---

## 🖥️ Web UI

The live app lets you paste or load transaction values and get an instant verdict.

- **Load sample** — auto-fills a real fraud transaction
- **Run detection** — calls `/predict` and shows probability + verdict
- **Project summary panel** — shows model, inputs, and output type at a glance

> ⚠️ The app uses real PCA-transformed features from the Kaggle dataset. Inputs must follow the same 30-feature order the model was trained on.

---

## 🚀 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Adikun2007/CodTech-Project-3-End-To-End-Data-Science-Pipeline.git
cd CodTech-Project-3-End-To-End-Data-Science-Pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download `creditcard.csv` from Kaggle:

> 📥 [Kaggle — Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` in the project root.

### 4. Train the model

```bash
python main.py
```

This generates `fraud_detection_model.pkl`, `scaler.pkl`, and `best_threshold.pkl`.

### 5. Start the API

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.x |
| ML | XGBoost, scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| API | Flask |
| Artifact storage | joblib |
| Deployment | Render |

---

## 📊 Results

The model is evaluated on a held-out stratified test set. The decision threshold is tuned to maximise F1-score on the fraud class rather than using the default 0.5 cutoff.

| Metric | Result |
|---|---|
| Best threshold | 0.9677 |
| Target class | Fraud (minority) |
| Tuning strategy | Maximise F1 on validation split |

---

## ⚠️ Disclaimer

This project is built for educational purposes as part of a data science portfolio. Predictions should not be used as the sole basis for financial decisions. Real fraud systems require additional rule-based controls, monitoring, and human review.

---

## 👤 Author

**Adikun** — [github.com/Adikun2007](https://github.com/Adikun2007)

*Built with Flask, scikit-learn, and XGBoost.*
