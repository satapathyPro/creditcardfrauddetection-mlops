# Credit Card Fraud Detection — MLOps Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white" alt="Grafana"/>
</p>

<p align="center">
  <strong>End-to-end MLOps pipeline for credit card fraud detection — automated training, model versioning, real-time inference, and interactive dashboard.</strong>
</p>

---

## Project Overview

This project delivers a production-grade Credit Card Fraud Detection system built around MLOps principles. The system automates the entire ML lifecycle — from raw data ingestion through model training, evaluation, versioning, and serving — with a real-time dashboard for monitoring fraud detection outcomes.

The stack combines supervised (Logistic Regression, Random Forest, XGBoost) and unsupervised (Clustering) approaches on a 28-feature PCA-transformed transaction dataset, with LangChain/OpenAI integration for human-readable prediction explanations.

---

## Pipeline Architecture

```
Raw Transaction Data
        │
        ▼
┌───────────────────┐
│  Data Ingestion   │  ← Automated pipeline, schema validation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing    │  ← StandardScaler, feature engineering, class balancing
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Training   │  ← Logistic Regression, Random Forest, XGBoost
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Evaluation      │  ← Precision, Recall, F1, ROC-AUC, confusion matrix
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Deployment      │  ← Dockerized Flask API, CI/CD
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Real-time Dashboard│ ← Live predictions, risk segmentation, Prometheus + Grafana
└───────────────────┘
```

---

## MLOps Practices

| Practice | Implementation |
|---|---|
| **Automated Pipelines** | End-to-end data ingestion and preprocessing pipelines eliminate manual steps |
| **Model Versioning** | Trained models are versioned and stored for reproducibility and rollback |
| **CI/CD** | Containerized deployments via Docker with continuous integration and delivery |
| **Real-time Monitoring** | Prometheus metrics + Grafana dashboards track prediction latency, drift, and throughput |
| **Explainability** | LangChain/OpenAI generates natural-language explanations for every prediction |
| **Risk Segmentation** | Predictions classified as high / medium / low risk with confidence scores |
| **Reproducibility** | Jupyter notebooks document experiment tracking and model comparison |

---

## Model Performance

The system evaluates all models against the following metrics on the held-out test set:

| Metric | Description |
|---|---|
| **Precision** | Fraction of flagged transactions that are genuinely fraudulent |
| **Recall** | Fraction of actual fraud cases caught by the model |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the receiver operating characteristic curve |
| **Fraud Probability** | Per-transaction confidence score (0–100%) surfaced in the dashboard |

Models compared: Logistic Regression, Random Forest, XGBoost. Class imbalance is addressed via StandardScaler normalization and evaluation on stratified splits.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **ML Models** | Logistic Regression, Random Forest, XGBoost |
| **Data Processing** | Pandas, Scikit-learn (StandardScaler) |
| **Notebooks** | Jupyter |
| **Backend API** | Flask |
| **Explainability** | LangChain + OpenAI API |
| **Frontend** | HTML, CSS |
| **Containerization** | Docker |
| **Monitoring** | Prometheus + Grafana |
| **Future Hosting** | Google Cloud / Azure / Vercel |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Docker (for containerized deployment)
- OpenAI API key (for prediction explanations)

### 1. Clone the Repository

```bash
git clone https://github.com/satapathyPro/creditcardfrauddetection-mlops.git
cd creditcardfrauddetection-mlops
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# OR
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key
```

### 5. Run the Flask Application

```bash
flask run
```

The API will be available at `http://127.0.0.1:5000`.

### 6. Build and Run with Docker

```bash
# Build the image
docker build -t fraud-detection-mlops .

# Run the container
docker run -d -p 5000:5000 fraud-detection-mlops
```

### 7. Make Predictions

Navigate to `http://127.0.0.1:5000` in your browser, then:

1. **Select a model** — Logistic Regression, Random Forest, or XGBoost
2. **Enter transaction features** — 28 PCA features, time, and amount
3. **View the prediction** — fraud / legitimate, risk tier, and confidence score
4. **Read the explanation** — LangChain-generated natural language summary

**Testing via cURL:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
    -d "features=1.0,-1.2,0.5,2.3,-0.7,0.1,-0.4,1.5,0.9,0.2,0.8,-0.6,0.3,-0.2,-1.0,0.5,0.1,-0.7,0.2,1.1,-0.3,-0.8,1.4,0.6,0.4,-1.3,0.1,4.6" \
    -d "time=12000" \
    -d "amount=100.0" \
    -d "model=Logistic Regression"
```

---

## Key Features

- **Multi-model inference** — switch between Logistic Regression, Random Forest, and XGBoost at runtime
- **Risk segmentation** — transactions classified as high / medium / low risk
- **Fraud probability score** — percentage confidence surfaced per prediction
- **AI-powered explanations** — LangChain/OpenAI generates readable summaries
- **Real-time dashboard** — live fraud detection monitoring with Prometheus + Grafana
- **Dockerized deployment** — portable, reproducible containers ready for cloud hosting

---

## Roadmap

- [ ] **Enhanced Visualizations** — Plotly charts for transaction risk distribution
- [ ] **Batch Analysis** — upload a transaction CSV and receive bulk fraud scoring
- [ ] **Live Transaction Feed** — real-time API integration for streaming transactions
- [ ] **Hyperparameter Tuning** — automated grid/random search with MLflow tracking
- [ ] **Cloud Deployment** — production deployment on Google Cloud, Azure, or Vercel
- [ ] **Drift Detection** — automated alerts when feature or concept drift is detected

---

## Maintainer

**Subham Satapathy** — Software Engineer with 6+ years of experience building cloud-scale distributed systems and production observability platforms. Specializes in Python, Scala, and AI automation with a focus on robust MLOps pipelines and scalable backend architectures.

- **GitHub**: [satapathyPro](https://github.com/satapathyPro)
- **LinkedIn**: [subhamumd](https://www.linkedin.com/in/subhamumd/)
- **Email**: satapathypro@gmail.com
