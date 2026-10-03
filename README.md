# 🏦 End-to-End Credit Risk & Default Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-brightgreen.svg)](https://peps.python.org/pep-0008/)

A production-ready **Credit Default Risk Prediction** project developed using industry-standard machine learning workflows and MLOps practices. Built for banking and financial applications, it features a modular Scikit-learn Pipeline architecture that estimates the probability of loan default directly from raw applicant profiles.

---

## 📌 Business Context & Objective

In retail banking, loan default occurs when a borrower fails to meet legal debt obligations. Inaccurate credit assessment leads to severe provisioning costs and portfolio losses, whereas overly conservative policies alienate profitable clients.

Key objectives:
- Classify loan applicants into **Non-Default / Safe (0)** or **Default / High Risk (1)**.
- Eliminate **Data Leakage** via an isolated, step-by-step transformation pipeline.
- Export an inference-ready pipeline artifact (`.pkl`) capable of serving real-time predictions.

---

## 🛠️ Architecture & Engineering Principles

1. **Modular Architecture:** Ingestion, pipeline construction, model evaluation, and inference are organized into dedicated modules inside `src/`.
2. **Automated Feature Engineering:** A critical banking metric, `Loan to Income` ratio ($Loan / Income$), is dynamically computed within both training and production inference routines.
3. **Data Leakage Prevention:** Preprocessing steps (`SimpleImputer` for missing values and `StandardScaler` for numeric scaling) are strictly fitted on `X_train` within a `ColumnTransformer` + `Pipeline` setup, completely shielding transformers from test data.
4. **Class Imbalance Handling:** To account for the natural skew in credit default data, models are trained and evaluated with balanced class weights (`class_weight='balanced'`).

---

## 📊 Model Performance & Evaluation

Linear (`Logistic Regression`) and ensemble tree-based (`Random Forest`) models were benchmarked on a stratified 20% holdout test set using the `ROC-AUC` metric:

| Model | ROC-AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.9873 | 94% | 0.71 | 0.96 | 0.81 |
| **Random Forest (Best)** | **1.0000** | **100%** | **1.00** | **1.00** | **1.00** |

*The top-performing **Random Forest Pipeline** is serialized and stored at `models/credit_default_pipeline.pkl`.*

---

## 📂 Project Structure

```text
credit-risk-prediction/
├── data/                    # Raw and interim data assets (ignored via .gitignore)
├── models/                  # Serialized pipeline artifacts (.pkl)
├── src/                     # Core source code modules
│   ├── make_dataset.py      # Automated data acquisition script
│   ├── train.py             # Feature engineering, pipeline training & evaluation
│   └── predict.py           # Production inference module & test scenarios
├── .gitignore               # Ignored environments, cache, and data files
├── requirements.txt         # Project dependencies and pinned library versions
└── README.md                # Comprehensive project documentation
```

---

## 🚀 Setup & Execution

Follow these steps to run the pipeline locally:

### 1. Clone the Repository
```bash
git clone https://github.com/ErmanDurak/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Configure Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fetch the Dataset
```bash
python src/make_dataset.py
```

### 5. Train and Export the Pipeline
```bash
python src/train.py
```

### 6. Run Inference on Test Profiles
```bash
python src/predict.py
```