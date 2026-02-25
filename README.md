## Production-Ready Credit Risk Scorecard for Banking

End-to-End Probability of Default (PD) Modeling and Deployment for Loan Risk Assessment

🚀 Project Overview

This repository contains a production-ready machine learning pipeline for predicting credit risk in banking, developed using real-world Lending Club data. The project covers the full lifecycle of a credit risk model—from raw data ingestion, validation, feature engineering with Weight of Evidence (WoE) transformations, logistic regression modeling, to real-time deployment via a REST API and web interface.

The goal is to deliver a robust, explainable, and deployable credit risk solution suitable for banking and financial institutions.

### 🛠 Key Features

Data Validation & Quality Assurance:

Automatic schema checks, type validation, and quarantining of invalid records.

Ensures only clean, reliable data is fed into the model.

### Feature Engineering:

Weight of Evidence (WoE) transformations for categorical variables.

Numeric scaling and encoding with ColumnTransformer.

### Predictive Modeling:

Logistic Regression for Probability of Default (PD) prediction.

Class imbalance handled with class weighting.

Model evaluation using ROC-AUC, Gini coefficient, and confusion matrices.

### Scorecard Implementation:

Converts PD predictions into an interpretable points-based credit score.

Includes risk band segmentation and decision thresholds.

Explainability & Transparency:

WoE contributions analyzed for key risk drivers (interest rate, income, term, issue date).

Provides insights into which factors influence credit risk most.

### Production Deployment:

REST API built with FastAPI.

Dockerized pipeline for consistent, reproducible deployment.

Web-based interface allowing real-time credit risk evaluation.

### Validation & Reliability

Implemented structured data validation checks for missing values and schema consistency

Ensured reproducibility through Docker containerisation

Designed pipeline to support audit-ready documentation and traceable model outputs

### 📦 Tech Stack

Python 3.11 | Pandas | NumPy | Scikit-learn | Joblib

FastAPI | Docker | ColumnTransformer | OneHotEncoder | StandardScaler

HTML/CSS/JS for frontend visualization

### 📈 Project Impact

This solution demonstrates the ability to:

Build production-grade ML systems for banking applications.

Translate complex risk models into interpretable scorecards.

Deploy models for real-time, web-based inference, bridging data science and engineering.

### 🏗 Project Structure
banking-credit-risk/
├── data/                  # Raw and processed data (not included)
├── pipelines/             # ETL, validation, and training scripts
├── models/                # Saved model artifacts (joblib, metrics.json)
├── api/                   # FastAPI endpoints
├── web/                   # HTML/JS frontend for scorecard input
├── Dockerfile             # Container for validation + inference
├── Dockerfile.train       # Container for model training
├── README.md              # Project documentation

### ⚡ How to Run

### Validate Data

docker build -t bank-grade-pipeline .
docker run --rm -v $(pwd)/data:/app/data bank-grade-pipeline


### Train Model

docker build -f Dockerfile.train -t bank-grade-trainer .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models bank-grade-trainer


### Run API & Web Interface

docker run -p 8000:8000 -v $(pwd)/models:/app/models bank-grade-api

Then open http://localhost:8000 in your browser.
