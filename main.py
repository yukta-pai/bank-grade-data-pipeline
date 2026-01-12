from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os



# ----------------------------
# Load model and WoE mappings
# ----------------------------
with open("lr_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("woe_mappings.pkl", "rb") as f:
    woe_mappings = pickle.load(f)

# ----------------------------
# Scorecard parameters
# ----------------------------
base_score = 600
PDO = 50
factor = PDO / np.log(2)
base_odds = 1/20
offset = base_score - factor * np.log(base_odds)

selected_features = ['int_rate', 'term', 'dti', 'home_ownership', 'annual_inc', 'loan_amnt']



# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Loan Scorecard API")

# Allow front-end JS to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Serve HTML front-end
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_html():
    return FileResponse("index.html")

# ----------------------------
# Input schema
# ----------------------------
class LoanInput(BaseModel):
    int_rate: float
    term: str
    dti: float
    home_ownership: str
    annual_inc: float
    loan_amnt: float

ALLOWED_HOME_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "ANY"}
ALLOWED_TERM = {"36 months", "60 months"}


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")

def predict_loan(data: LoanInput):
    try:
        # Defensive validation
        if data.home_ownership not in ALLOWED_HOME_OWNERSHIP:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid home_ownership value. Allowed: {ALLOWED_HOME_OWNERSHIP}"
            )

        X_input = pd.DataFrame([data.dict()])

        X_woe = pd.DataFrame()
        for col in selected_features:
            optb = woe_mappings[col]
            try:
                X_woe[col] = optb.transform(X_input[col], metric="woe")
            except:
                X_woe[col] = 0  # fill unseen / missing values with 0

        logit = lr.intercept_[0] + np.sum(lr.coef_[0] * X_woe.values)
        pd_pred = 1 / (1 + np.exp(-logit))

        score = offset
        for i, col in enumerate(selected_features):
            score += -lr.coef_[0][i] * X_woe[col].iloc[0] * factor

        return {
            "PD": float(pd_pred),
            "Score": float(score)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
