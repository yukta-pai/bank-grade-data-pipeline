import pandas as pd
from pathlib import Path
import json
import re

# =====================================================
# Paths
# =====================================================
RAW_DATA_PATH = Path("data/raw/lendingclub.csv")
QUARANTINE_DIR = Path("data/quarantine")
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Contract
# =====================================================
CONTRACT_COLUMNS = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "issue_d",
    "loan_status",
    "annual_inc",
]

ALLOWED_TERM = {"36 months", "60 months"}

ALLOWED_STATUS = {
    "Fully Paid",
    "Charged Off",
    "Current",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Default",
    "In Grace Period",
}

ISSUE_D_PATTERN = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$")

# =====================================================
# IO
# =====================================================
def read_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="latin1",
        low_memory=False,
        usecols=lambda c: c in CONTRACT_COLUMNS,
    )

def quarantine(df: pd.DataFrame, reason: str):
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = QUARANTINE_DIR / f"quarantine_{ts}.csv"
    json_path = csv_path.with_suffix(".json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(
            {
                "reason": reason,
                "row_count": len(df),
                "columns": list(df.columns),
            },
            f,
            indent=2,
        )

    print(f"⚠️  {len(df)} rows quarantined ({reason}) → {csv_path.name}")

# =====================================================
# Validation
# =====================================================
def validate_schema():
    df = read_raw_data(RAW_DATA_PATH)

    # ---- Presence check (NOT order)
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        quarantine(df, f"Missing required columns: {missing}")
        print("❌ Schema validation failed")
        return

    df = df[CONTRACT_COLUMNS].copy()

    # ---- Normalize strings (CRITICAL)
    df["term"] = df["term"].astype(str).str.strip()
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    df["issue_d"] = df["issue_d"].astype(str).str.strip()

    failures = pd.DataFrame()

    # ---- Numeric convertibility
    for col in ["loan_amnt", "funded_amnt", "int_rate", "annual_inc"]:
        bad = pd.to_numeric(df[col], errors="coerce").isna()
        failures = pd.concat([failures, df[bad]])

    # ---- Term
    failures = pd.concat([failures, df[~df["term"].isin(ALLOWED_TERM)]])

    # ---- Status
    failures = pd.concat([failures, df[~df["loan_status"].isin(ALLOWED_STATUS)]])

    # ---- issue_d
    failures = pd.concat([failures, df[~df["issue_d"].str.match(ISSUE_D_PATTERN)]])

    failures = failures.drop_duplicates()

    if not failures.empty:
        quarantine(failures, "Business rule validation failed")
        df = df.drop(failures.index)

    if df.empty:
        print("❌ All rows failed validation.")
    else:
        print(f"✅ Schema validation passed for {len(df)} rows")

# =====================================================
if __name__ == "__main__":
    validate_schema()
