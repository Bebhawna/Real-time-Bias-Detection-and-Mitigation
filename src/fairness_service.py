from fastapi import FastAPI
from src.db_config import fetch_latest_records
from src.fairness_matrics import compute_fairness_metrics
import pandas as pd

app = FastAPI()

@app.get("/fairness")
def check_fairness():

    records = fetch_latest_records(100)

    if not records:
        return {"dpd": None, "di": None}

    df = pd.DataFrame(records)

    metrics = compute_fairness_metrics(df)

    dpd = metrics["demographic_parity_difference"]
    di = metrics["disparate_impact"]

    return {"dpd": dpd, "di": di}
