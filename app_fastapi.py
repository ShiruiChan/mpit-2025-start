from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from autobid_utils import feature_candidates, recommend_bid_for_row, compute_score_for_row

MODEL_PATH = "autobid_acceptance_model.joblib"
model = load(MODEL_PATH)

app = FastAPI(title="Drivee AutoBid API", version="0.1")

class Order(BaseModel):
    price_start_local: float
    price_bid_local: float | None = None
    distance_in_meters: float | None = None
    duration_in_seconds: float | None = None
    pickup_in_meters: float | None = None
    pickup_in_seconds: float | None = None
    carname: str | None = None
    carmodel: str | None = None
    platform: str | None = None
    order_hour: int | None = 12
    order_dow: int | None = 3
    is_weekend: int | None = 0
    lag_tender_seconds: float | None = 0
    driver_tenure_days: float | None = 0
    bid_uplift_abs: float | None = None
    bid_uplift_pct: float | None = None
    centrality_proxy: float | None = None

@app.post("/recommend")
def recommend(order: Order):
    row = pd.Series(order.dict())
    row.setdefault("price_bid_local", row["price_start_local"])
    row.setdefault("bid_uplift_abs", row["price_bid_local"] - row["price_start_local"])
    if row["price_start_local"]>0:
        row.setdefault("bid_uplift_pct", row["bid_uplift_abs"]/row["price_start_local"])

    best = recommend_bid_for_row(row, model)
    sc   = compute_score_for_row(row, best)
    return {"best_price": best["price"], "p_accept": best["p_accept"], "expected_rev": best["er"], "score": sc["score"]}
