import sys, pandas as pd, numpy as np
from joblib import load
from autobid_utils import feature_candidates, recommend_bid_for_row

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_predict.py input.csv output.csv")
        return
    inp, out = sys.argv[1], sys.argv[2]
    df = pd.read_csv(inp)
    # базовые фичи, если нет — заполним
    if "order_timestamp" in df.columns:
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_hour"] = df["order_timestamp"].dt.hour
        df["order_dow"]  = df["order_timestamp"].dt.dayofweek
        df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    if "driver_reg_date" in df.columns:
        df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")
        if "order_timestamp" in df.columns:
            df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.days.clip(lower=0)
    if "tender_timestamp" in df.columns and "order_timestamp" in df.columns:
        df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"], errors="coerce")
        df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)

    df["centrality_proxy"] = -df.get("pickup_in_meters", pd.Series([np.nan]*len(df)))

    model = load("autobid_acceptance_model.joblib")
    rec_prices, rec_probs, rec_er = [], [], []
    for _, r in df.iterrows():
        best = recommend_bid_for_row(r, model)
        rec_prices.append(best["price"])
        rec_probs.append(best["p_accept"])
        rec_er.append(best["er"])

    out_df = pd.DataFrame({
        "order_id": df.get("order_id", range(len(df))),
        "recommended_price_bid_local": rec_prices,
        "p_accept": rec_probs,
        "expected_revenue": rec_er
    })
    out_df.to_csv(out, index=False)
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
