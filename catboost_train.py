import json, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

TRAIN_PATH = os.environ.get("TRAIN_PATH", "train.csv")
MODEL_CBM = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")

# Базовые списки фич
CAT_COLS = ["carmodel","carname","platform"]
NUM_COLS = [
    "distance_in_meters","duration_in_seconds",
    "pickup_in_meters","pickup_in_seconds",
    "price_start_local","price_bid_local",
    "order_hour","order_dow","is_weekend",
    "lag_tender_seconds","driver_tenure_days",
    "bid_uplift_abs","bid_uplift_pct",
    "centrality_proxy"
]
FEATURES = NUM_COLS + CAT_COLS

def _as_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _as_dt(df, "order_timestamp"); _as_dt(df, "tender_timestamp"); _as_dt(df, "driver_reg_date")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"] = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()
    df["lag_tender_seconds"] = df["lag_tender_seconds"].fillna(0).clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.days
    df["driver_tenure_days"] = df["driver_tenure_days"].fillna(0).clip(lower=0)
    df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"].replace(0, np.nan)
    df["centrality_proxy"] = -df.get("pickup_in_meters", pd.Series([np.nan]*len(df)))
    return df

def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Нет {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "is_done" not in df.columns:
        raise ValueError("Ожидаю столбец is_done ('done'/'cancel')")
    df["is_done"] = df["is_done"].map({"done":1,"cancel":0}).astype(int)

    df = build_features(df)

    # CatBoost: категориальные — по индексам
    cat_idx = [FEATURES.index(c) for c in CAT_COLS if c in FEATURES]
    # Monotone: price_bid_local должен уменьшать P(accept) → -1
    # Соберём массив ограничений по числовым фичам (в порядке FEATURES)
    mono = []
    for f in FEATURES:
        if f == "price_bid_local":
            mono.append(-1)
        else:
            mono.append(0)

    # Тренировочный Pool
    X = df[FEATURES]
    y = df["is_done"].values
    train_pool = Pool(X, y, cat_features=cat_idx)

    model = CatBoostClassifier(
        depth=6, learning_rate=0.08, iterations=1200,
        loss_function="Logloss", eval_metric="Logloss",
        random_seed=42, verbose=200,
        monotone_constraints=mono,
        allow_const_label=True
    )
    model.fit(train_pool)

    model.save_model(MODEL_CBM)
    with open(FNAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(FEATURES, f, ensure_ascii=False, indent=2)

    print(f"Saved: {MODEL_CBM} and {FNAMES_JSON}")

if __name__ == "__main__":
    main()
