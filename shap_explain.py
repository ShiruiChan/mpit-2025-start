import os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

MODEL_CBM = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")
TRAIN_PATH = os.environ.get("TRAIN_PATH", "train.csv")

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
    with open(FNAMES_JSON, "r", encoding="utf-8") as f:
        FEATURES = json.load(f)
    CAT_COLS = [c for c in ["carmodel","carname","platform"] if c in FEATURES]
    cat_idx = [FEATURES.index(c) for c in CAT_COLS]

    model = CatBoostClassifier()
    model.load_model(MODEL_CBM)

    df = pd.read_csv(TRAIN_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = build_features(df)
    X = df[FEATURES].head(5000)  # хватит для summary
    pool = Pool(X, cat_features=cat_idx)

    # SHAP values (CatBoost поддерживает напрямую)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    # Последний столбец — base value; убираем
    sv = shap_values[:, :-1]

    # Summary bar
    plt.figure(figsize=(10,6))
    shap.summary_plot(sv, features=X, feature_names=FEATURES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200)
    plt.close()

    # Waterfall для первой строки
    explainer = shap.TreeExplainer(model)
    one_sv = explainer.shap_values(X.iloc[[0]])  # список из 2 классов; берём положительный (класс=1)
    if isinstance(one_sv, list):
        one_sv = one_sv[1]
    shap.plots._waterfall.waterfall_legacy(
        shap.Explanation(values=one_sv[0], base_values=explainer.expected_value[1], data=X.iloc[0], feature_names=FEATURES),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=200)
    plt.close()

    print("Saved shap_summary.png and shap_waterfall.png")

if __name__ == "__main__":
    main()
