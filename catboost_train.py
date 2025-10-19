import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
from joblib import dump

TRAIN_PATH = os.environ.get("TRAIN_PATH", "train.csv")
MODEL_CBM  = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")
CALIBRATOR_PATH = os.environ.get("CALIBRATOR_PATH", "autobid_isotonic.joblib")
TE_MAPS_JSON = os.environ.get("TE_MAPS_JSON", "cb_te_maps.json")

ID_COLS = ["order_id","tender_id","driver_id","user_id"]
RAW_CAT = ["carmodel","carname","platform"]
NUM_BASE = [
    "distance_in_meters","duration_in_seconds",
    "pickup_in_meters","pickup_in_seconds",
    "price_start_local","price_bid_local",
    "driver_rating","user_rating"
]

def _as_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])
    for c in ["order_timestamp","tender_timestamp","driver_reg_date"]:
        _as_dt(df, c)
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"]  = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype("Int64")
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()
    df["lag_tender_seconds"] = df["lag_tender_seconds"].fillna(0).clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / 86400.0
    df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"].replace(0, np.nan)
    df["bid_uplift_pct"] = df["bid_uplift_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["centrality_proxy"] = np.exp(- df["pickup_in_meters"].fillna(0) / 1000.0)
    return df

def cv_target_encode(train_df, col, y, n_splits=5, min_samples=20, prior=0.5):
    """Leakage-safe CV target encoding; returns encoded column and fitted global mapping."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    enc = np.zeros(len(train_df), dtype=float)
    global_map = {}
    for tr_idx, val_idx in skf.split(train_df, y):
        part = train_df.iloc[tr_idx]
        stats = part.groupby(col)["accepted"].agg(["mean","count"])
        stats["enc"] = (stats["mean"] * stats["count"] + prior * min_samples) / (stats["count"] + min_samples)
        fold_map = stats["enc"].to_dict()
        enc[val_idx] = train_df.iloc[val_idx][col].map(fold_map).fillna(prior).values
        global_map.update(fold_map)
    return enc, global_map, prior

def main():
    df = pd.read_csv(TRAIN_PATH)
    # target
    y = (df["is_done"].astype(str).str.lower() == "done").astype(int).values
    df_feat = build_features_df(df)

    te_maps = {}
    for idc in ID_COLS:
        if idc in df_feat.columns:
            enc, gmap, prior = cv_target_encode(
                pd.concat([df_feat[[idc]], pd.Series(y, name="accepted")], axis=1),
                idc, y
            )
            df_feat[f"te_{idc}"] = enc
            te_maps[idc] = {"map": gmap, "prior": prior}

    # Feature list
    cat_cols = [c for c in RAW_CAT if c in df_feat.columns]
    num_cols = [c for c in [
        *NUM_BASE, "order_hour","order_dow","is_weekend","lag_tender_seconds",
        "driver_tenure_days","bid_uplift_abs","bid_uplift_pct","centrality_proxy",
        *[f"te_{c}" for c in ID_COLS if f"te_{c}" in df_feat.columns]
    ] if c in df_feat.columns]

    X = df_feat[cat_cols + num_cols]
    from sklearn.model_selection import train_test_split
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    mono = []
    for col in cat_cols:
        mono.append(0)
    for col in num_cols:
        if col in ["price_bid_local","bid_uplift_abs","bid_uplift_pct"]:
            mono.append(-1)
        else:
            mono.append(0)

    train_pool = Pool(X_train, y_train, cat_features=[X.columns.get_loc(c) for c in cat_cols])
    cal_pool   = Pool(X_cal, y_cal,   cat_features=[X.columns.get_loc(c) for c in cat_cols])

    pos_w = (len(y) - y.sum()) / y.sum()
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=8,
        learning_rate=0.05,
        iterations=1500,
        l2_leaf_reg=6.0,
        random_seed=42,
        eval_metric="AUC",
        use_best_model=True,
        od_type="Iter",
        od_wait=100,
        monotone_constraints=mono,
        scale_pos_weight=pos_w
    )
    model.fit(train_pool, eval_set=cal_pool, verbose=200)
    p_raw = model.predict_proba(cal_pool)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_cal)
    p_cal = iso.transform(p_raw)
    from sklearn.metrics import roc_auc_score, log_loss
    print(f"Holdout AUC raw={roc_auc_score(y_cal, p_raw):.4f}  cal={roc_auc_score(y_cal, p_cal):.4f} "
          f"  logloss raw={log_loss(y_cal, p_raw):.4f} cal={log_loss(y_cal, p_cal):.4f}")

    model.save_model(MODEL_CBM)
    dump(iso, CALIBRATOR_PATH)
    with open(FNAMES_JSON, "w", encoding="utf-8") as f:
        json.dump({"cat": cat_cols, "num": num_cols}, f, ensure_ascii=False, indent=2)
    with open(TE_MAPS_JSON, "w", encoding="utf-8") as f:
        json.dump(te_maps, f)  # persisted TE maps for inference
    print(f"Saved: {MODEL_CBM}, {CALIBRATOR_PATH}, {FNAMES_JSON}, {TE_MAPS_JSON}")

if __name__ == "__main__":
    main()
