
import numpy as np, pandas as pd, json
from joblib import load
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from functools import lru_cache
from catboost import Pool

ID_COLS = ["order_id","tender_id","driver_id","user_id"]
NUM_SAFE = [
    "distance_in_meters","duration_in_seconds",
    "pickup_in_meters","pickup_in_seconds",
    "price_start_local","price_bid_local",
    "driver_rating","user_rating",
    "order_hour","order_dow","is_weekend",
    "lag_tender_seconds","driver_tenure_days",
    "bid_uplift_abs","bid_uplift_pct","centrality_proxy"
]
RAW_CAT = ["carmodel","carname","platform"]

def _prepare_te_maps(te_maps_raw: dict) -> dict:
    """Привести TE-карты к виду: {id_col: {"map": dict[str,float], "prior": float}} (один раз)."""
    te = {}
    for idc, obj in (te_maps_raw or {}).items():
        raw_map = obj.get("map", {})
        # Ключи уже строки после JSON, просто убеждаемся и приводим значения к float один раз
        te[idc] = {
            "map": {str(k): float(v) for k, v in raw_map.items()},
            "prior": float(obj.get("prior", 0.5))
        }
    return te
  
def _coerce_dtypes_for_infer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ЧИСЛОВЫЕ: обрабатываем через ndarray, чтобы не трогать Series.name
    for c in NUM_SAFE:
        if c in df.columns:
            arr = pd.to_numeric(np.asarray(df[c]), errors="coerce")
            df.loc[:, c] = arr.astype("float64")

    # TE-колонки -> float64 (тоже через ndarray)
    for c in list(df.columns):
        if isinstance(c, str) and c.startswith("te_"):
            arr = pd.to_numeric(np.asarray(df[c]), errors="coerce")
            df.loc[:, c] = arr.astype("float64")

    # КАТЕГОРИАЛЬНЫЕ: object-строки, но NaN оставляем NaN
    for c in RAW_CAT:
        if c in df.columns:
            mask = df[c].notna()
            # только не-NaN приводим к str; NaN оставляем как есть
            df.loc[mask, c] = df.loc[mask, c].astype(str)
            df.loc[:, c] = df[c].astype("object")

    return df

def load_artifacts(model_path="autobid_catboost.cbm",
                   feature_names_json="cb_feature_names.json",
                   calibrator_path="autobid_isotonic.joblib",
                   te_maps_json="cb_te_maps.json"):
    from catboost import CatBoostClassifier
    from joblib import load
    import json

    model = CatBoostClassifier()
    model.load_model(model_path)
    with open(feature_names_json, "r", encoding="utf-8") as f:
        fns = json.load(f)

    calibrator = None
    try:
        calibrator = load(calibrator_path)
    except Exception:
        pass

    te_maps = {}
    try:
        with open(te_maps_json, "r", encoding="utf-8") as f:
            te_maps_raw = json.load(f)
        te_maps = _prepare_te_maps(te_maps_raw)   # <<< ПОДГОТОВКА ОДИН РАЗ
    except Exception:
        te_maps = {}

    return model, fns, calibrator, te_maps

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])
    for c in ["order_timestamp","tender_timestamp","driver_reg_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"]  = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()
    df["lag_tender_seconds"] = df["lag_tender_seconds"].fillna(0).clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / 86400.0
    df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"].replace(0, np.nan)
    df["bid_uplift_pct"] = df["bid_uplift_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["centrality_proxy"] = np.exp(- df["pickup_in_meters"].fillna(0) / 1000.0)
    for col in ["price_start_local","price_bid_local","bid_uplift_abs","bid_uplift_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df

def apply_te_maps(df: pd.DataFrame, te_maps: dict) -> pd.DataFrame:
    df = df.copy()
    for idc in ID_COLS:
        if idc in df.columns and idc in te_maps:
            m = te_maps[idc]["map"]      # уже готовый dict[str, float]
            prior = te_maps[idc]["prior"]
            df[f"te_{idc}"] = (
                df[idc].astype(str).map(m).fillna(prior).astype(float)
            )
    return df

def build_features_df(df: pd.DataFrame, te_maps: dict) -> pd.DataFrame:
    dfb = build_base_features(df)
    dfb = apply_te_maps(dfb, te_maps)
    return dfb

def _coerce_dtypes_for_infer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # числовые -> float64 (если есть)
    for c in NUM_SAFE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # TE-колонки -> float64
    for c in df.columns:
        if c.startswith("te_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # категорические -> object (строки), но не превращаем NaN в "nan"
    for c in RAW_CAT:
        if c in df.columns:
            # где не NaN — к строке, где NaN — оставим NaN
            mask = df[c].notna()
            df.loc[mask, c] = df.loc[mask, c].astype(str)
            df[c] = df[c].astype("object")

    return df

def predict_accept_prob(df_features: pd.DataFrame, model, fns: dict, calibrator=None):
    needed = fns["cat"] + fns["num"]
    X = df_features.reindex(columns=needed)

    # создать отсутствующие колонки
    for c in needed:
        if c not in X.columns:
            X[c] = 0.0 if (isinstance(c, str) and c.startswith("te_")) else np.nan

    X = X[needed]
    X = _coerce_dtypes_for_infer(X)

    cat_idx = [X.columns.get_loc(c) for c in fns["cat"] if c in X.columns]
    pool = Pool(X, cat_features=cat_idx)

    p = model.predict_proba(pool)[:, 1]

    # защита от NaN/inf перед калибровкой
    p = np.asarray(p, dtype="float64")
    bad = ~np.isfinite(p)
    if bad.any():
        p[bad] = 0.5
    p = np.clip(p, 1e-6, 1 - 1e-6)

    if calibrator is not None:
        p = calibrator.transform(p)

    return p

def _recompute_price_fields(df_row: pd.DataFrame) -> pd.DataFrame:
    # df_row — DataFrame с ОДНОЙ строкой
    idx = df_row.index[0]

    # читаем как float, НЕ меняя тип всей колонки
    start = float(df_row.at[idx, "price_start_local"]) if "price_start_local" in df_row.columns else float("nan")
    bid   = float(df_row.at[idx, "price_bid_local"])   if "price_bid_local" in df_row.columns else float("nan")

    # пишем ТОЛЬКО нужные ячейки
    uplift_abs = bid - start if (start == start and bid == bid) else 0.0  # NaN-safe
    df_row.at[idx, "bid_uplift_abs"] = float(uplift_abs)

    if start and start == start and start != 0.0:
        df_row.at[idx, "bid_uplift_pct"] = float(uplift_abs / start)
    else:
        df_row.at[idx, "bid_uplift_pct"] = 0.0

    # на всякий случай: никаких массовых astype тут НЕ делаем
    return df_row

def recommend_bid_for_row(row, model, fns, calibrator=None, te_maps=None, price_grid_pct=None):
    import numpy as np
    if price_grid_pct is None:
        price_grid_pct = np.arange(0.8, 1.601, 0.025)
    if te_maps is None:
        te_maps = {}

    # 1) Построим БАЗОВЫЕ фичи и TE один раз
    base = pd.DataFrame([row.copy()])
    feat_base = build_base_features(base)      # твоя базовая функция (без TE)
    feat_base = apply_te_maps(feat_base, te_maps)

    # 2) Готовим фрейм нужных столбцов под модель
    needed = fns["cat"] + fns["num"]
    # Если каких-то колонок нет, добавим безопасные значения
    if "price_bid_local" in feat_base.columns:
      feat_base["price_bid_local"] = pd.to_numeric(feat_base["price_bid_local"], errors="coerce").astype("float64")
      
    for c in needed:
        if c not in feat_base.columns:
            feat_base[c] = 0.0 if c.startswith("te_") else np.nan

    start = row.get("price_start_local", np.nan)
    if not (isinstance(start, (int, float)) and np.isfinite(start) and start > 0):
        start = 100.0

    best = {"price": None, "p_accept": None, "er": -1.0}

    for k in price_grid_pct:
        # 3) Клонируем только ОДНУ строку признаков и обновляем ТОЛЬКО ценовые поля
        f = feat_base.copy()
        f.at[f.index[0], "price_bid_local"] = float(start * k)
        f = _recompute_price_fields(f)

        # 4) Подгоняем порядок колонок
        p = predict_accept_prob(f, model, fns, calibrator)
        p1 = float(p[0])
        er = float(f.at[f.index[0], "price_bid_local"] * p1)

        if er > best["er"]:
            best = {"price": float(f.at[f.index[0], "price_bid_local"]), "p_accept": p1, "er": er}

    return best
