import os, sys, json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

try:
    import joblib
except Exception:
    joblib = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# -------------------------
# Logging
# -------------------------
VERBOSE = os.environ.get("AUTOBID_VERBOSE", "1") not in ("0", "false", "False")
def _log(msg: str):
    if VERBOSE:
        print(f"[autobid] {msg}", file=sys.stderr, flush=True)

def _ensure_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Привести ВСЕ имена колонок к строкам
    if not all(isinstance(c, str) for c in df.columns):
        df = df.rename(columns=lambda c: str(c))
    return df


def load_artifacts(
    model_path: str = "autobid_catboost.cbm",
    fns_path: str = "cb_feature_names.json",
    calibrator_path: str = "autobid_isotonic.joblib",
    te_maps_path: str = "cb_te_maps.json",
):
    if CatBoostClassifier is None:
      raise RuntimeError("catboost is not installed")
    model = CatBoostClassifier()
    model.load_model(model_path)
    _log(f"Loaded CatBoost model: {model_path}")
    
    model = CatBoostClassifier()
    model.load_model(model_path)

    with open(fns_path, "r", encoding="utf-8") as f:
        fns = json.load(f)
    # Нормализуем имена фич в fns
    fns["num"] = [str(c) for c in fns.get("num", []) if c is not None and str(c) != ""]
    fns["cat"] = [str(c) for c in fns.get("cat", []) if c is not None and str(c) != ""]
    _log(f"Feature names: num={len(fns['num'])}, cat={len(fns['cat'])}")

    calibrator = None
    if joblib is not None:
        try:
            calibrator = joblib.load(calibrator_path)
            _log(f"Loaded calibrator: {calibrator_path}")
        except Exception:
            calibrator = None
            _log("Calibrator not found/disabled.")

    try:
        with open(te_maps_path, "r", encoding="utf-8") as f:
            te_maps = json.load(f)
            if isinstance(te_maps, list):
                te_maps = dict(te_maps)
    except Exception:
        te_maps = {}
    _log(f"TE-maps loaded: {len(te_maps)} entries")

    return model, fns, calibrator, te_maps


_ID_COLS = ["order_id", "tender_id", "driver_id", "user_id"]

def apply_te_maps(df: pd.DataFrame, te_maps: Dict[str, float], prior: float = 0.5) -> pd.DataFrame:
    if not te_maps:
        return df

    out = _ensure_str_columns(df.copy())
    for idc in [c for c in _ID_COLS if c in out.columns]:
        te_col = f"te_{idc}"
        # Числовизация до fillna → без FutureWarning
        mapped = pd.to_numeric(out[idc].astype(str).map(te_maps), errors="coerce")
        out.loc[:, te_col] = mapped.fillna(float(prior)).astype("float64")
    return out


def _coerce_dtypes_for_infer(df: pd.DataFrame, fns: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()

    out = _ensure_str_columns(df.copy())

    # Нормализуем имена фич из fns в строки
    num_cols = [str(c) for c in fns.get("num", []) if c is not None and str(c) != ""]
    cat_cols = [str(c) for c in fns.get("cat", []) if c is not None and str(c) != ""]

    for c in num_cols:
        if c not in out.columns:
            out.loc[:, c] = 0.0
    for c in cat_cols:
        out.loc[:, c] = out[c].astype(str)

    for c in num_cols:
        arr = pd.to_numeric(np.asarray(out[c]), errors="coerce")
        out.loc[:, c] = np.nan_to_num(arr, nan=0.0).astype("float64")
    for c in cat_cols:
        mask = out[c].notna()
        out.loc[mask, c] = out.loc[mask, c].astype(str)
        out.loc[:, c] = out[c].astype("object")

    ordered = cat_cols + num_cols
    return out[ordered]


def build_features_df(df_row_like: pd.DataFrame, te_maps: Dict[str, float]) -> pd.DataFrame:
    if not isinstance(df_row_like, pd.DataFrame):
        df = pd.DataFrame([df_row_like])
    else:
        df = df_row_like.copy()
    df = _ensure_str_columns(df)

    price_cols = [c for c in ["price_bid_local", "price_start_local", "bid_uplift_abs"] if c in df.columns]
    for c in price_cols:
        df.loc[:, c] = pd.to_numeric(np.asarray(df[c]), errors="coerce").astype("float64")

    df = apply_te_maps(df, te_maps, prior=0.5)
    return df


def predict_accept_prob(df_features: pd.DataFrame, model, fns, calibrator=None) -> np.ndarray:
    X = _coerce_dtypes_for_infer(df_features, fns)
    if VERBOSE:
        _log(f"Predict on shape={X.shape}; cats={len(fns.get('cat', []))}, nums={len(fns.get('num', []))}")
    p = model.predict_proba(X)[:, 1]
    p = np.asarray(p, dtype="float64")
    if calibrator is not None:
        p = calibrator.transform(p)
        p = np.asarray(p, dtype="float64")
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return p


def _recompute_price_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_str_columns(df.copy())
    for c in ["price_bid_local", "price_start_local", "bid_uplift_abs"]:
        if c in out.columns:
            out.loc[:, c] = pd.to_numeric(np.asarray(out[c]), errors="coerce").astype("float64")
    return out


def recommend_bid_for_row(row: pd.Series, model, fns, calibrator=None, te_maps: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    base = build_features_df(pd.DataFrame([row]), te_maps or {})
    base = base.reset_index(drop=True)
    idx0 = base.index[0]
    for c in ("price_bid_local", "price_start_local", "bid_uplift_abs"):
        if c in base.columns:
            base.loc[:, c] = pd.to_numeric(base[c], errors="coerce").astype("float64")
    start = None
    if "price_start_local" in base.columns and pd.notna(base.at[idx0, "price_start_local"]):
        start = float(base.at[idx0, "price_start_local"])
    elif "price_bid_local" in base.columns and pd.notna(base.at[idx0, "price_bid_local"]):
        start = float(base.at[idx0, "price_bid_local"])
    else:
        start = 0.0
        base["price_start_local"] = float(start)
        base.at[idx0, "price_start_local"] = float(start)
    _log(f"Row start price: {start}")

    ks = np.arange(0.75, 1.601, 0.025)

    best_price, best_p, best_er = start, 0.0, -np.inf
    base = base.copy(deep=True)  # безопасная база
    col = "price_bid_local"
    if col not in base.columns:
      base[col] = np.nan
    base.loc[:, col] = pd.to_numeric(base[col], errors="coerce").astype("float64")
    for k in ks:
      price = float(start * k)                # 1) считаем цену
      f = base.copy(deep=True)                # 2) берём копию
      idx = f.index[0]
      f.at[idx, "price_bid_local"] = price    # 3) записываем цену (dtype уже float64)
      f = _recompute_price_fields(f)
      if "bid_uplift_abs" in f.columns:
        f.at[idx, "bid_uplift_abs"] = float(price - start)
        p = predict_accept_prob(f, model, fns, calibrator)[0]
        er = float(price * p)
        if er > best_er:
          best_price, best_p, best_er = price, p, er
          best_price, best_p, best_er = price, p, er
    _log(f"Best: k={best_price/(start or 1.0):.3f}, price={best_price:.2f}, p={best_p:.4f}, ER={best_er:.2f}")

    return {"price": float(best_price), "p_accept": float(best_p), "er": float(best_er)}
