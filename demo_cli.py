import os, sys, json, math, warnings, argparse
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from joblib import load, dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

MODEL_PATH = os.environ.get("AUTOBID_MODEL", "autobid_acceptance_model.joblib")
TRAIN_PATH = os.environ.get("TRAIN_PATH", "train.csv")
FORCE_REFIT = os.environ.get("FORCE_REFIT", "0") == "1"

# ---------- фичи ----------
FEATURES = [
    "distance_in_meters","duration_in_seconds",
    "pickup_in_meters","pickup_in_seconds",
    "carmodel","carname","platform",
    "price_start_local","price_bid_local",
    "order_hour","order_dow","is_weekend",
    "lag_tender_seconds","driver_tenure_days",
    "bid_uplift_abs","bid_uplift_pct","centrality_proxy"
]
NUM_COLS = [c for c in FEATURES if c not in ["carmodel","carname","platform"]]
CAT_COLS = ["carmodel","carname","platform"]

PREPROCESS = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler",  StandardScaler(with_mean=False))]), NUM_COLS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), CAT_COLS),
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# ---------- подготовка датасета ----------
def _as_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def _build_features_df(df):
    _as_dt(df, "order_timestamp"); _as_dt(df, "tender_timestamp"); _as_dt(df, "driver_reg_date")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"]  = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.days.clip(lower=0)
    df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"].replace(0, np.nan)
    df["centrality_proxy"] = -df.get("pickup_in_meters", pd.Series([np.nan]*len(df)))
    return df

def train_model(train_path: str):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Не найден {train_path}. Положите train.csv рядом со скриптом или задайте TRAIN_PATH.")
    df = pd.read_csv(train_path)
    if "is_done" not in df.columns:
        raise ValueError("В train.csv нет столбца is_done (ожидаются значения 'done'/'cancel').")
    df["is_done"] = df["is_done"].map({"done":1,"cancel":0}).astype(int)
    for c in list(df.columns):
        if c.startswith("Unnamed"):
            df = df.drop(columns=[c])

    df = _build_features_df(df)
    X = df[FEATURES].copy()
    y = df["is_done"].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([("preprocess", PREPROCESS),
                      ("gb", GradientBoostingClassifier(random_state=42))])
    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)
    return model

def load_or_fit():
    if FORCE_REFIT and os.path.exists(MODEL_PATH):
        try: os.remove(MODEL_PATH)
        except: pass
    try:
        return load(MODEL_PATH)
    except Exception as e:
        print(f"[!] Загрузка {MODEL_PATH} не удалась ({e}). Обучаю локально на {TRAIN_PATH} …")
        return train_model(TRAIN_PATH)

# ---------- инференс одной заявки ----------
def _ensure_all_features(row: pd.Series) -> pd.Series:
    """Гарантируем наличие всех ключей FEATURES в Series: дозаполняем разумными дефолтами."""
    defaults_num = {
        "distance_in_meters": np.nan, "duration_in_seconds": np.nan,
        "pickup_in_meters": np.nan, "pickup_in_seconds": np.nan,
        "price_start_local": np.nan, "price_bid_local": np.nan,
        "order_hour": 12, "order_dow": 3, "is_weekend": 0,
        "lag_tender_seconds": 0.0, "driver_tenure_days": 0.0,
        "bid_uplift_abs": np.nan, "bid_uplift_pct": np.nan,
        "centrality_proxy": np.nan
    }
    defaults_cat = {"carmodel":"unknown", "carname":"unknown", "platform":"unknown"}

    for k,v in {**defaults_num, **defaults_cat}.items():
        if k not in row:
            row[k] = v

    # Вычисляем производные, если надо
    if pd.isna(row["price_bid_local"]) and not pd.isna(row["price_start_local"]):
        row["price_bid_local"] = row["price_start_local"]
    if pd.isna(row["bid_uplift_abs"]) and not pd.isna(row["price_bid_local"]) and not pd.isna(row["price_start_local"]):
        row["bid_uplift_abs"] = row["price_bid_local"] - row["price_start_local"]
    if pd.isna(row["bid_uplift_pct"]) and not pd.isna(row["price_start_local"]) and row["price_start_local"]>0:
        row["bid_uplift_pct"] = row["bid_uplift_abs"]/row["price_start_local"]
    if pd.isna(row["centrality_proxy"]) and not pd.isna(row["pickup_in_meters"]):
        row["centrality_proxy"] = -row["pickup_in_meters"]
    if "is_weekend" not in row or pd.isna(row["is_weekend"]):
        row["is_weekend"] = int(int(row["order_dow"]) in (5,6))

    return row

def recommend_bid_for_row(row: pd.Series, model, price_grid_pct=np.arange(0.85, 1.401, 0.05)):
    start = row.get("price_start_local", np.nan)
    if not np.isfinite(start) or start <= 0:
        start = 100.0
    best = {"price": None, "p_accept": None, "er": -1}
    base = _ensure_all_features(row.copy())

    for pct in price_grid_pct:
        price = float(start * pct)
        base_tmp = base.copy()
        base_tmp["price_bid_local"] = price
        base_tmp["bid_uplift_abs"] = price - start
        base_tmp["bid_uplift_pct"] = (price - start) / (start if start!=0 else 1.0)

        # ВАЖНО: конструируем полноценную строку по FEATURE-списку (без KeyError)
        feat_row = {f: base_tmp.get(f, np.nan) for f in FEATURES}
        feat = pd.DataFrame([feat_row])
        p = float(model.predict_proba(feat)[0,1])
        er = price * p
        if er > best["er"]:
            best = {"price": price, "p_accept": p, "er": er}
    return best

def compute_score_for_row(row, best_bid, weights=None):
    if weights is None:
        weights = {"w_er":0.6,"w_demand":0.25,"w_central":0.2,"w_pickup_penalty":0.05}
    start  = row.get("price_start_local", np.nan)
    pickup = row.get("pickup_in_meters", np.nan)
    er_norm = (best_bid["er"] / start) if (isinstance(start,(int,float)) and start>0) else (best_bid["er"]/100.0)
    demand = 0.5
    central = math.exp(-pickup/1000.0) if isinstance(pickup,(int,float)) and pickup>=0 else 1.0
    pickup_penalty = (pickup/1000.0) if isinstance(pickup,(int,float)) and pickup>=0 else 0.0
    score = (weights["w_er"]*er_norm + weights["w_demand"]*demand +
             weights["w_central"]*central - weights["w_pickup_penalty"]*pickup_penalty)
    return {"score": float(score), "er_norm": float(er_norm)}

def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    def line(ch="-"): return "+" + "+".join(ch*(w+2) for w in widths) + "+"
    def fmt_row(cells): return "| " + " | ".join(str(c).ljust(widths[i]) for i,c in enumerate(cells)) + " |"
    print(line()); print(fmt_row(headers)); print(line("="))
    for r in rows: print(fmt_row(r))
    print(line())

def parse_args_payload():
    # 1) Если передали один позиционный JSON/py-словарь
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        raw = sys.argv[1]
        try:
            return json.loads(raw)
        except Exception:
            try:
                import ast
                return ast.literal_eval(raw)
            except Exception:
                pass
    # 2) Ключи
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--start",  type=float, dest="price_start_local")
    p.add_argument("--pickup", type=float, dest="pickup_in_meters")
    p.add_argument("--hour",   type=int,   dest="order_hour")
    p.add_argument("--dow",    type=int,   dest="order_dow")
    args, _ = p.parse_known_args()
    payload = {k:v for k,v in vars(args).items() if v is not None}
    if payload: return payload
    # 3) Дефолт
    return {"price_start_local":300,"pickup_in_meters":400,"order_hour":18,"order_dow":4}

def main():
    payload = parse_args_payload()
    row = pd.Series(payload)

    # безопасное заполнение дефолтов
    if "price_bid_local" not in row:
        row["price_bid_local"] = row["price_start_local"]
    if "order_hour" not in row:
        row["order_hour"] = payload.get("order_timestamp_hour", payload.get("order_hour", 12))
    if "order_dow" not in row:
        row["order_dow"] = payload.get("order_timestamp_dow", payload.get("order_dow", 3))
    row["is_weekend"] = int(int(row["order_dow"]) in (5,6))

    # модель
    model = load_or_fit()

    # --- основная рекомендация ---
    best = recommend_bid_for_row(row, model)
    sc   = compute_score_for_row(row, best)

    start = float(row["price_start_local"])
    uplift_pct = (best["price"]/start - 1.0)*100.0 if start>0 else 0.0

    headers = ["Старт ₽","Рекоменд. ₽","Аплифт %","P(accept)","ER ₽","Скор"]
    rows = [[f"{start:.2f}", f"{best['price']:.2f}", f"{uplift_pct:+.1f}",
             f"{best['p_accept']:.3f}", f"{best['er']:.2f}", f"{sc['score']:.3f}"]]
    print("\n🔹 Оптимальная рекомендация")
    print_table(headers, rows)

    # --- три сценария ---
    tiers = [0.95, 1.00, 1.05]  # conservative / optimal / bold
    rows_out = []
    for t in tiers:
        price = start * t
        rr = row.copy()
        rr["price_bid_local"] = price
        rr["bid_uplift_abs"] = price - start
        rr["bid_uplift_pct"] = (price - start) / (start if start>0 else 1.0)
        feat_row = {f: rr.get(f, np.nan) for f in FEATURES}
        p = float(model.predict_proba(pd.DataFrame([feat_row]))[0,1])
        er = price * p
        mode = "🟢 Conservative" if t<1.0 else ("⚪ Optimal" if t==1.0 else "🔴 Bold")
        rows_out.append([mode, f"{price:.2f}", f"{(t-1)*100:+.1f}",
                         f"{p:.3f}", f"{er:.2f}", f"{er/start:.3f}"])

    print("\n🎯 Сравнение режимов ценообразования")
    headers2 = ["Режим","Цена ₽","Δ%","P(accept)","ER ₽","ER/Start"]
    print_table(headers2, rows_out)

if __name__ == "__main__":
    main()