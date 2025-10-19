import os, sys, json, math, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# === Артефакты ===
MODEL_CBM   = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")
TRAIN_PATH  = os.environ.get("TRAIN_PATH", "train.csv")

# === Фичи ===
FEATURES = None
CAT_COLS = ["carmodel", "carname", "platform"]
CAT_IDX  = []

# === Вспомогательное: даты -> фичи === 
def _as_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    _as_dt(df, "order_timestamp"); _as_dt(df, "tender_timestamp"); _as_dt(df, "driver_reg_date")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"]  = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()
    df["lag_tender_seconds"] = df["lag_tender_seconds"].fillna(0).clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.days
    df["driver_tenure_days"] = df["driver_tenure_days"].fillna(0).clip(lower=0)
    df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"].replace(0, np.nan)
    df["centrality_proxy"] = -df.get("pickup_in_meters", pd.Series([np.nan]*len(df)))
    return df

# === Нормализация одной заявки ===
def ensure_all_features(row: pd.Series) -> pd.Series:
    """Заполнить недостающие фичи дефолтами + привести категории к строкам."""
    defaults_num = {
        "distance_in_meters": np.nan, "duration_in_seconds": np.nan,
        "pickup_in_meters": np.nan, "pickup_in_seconds": np.nan,
        "price_start_local": np.nan, "price_bid_local": np.nan,
        "order_hour": 12, "order_dow": 3, "is_weekend": 0,
        "lag_tender_seconds": 0.0, "driver_tenure_days": 0.0,
        "bid_uplift_abs": np.nan, "bid_uplift_pct": np.nan, "centrality_proxy": np.nan
    }
    defaults_cat = {"carmodel":"unknown","carname":"unknown","platform":"unknown"}

    for k,v in {**defaults_num, **defaults_cat}.items():
        if k not in row:
            row[k] = v

    if pd.isna(row["price_bid_local"]) and not pd.isna(row["price_start_local"]):
        row["price_bid_local"] = row["price_start_local"]
    if pd.isna(row["bid_uplift_abs"]) and not pd.isna(row["price_bid_local"]) and not pd.isna(row["price_start_local"]):
        row["bid_uplift_abs"] = row["price_bid_local"] - row["price_start_local"]
    if pd.isna(row["bid_uplift_pct"]) and not pd.isna(row["price_start_local"]) and row["price_start_local"]>0:
        row["bid_uplift_pct"] = row["bid_uplift_abs"]/row["price_start_local"]
    if pd.isna(row["centrality_proxy"]) and not pd.isna(row["pickup_in_meters"]):
        row["centrality_proxy"] = -row["pickup_in_meters"]
    row["is_weekend"] = int(int(row["order_dow"]) in (5,6))

    for c in CAT_COLS:
        if c not in row or pd.isna(row[c]):
            row[c] = "unknown"
        else:
            row[c] = str(row[c])

    return row

def build_predict_df(row: dict) -> pd.DataFrame:
    """Собрать DataFrame с нужными колонками и типами для CatBoost (cat -> str)."""
    feat_row = {}
    for f in FEATURES:
        v = row.get(f, np.nan)
        if f in CAT_COLS:
            if pd.isna(v): v = "unknown"
            v = str(v)
        feat_row[f] = v
    X = pd.DataFrame([feat_row], columns=FEATURES)
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("unknown")
    return X

# === Модель / артефакты ===
def load_or_train_model():
    global FEATURES, CAT_IDX
    if not os.path.exists(MODEL_CBM):
        os.system("python catboost_train.py")
    if not os.path.exists(MODEL_CBM):
        raise FileNotFoundError("Нет модели autobid_catboost.cbm. Убедись, что catboost_train.py отработал успешно.")

    model = CatBoostClassifier()
    model.load_model(MODEL_CBM)

    if not os.path.exists(FNAMES_JSON):
        raise FileNotFoundError("Нет файла cb_feature_names.json — обучи модель через catboost_train.py.")

    with open(FNAMES_JSON, "r", encoding="utf-8") as f:
        FEATURES = json.load(f)

    CAT_IDX = [FEATURES.index(c) for c in CAT_COLS if c in FEATURES]
    return model

# === Поиск «золота»: recommend_bid ===
def recommend_bid(row: pd.Series, model, price_grid_pct=np.arange(0.85, 1.401, 0.05)):
    """
    Перебираем цену вокруг старта и ищем максимум ER = price * P(accept).
    Возвращаем dict: {"price": ..., "p_accept": ..., "er": ...}
    """
    start = row.get("price_start_local", np.nan)
    if not np.isfinite(start) or start <= 0:
        start = 100.0

    best = {"price": None, "p_accept": None, "er": -1}
    base = ensure_all_features(row.copy())

    for pct in price_grid_pct:
        price = float(start * pct)
        r = base.copy()
        r["price_bid_local"] = price
        r["bid_uplift_abs"] = price - start
        r["bid_uplift_pct"] = (price - start) / (start if start!=0 else 1.0)

        X = build_predict_df(r)
        p = float(model.predict_proba(Pool(X, cat_features=CAT_IDX))[0,1])
        er = price * p

        if er > best["er"]:
            best = {"price": price, "p_accept": p, "er": er}

    return best

# === «Говорящая» фраза ===
def mood_phrase(p):
    if p >= 0.65: return "💬 Отличный шанс!"
    if p >= 0.50: return "💬 Нормально, средний риск."
    if p >= 0.35: return "💬 Осторожно: может не зайти."
    return "💬 Сомнительно — лучше снизить цену."

# === Таблица ===
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

# === Парсинг аргументов ===
def parse_args_payload():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        raw = sys.argv[1]
        try:
            return json.loads(raw)
        except Exception:
            import ast
            return ast.literal_eval(raw)
    # 2) Ключи
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--start",  type=float, dest="price_start_local")
    p.add_argument("--pickup", type=float, dest="pickup_in_meters")
    p.add_argument("--hour",   type=int,   dest="order_hour")
    p.add_argument("--dow",    type=int,   dest="order_dow")
    args, _ = p.parse_known_args()
    payload = {k:v for k,v in vars(args).items() if v is not None}
    if payload: return payload
    return {"price_start_local":300,"pickup_in_meters":400,"order_hour":18,"order_dow":4}

# === main ===
def main():
    model = load_or_train_model()

    payload = parse_args_payload()
    row = pd.Series(payload)
    if "price_bid_local" not in row:
        row["price_bid_local"] = row["price_start_local"]
    if "order_hour" not in row:
        row["order_hour"] = payload.get("order_timestamp_hour", payload.get("order_hour", 12))
    if "order_dow" not in row:
        row["order_dow"] = payload.get("order_timestamp_dow", payload.get("order_dow", 3))
    row["is_weekend"] = int(int(row["order_dow"]) in (5,6))

    best = recommend_bid(row, model)
    start = float(row["price_start_local"])
    uplift_pct = (best["price"]/start - 1.0)*100.0 if start>0 else 0.0

    print("\n🔹 Оптимальная рекомендация")
    headers = ["Старт ₽","Рекоменд. ₽","Аплифт %","P(accept)","ER ₽"]
    rows = [[f"{start:.2f}", f"{best['price']:.2f}", f"{uplift_pct:+.1f}",
             f"{best['p_accept']:.3f}", f"{best['er']:.2f}"]]
    print_table(headers, rows)
    print(mood_phrase(best["p_accept"]))

    tiers = [0.95, 1.00, 1.05]
    rows_out = []
    for t in tiers:
        price = start * t
        rr = row.copy()
        rr["price_bid_local"] = price
        rr["bid_uplift_abs"] = price - start
        rr["bid_uplift_pct"] = (price - start) / (start if start>0 else 1.0)
        X = build_predict_df(rr)
        p = float(model.predict_proba(Pool(X, cat_features=CAT_IDX))[0,1])
        er = price * p
        mode = "🟢 Conservative" if t<1.0 else ("⚪ Optimal" if t==1.0 else "🔴 Bold")
        rows_out.append([mode, f"{price:.2f}", f"{(t-1)*100:+.1f}", f"{p:.3f}", f"{er:.2f}"])

    print("\n🎯 Сравнение режимов")
    headers2 = ["Режим","Цена ₽","Δ%","P(accept)","ER ₽"]
    print_table(headers2, rows_out)

if __name__ == "__main__":
    main()
