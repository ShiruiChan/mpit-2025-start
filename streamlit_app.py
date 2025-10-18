import os, json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
import plotly.express as px

MODEL_CBM = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")
TRAIN_PATH = os.environ.get("TRAIN_PATH", "train.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(TRAIN_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"] = df["order_timestamp"].dt.dayofweek
    return df

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_CBM)
    with open(FNAMES_JSON, "r", encoding="utf-8") as f:
        FEATURES = json.load(f)
    return model, FEATURES

def ensure_row(row: dict, FEATURES):
    # Дополним и вернём в нужном порядке
    start = row["price_start_local"]
    row.setdefault("price_bid_local", start)
    row.setdefault("is_weekend", int(row["order_dow"] in (5,6)))
    row["bid_uplift_abs"] = row["price_bid_local"] - start
    row["bid_uplift_pct"] = (row["bid_uplift_abs"] / start) if start>0 else 0.0
    row.setdefault("distance_in_meters", np.nan)
    row.setdefault("duration_in_seconds", np.nan)
    row.setdefault("pickup_in_seconds", np.nan)
    row.setdefault("lag_tender_seconds", 0.0)
    row.setdefault("driver_tenure_days", 0.0)
    row.setdefault("centrality_proxy", -row["pickup_in_meters"])
    row.setdefault("carmodel","unknown"); row.setdefault("carname","unknown"); row.setdefault("platform","unknown")
    return pd.DataFrame([{f: row.get(f, np.nan) for f in FEATURES}])

def sigmoid(x): return 1/(1+np.exp(-x))

def main():
    st.set_page_config(page_title="AutoBid — Drivee", layout="wide")
    st.title("AutoBid — умный авто-бидинг")

    df = load_data()
    if not os.path.exists(MODEL_CBM):
        st.warning("Модель не найдена. Запусти `python catboost_train.py` в консоли, затем обнови страницу.")
        st.stop()
    model, FEATURES = load_model()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start = st.number_input("Стартовая цена, ₽", value=300.0, step=5.0)
    with col2:
        pickup = st.number_input("Расстояние до подачи, м", value=400.0, step=50.0)
    with col3:
        hour = st.number_input("Час заказа", min_value=0, max_value=23, value=18, step=1)
    with col4:
        dow = st.number_input("День недели (0=пн)", min_value=0, max_value=6, value=4, step=1)

    w_outskirts = st.slider("Охват окраин (fairness)", 0.0, 0.5, 0.15, 0.01,
                            help="Повышать шанс/цену для дальних подач, чтобы не перегревать центр")

    # Теплокарта спроса (по количеству заказов)
    heat = df.groupby(["order_dow","order_hour"]).size().reset_index(name="count")
    heat_pivot = heat.pivot(index="order_dow", columns="order_hour", values="count").fillna(0)
    st.subheader("Теплокарта спроса (заказы по часам/дням)")
    st.dataframe(heat_pivot, height=240)

    # ER-кривая
    pct_grid = np.linspace(0.8, 1.4, 25)
    prices, probs, ers, ers_fair = [], [], [], []

    q50 = df["pickup_in_meters"].dropna().quantile(0.5) if "pickup_in_meters" in df.columns else 500.0
    fairness_boost = sigmoid(((pickup - q50) / max(q50,1.0)))  # >0, если дальше, ~0 — если ближе центра

    for t in pct_grid:
        price = start * t
        row = {
            "price_start_local": start,
            "price_bid_local": price,
            "pickup_in_meters": pickup,
            "order_hour": int(hour),
            "order_dow": int(dow)
        }
        X = ensure_row(row, FEATURES)
        p = float(model.predict_proba(X)[0,1])
        er = price * p
        er_fair = er * (1 + w_outskirts * fairness_boost)
        prices.append(price); probs.append(p); ers.append(er); ers_fair.append(er_fair)

    df_plot = pd.DataFrame({"price":prices, "P(accept)":probs, "ER":ers, "ER_fair":ers_fair})
    st.subheader("Ожидаемая выручка vs цена")
    st.plotly_chart(px.line(df_plot, x="price", y=["ER","ER_fair"], markers=True), use_container_width=True)
    st.caption("ER_fair — с мягкой поддержкой окраин (fairness)")

    # Рекомендации
    idx_opt = int(np.argmax(df_plot["ER"].values))
    idx_fair = int(np.argmax(df_plot["ER_fair"].values))
    p_opt, p_fair = df_plot.loc[idx_opt], df_plot.loc[idx_fair]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Рекоменд. цена (классика)", f"{p_opt['price']:.0f} ₽", help=f"P(accept)={probs[idx_opt]:.3f}, ER={p_opt['ER']:.2f}")
    with c2:
        st.metric("Рекоменд. цена (с fairness)", f"{p_fair['price']:.0f} ₽", help=f"P(accept)={probs[idx_fair]:.3f}, ER_fair={p_fair['ER_fair']:.2f}")

    # 3 сценария
    st.subheader("Сценарии: Conservative / Optimal / Bold")
    tiers = [0.95, 1.0, 1.05]
    tbl = []
    for t in tiers:
        price = start * t
        row = {
            "price_start_local": start,
            "price_bid_local": price,
            "pickup_in_meters": pickup,
            "order_hour": int(hour),
            "order_dow": int(dow)
        }
        X = ensure_row(row, FEATURES)
        p = float(model.predict_proba(X)[0,1])
        er = price * p
        er_f = er * (1 + w_outskirts*fairness_boost)
        name = "🟢 Conservative" if t<1 else ("⚪ Optimal" if t==1 else "🔴 Bold")
        tbl.append({"Режим":name, "Цена ₽":f"{price:.0f}", "Δ%":f"{(t-1)*100:+.1f}",
                    "P(accept)":f"{p:.3f}", "ER":f"{er:.2f}", "ER_fair":f"{er_f:.2f}"})
    st.dataframe(pd.DataFrame(tbl), use_container_width=True)

    st.info("Fairness: даже если в центре шанс выше, ER_fair мягко подталкивает систему поддерживать окраины.")

if __name__ == "__main__":
    main()
