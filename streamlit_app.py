import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from catboost import CatBoostClassifier
from catboost import Pool

# === настройки путей ===
MODEL_CBM   = os.environ.get("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.environ.get("FNAMES_JSON", "cb_feature_names.json")
TRAIN_PATH  = os.environ.get("TRAIN_PATH", "train.csv")

# === полезные константы ===
CAT_COLS = ["carmodel","carname","platform"]

# === Утилиты ===

def _as_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Быстрый фичеинж для датасета (для аналитики/теплокарт)."""
    _as_dt(df, "order_timestamp")
    _as_dt(df, "tender_timestamp")
    _as_dt(df, "driver_reg_date")
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dow"]  = df["order_timestamp"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5,6]).astype(int)
    df["lag_tender_seconds"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()
    df["lag_tender_seconds"] = df["lag_tender_seconds"].fillna(0).clip(lower=0)
    df["driver_tenure_days"] = (df["order_timestamp"] - df["driver_reg_date"]).dt.days
    df["driver_tenure_days"] = df["driver_tenure_days"].fillna(0).clip(lower=0)
    if "price_start_local" in df.columns and "price_bid_local" in df.columns:
        df["bid_uplift_abs"] = df["price_bid_local"] - df["price_start_local"]
        with np.errstate(divide='ignore', invalid='ignore'):
            df["bid_uplift_pct"] = df["bid_uplift_abs"] / df["price_start_local"]
    if "pickup_in_meters" in df.columns:
        df["centrality_proxy"] = -df["pickup_in_meters"]
    return df

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def ensure_row(row: dict, FEATURES: list) -> pd.DataFrame:
    """Собрать одну строку признаков под CatBoost (cat -> str, NaN -> 'unknown')."""
    start = row["price_start_local"]

    row.setdefault("price_bid_local", start)
    row.setdefault("order_dow", 3)
    row.setdefault("order_hour", 12)
    row.setdefault("is_weekend", int(row["order_dow"] in (5,6)))
    row.setdefault("distance_in_meters", np.nan)
    row.setdefault("duration_in_seconds", np.nan)
    row.setdefault("pickup_in_seconds", np.nan)
    row.setdefault("lag_tender_seconds", 0.0)
    row.setdefault("driver_tenure_days", 0.0)

    row["bid_uplift_abs"] = row["price_bid_local"] - start
    row["bid_uplift_pct"] = (row["bid_uplift_abs"] / start) if start and start>0 else 0.0
    row.setdefault("centrality_proxy", -row.get("pickup_in_meters", np.nan))

    for c in CAT_COLS:
        v = row.get(c, "unknown")
        if pd.isna(v):
            v = "unknown"
        row[c] = str(v)

    feat_row = {f: row.get(f, np.nan) for f in FEATURES}
    X = pd.DataFrame([feat_row], columns=FEATURES)
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("unknown")
    return X

# === cached loaders ===

@st.cache_data
def load_data(train_path: str):
    if not os.path.exists(train_path):
        st.error(f"Не найден {train_path}. Положи рядом с приложением.")
        st.stop()
    df = pd.read_csv(train_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = build_features(df)
    return df

@st.cache_resource
def load_model_and_features(model_path: str, fnames_json: str):
    if not os.path.exists(model_path):
        st.error(f"Не найден {model_path}. Сначала обучи модель: `python catboost_train.py`.")
        st.stop()
    if not os.path.exists(fnames_json):
        st.error(f"Не найден {fnames_json}. Он создаётся при обучении модели.")
        st.stop()

    model = CatBoostClassifier()
    model.load_model(model_path)
    with open(fnames_json, "r", encoding="utf-8") as f:
        FEATURES = json.load(f)
    cat_idx = [FEATURES.index(c) for c in CAT_COLS if c in FEATURES]
    return model, FEATURES, cat_idx

# === main app ===

def main():
    st.set_page_config(page_title="AutoBid — Drivee", layout="wide")
    st.title("AutoBid — умный авто-бидинг для Drivee")

    df = load_data(TRAIN_PATH)
    model, FEATURES, CAT_IDX = load_model_and_features(MODEL_CBM, FNAMES_JSON)

    col1, col2, col3, col4, col5 = st.columns([1.3,1.3,1,1,1.4])
    with col1:
        start = st.number_input("Стартовая цена, ₽", value=300.0, step=5.0, min_value=1.0)
    with col2:
        pickup = st.number_input("Расстояние до подачи, м", value=400.0, step=25.0, min_value=0.0)
    with col3:
        hour = st.number_input("Час (0–23)", min_value=0, max_value=23, value=18, step=1)
    with col4:
        dow = st.number_input("День (0=пн)", min_value=0, max_value=6, value=4, step=1)
    with col5:
        w_outskirts = st.slider("Охват окраин (fairness)", 0.0, 0.5, 0.15, 0.01,
                                help="Мягкая поддержка дальних подач, чтобы не «перегревать» центр")

    st.markdown("---")

    st.subheader("📈 Теплокарта спроса (кол-во заказов)")
    heat = df.groupby(["order_dow","order_hour"]).size().reset_index(name="count")
    heat_pivot = heat.pivot(index="order_dow", columns="order_hour", values="count").fillna(0)
    st.dataframe(heat_pivot.style.format("{:.0f}"), height=260, use_container_width=True)

    st.subheader("💰 Ожидаемая выручка vs цена")
    pct_grid = np.linspace(0.8, 1.4, 25)
    prices, probs, ers, ers_fair = [], [], [], []

    if "pickup_in_meters" in df.columns and df["pickup_in_meters"].notna().any():
        q50 = float(df["pickup_in_meters"].median())
    else:
        q50 = 500.0
    fairness_boost = sigmoid(((pickup - q50) / max(q50, 1.0)))  # >0 на окраинах, ~0 в центре

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
        p = float(model.predict_proba(Pool(X, cat_features=[FEATURES.index(c) for c in CAT_COLS if c in FEATURES]))[0,1])
        er = price * p
        er_fair = er * (1 + w_outskirts * fairness_boost)

        prices.append(price); probs.append(p); ers.append(er); ers_fair.append(er_fair)

    df_plot = pd.DataFrame({"price":prices, "P(accept)":probs, "ER":ers, "ER_fair":ers_fair})
    fig = px.line(df_plot, x="price", y=["ER","ER_fair"], markers=True)
    fig.update_layout(legend_title_text="", xaxis_title="Цена, ₽", yaxis_title="Ожидаемая выручка")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ER_fair — учитывает мягкую поддержку окраин: даже если центр выигрывает, окраины тоже получают шанс.")

    idx_opt  = int(np.argmax(df_plot["ER"].values))
    idx_fair = int(np.argmax(df_plot["ER_fair"].values))
    p_opt, p_fair = df_plot.loc[idx_opt], df_plot.loc[idx_fair]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Рекоменд. цена (классика)", f"{p_opt['price']:.0f} ₽",
                  help=f"P(accept)={probs[idx_opt]:.3f}, ER={p_opt['ER']:.2f}")
    with c2:
        st.metric("Рекоменд. цена (с fairness)", f"{p_fair['price']:.0f} ₽",
                  help=f"P(accept)={probs[idx_fair]:.3f}, ER_fair={p_fair['ER_fair']:.2f}")
    with c3:
        pa = probs[idx_fair]
        if pa >= 0.65: txt = "💬 Отличный шанс!"
        elif pa >= 0.50: txt = "💬 Нормально, средний риск."
        elif pa >= 0.35: txt = "💬 Осторожно: может не зайти."
        else: txt = "💬 Сомнительно — лучше снизить цену."
        st.success(txt)

    st.subheader("🎯 Сценарии: Conservative / Optimal / Bold")
    tiers = [0.95, 1.00, 1.05]
    rows_tbl = []
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
        p = float(model.predict_proba(Pool(X, cat_features=[FEATURES.index(c) for c in CAT_COLS if c in FEATURES]))[0,1])
        er = price * p
        er_f = er * (1 + w_outskirts * fairness_boost)
        name = "🟢 Conservative" if t<1 else ("⚪ Optimal" if t==1 else "🔴 Bold")
        rows_tbl.append({
            "Режим": name,
            "Цена ₽": f"{price:.0f}",
            "Δ%": f"{(t-1)*100:+.1f}",
            "P(accept)": f"{p:.3f}",
            "ER": f"{er:.2f}",
            "ER_fair": f"{er_f:.2f}"
        })
    st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True)

    st.markdown("---")

    with st.expander("🔍 Почему именно такая цена? (SHAP объяснение)"):
        st.markdown("Модель объясняет вклад каждого признака в вероятность принятия для выбранной конфигурации.")
        try:
            import shap
            import matplotlib.pyplot as plt
            X_one = ensure_row({
                "price_start_local": start,
                "price_bid_local": float(p_fair["price"]),
                "pickup_in_meters": pickup,
                "order_hour": int(hour),
                "order_dow": int(dow)
            }, FEATURES)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_one)
            if isinstance(shap_values, list):
                sv = shap_values[1]
                base_val = explainer.expected_value[1]
            else:
                sv = shap_values
                base_val = explainer.expected_value
            fig_shap = plt.figure(figsize=(8, 6))
            shap.plots._waterfall.waterfall_legacy(
                shap.Explanation(
                    values=sv[0],
                    base_values=base_val,
                    data=X_one.iloc[0],
                    feature_names=list(X_one.columns)
                ),
                max_display=14, show=False
            )
            st.pyplot(fig_shap, clear_figure=True)
            st.caption("🔹 Красные — понижают вероятность, 🔹 Синие — повышают. Чем длиннее столбик, тем сильнее влияние.")
        except Exception as e:
            st.warning(f"Не удалось отрисовать SHAP-график: {e}\n"
                       f"Убедись, что установлены пакеты: shap, matplotlib")

    st.info("Fairness-вес позволяет не игнорировать окраины: даже если центр даёт лучший ER, система мягко поддерживает дальние подачи.")

# === run ===
if __name__ == "__main__":
    main()