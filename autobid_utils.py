import math, numpy as np, pandas as pd
from joblib import load

feature_candidates = ['distance_in_meters', 'duration_in_seconds', 'pickup_in_meters', 'pickup_in_seconds', 'carmodel', 'carname', 'platform', 'price_start_local', 'price_bid_local', 'order_hour', 'order_dow', 'is_weekend', 'lag_tender_seconds', 'driver_tenure_days', 'bid_uplift_abs', 'bid_uplift_pct', 'centrality_proxy']

def load_model(path):
    return load(path)

def recommend_bid_for_row(row, model, price_grid_pct=None):
    if price_grid_pct is None:
        price_grid_pct = np.arange(0.8, 1.601, 0.025)
    start = row.get("price_start_local", np.nan)
    if not np.isfinite(start) or start <= 0:
        start = 100.0
    best = {"price": None, "p_accept": None, "er": -1}
    base = row.copy()
    for pct in price_grid_pct:
        price = float(start * pct)
        base_tmp = base.copy()
        base_tmp["price_bid_local"] = price
        base_tmp["bid_uplift_abs"] = price - start
        base_tmp["bid_uplift_pct"] = (price - start) / (start if start!=0 else 1.0)
        feat_vec = pd.DataFrame([base_tmp[feature_candidates]])
        p = float(model.predict_proba(feat_vec)[0,1])
        er = price * p
        if er > best["er"]:
            best = {"price": price, "p_accept": p, "er": er}
    return best

def build_demand_lookup(heatmap_df):
    if heatmap_df is None:
        return {(dow,h):1.0 for dow in range(7) for h in range(24)}
    hm_norm = (heatmap_df - heatmap_df.min().min()) / (heatmap_df.max().max() - heatmap_df.min().min() + 1e-9)
    lookup = {}
    for dow in hm_norm.index:
        for h in hm_norm.columns:
            lookup[(int(dow), int(h))] = float(hm_norm.loc[dow, h])
    return lookup

def compute_score_for_row(row, best_bid, demand_lookup=None, weights=None):
    if weights is None:
        weights = {"w_er":0.6,"w_demand":0.25,"w_central":0.2,"w_pickup_penalty":0.05}
    if demand_lookup is None:
        demand_lookup = {(d,h):0.5 for d in range(7) for h in range(24)}

    start = row.get("price_start_local", np.nan)
    pickup = row.get("pickup_in_meters", np.nan)
    hour   = int(row.get("order_hour", 12))
    dow    = int(row.get("order_dow", 3))

    if np.isfinite(start) and start>0:
        er_norm = (best_bid["er"] / start)
    else:
        er_norm = best_bid["er"] / 100.0

    demand = demand_lookup.get((dow, hour), 0.5)

    if np.isfinite(pickup):
        central = math.exp(-pickup/1000.0)
        pickup_penalty = pickup/1000.0
    else:
        central = 1.0; pickup_penalty = 0.0

    score = (weights["w_er"]*er_norm
            + weights["w_demand"]*demand
            + weights["w_central"]*central
            - weights["w_pickup_penalty"]*pickup_penalty)

    return {"score": float(score), "er_norm": float(er_norm), "demand": float(demand), "central": float(central)}
