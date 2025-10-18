from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import io, os, json, math

import joblib
from catboost import CatBoostClassifier

MODEL_CBM = os.getenv("MODEL_CBM", "autobid_catboost.cbm")
FNAMES_JSON = os.getenv("FNAMES_JSON", "cb_feature_names.json")

class RecommendInput(BaseModel):
    price_start_local: float = Field(..., description="Базовая цена")
    pickup_in_meters: Optional[float] = Field(0, description="Дистанция подачи (м)")
    order_hour: Optional[int] = Field(12, ge=0, le=23)
    order_dow: Optional[int] = Field(3, ge=0, le=6)
    lat: Optional[float] = None
    lng: Optional[float] = None

class RecommendOutput(BaseModel):
    recommended_price_bid_local: float
    p_accept: float
    expected_revenue: float
    alt_up_5_price: float
    alt_up_5_p_accept: float
    alt_up_10_price: float
    alt_up_10_p_accept: float
    reason: str

_cached_model = None
def load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if os.path.exists("autobid_acceptance_model.joblib"):
        _cached_model = joblib.load("autobid_acceptance_model.joblib")
        return _cached_model
    if os.path.exists(MODEL_CBM):
        cb = CatBoostClassifier()
        cb.load_model(MODEL_CBM)
        _cached_model = cb
        return _cached_model
    raise FileNotFoundError("No acceptance model found (autobid_acceptance_model.joblib or CatBoost .cbm).")

FEATURES = ["price_start_local","pickup_in_meters","order_hour","order_dow","recommended_price_bid_local"]

def p_accept(model, row: Dict[str, Any], price: float) -> float:
    data = dict(row)
    data["recommended_price_bid_local"] = float(price)
    X = [[float(data.get(k, 0)) for k in FEATURES]]
    try:
        proba = model.predict_proba(X)[0][1]
    except Exception:
        p = model.predict(X)
        proba = float(p) if np.ndim(p) == 0 else float(p[0])
    return max(0.0, min(1.0, float(proba)))

def recommend_for_row(model, row: Dict[str, Any], grid=(0.8,1.6,0.02)):
    base = float(row.get("price_start_local", 300))
    lo, hi, step = grid
    grid_vals = np.arange(lo, hi + 1e-9, step)
    cand = []
    for g in grid_vals:
        price = round(base*g, 2)
        pa = p_accept(model, row, price)
        er = price*pa
        cand.append((price, pa, er))
    price, pa, er = max(cand, key=lambda t: t[2])
    up5, up10 = round(price*1.05,2), round(price*1.10,2)
    p5, p10 = p_accept(model, row, up5), p_accept(model, row, up10)
    reason = f"Максимум ER при цене {price} (P(accept)={pa:.3f}, ER={er:.2f}); альтернативы: +5%→P={p5:.3f}, +10%→P={p10:.3f}."
    return {
        "recommended_price_bid_local": price,
        "p_accept": round(pa, 6),
        "expected_revenue": round(er, 6),
        "alt_up_5_price": up5,
        "alt_up_5_p_accept": round(p5, 6),
        "alt_up_10_price": up10,
        "alt_up_10_p_accept": round(p10, 6),
        "reason": reason
    }

def haversine_meters(lat1, lng1, lat2, lng2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlng/2)**2
    return 2*R*math.asin(math.sqrt(a))

app = FastAPI(title="AutoBid API (combined)", version="1.0.0")
router = APIRouter()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendOutput)
def recommend(payload: RecommendInput):
    model = load_model()
    row = payload.dict()
    out = recommend_for_row(model, row)
    return out

@router.post("/batch_explain")
async def batch_explain(file: UploadFile = File(...), as_csv: int = Query(0, ge=0, le=1)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Пожалуйста, загрузите CSV (train.csv).")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать CSV: {e}")
    if df.empty:
        raise HTTPException(400, "CSV пустой.")
    model = load_model()
    rows = df.to_dict(orient="records")
    out = []
    for r in rows:
        rec = recommend_for_row(model, r)
        out.append({**({"order_id": r.get("order_id")} if "order_id" in r else {}), **rec})
    out_df = pd.DataFrame(out)
    if as_csv:
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="predictions_explained.csv"'})
    return JSONResponse(out)

@router.post("/recommend_by_coords")
async def recommend_by_coords(payload: Dict[str, Any]):
    driver_lat = payload.get("driver_lat")
    driver_lng = payload.get("driver_lng")
    orders = payload.get("orders", [])
    if driver_lat is None or driver_lng is None or not orders:
        raise HTTPException(400, "Нужно передать driver_lat, driver_lng и список orders.")
    order = orders[0]
    model = load_model()
    rec = recommend_for_row(model, order)
    if "lat" in order and "lng" in order:
        try:
            dist = haversine_meters(float(driver_lat), float(driver_lng), float(order["lat"]), float(order["lng"]))
            rec["distance_meters"] = round(dist, 1)
        except Exception:
            pass
    return {"order_id": order.get("order_id"), **rec}

app.include_router(router)