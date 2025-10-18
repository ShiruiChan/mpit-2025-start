
import os, sys, pandas as pd
from autobid_utils_robust import load_artifacts, recommend_bid_for_row, _log

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_predict.py input.csv output.csv")
        return
    inp, outp = sys.argv[1], sys.argv[2]
    _log(f"Reading input CSV: {inp}")
    df = pd.read_csv(inp)
    model, fns, calibrator, te_maps = load_artifacts("autobid_catboost.cbm", "cb_feature_names.json", "autobid_isotonic.joblib", "cb_te_maps.json")
    _log(f"Input shape: {df.shape}")
    rows = []
    n = len(df)
    use_tqdm = False
    # базовый итератор всегда один и тот же
    base_iter = df.iterrows()  # -> (idx, row)
    try:
        from tqdm import tqdm  # type: ignore
        it = tqdm(base_iter, total=n, ncols=80)
        use_tqdm = True
    except Exception:
        it = base_iter
    for i, (idx, row) in enumerate(it):
        try:
            best = recommend_bid_for_row(row, model, fns, calibrator, te_maps)
            rows.append({"price": best["price"], "p_accept": best["p_accept"], "er": best["er"]})
        except Exception as e:
            _log(f"Row {i}: ERROR {type(e).__name__}: {e}")
            rows.append({"price": float("nan"), "p_accept": float("nan"), "er": float("nan")})
        if not use_tqdm:
            # односложный прогресс в той же строке
            pct = (i + 1) * 100 // max(1, n)
            print(f"\rProgress: {i+1}/{n} ({pct}%)", end="", flush=True)
    if not use_tqdm:
        print()  # перенос после прогресса

if __name__ == "__main__":
    main()
