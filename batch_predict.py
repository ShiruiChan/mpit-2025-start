"""
make_predictions.py
Запуск:
    python make_predictions.py input.csv output.csv \
        --model autobid_catboost.cbm \
        --features cb_feature_names.json \
        --calibrator autobid_isotonic.joblib \
        --te-maps cb_te_maps.json

На выходе: CSV с колонками
order_id, recommended_price_bid_local, p_accept, expected_revenue
"""

import os
import sys
import argparse
import pandas as pd

from autobid_utils_robust import load_artifacts, recommend_bid_for_row, _log


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="make_predictions.py", description="Batch recommend bids for Drivee case")
    p.add_argument("input", help="Входной CSV с заявками (минимум order_id и фичи, необходимые recommend_bid_for_row)")
    p.add_argument("output", help="Путь для сохранения результата predictions.csv")
    p.add_argument("--model", default="autobid_catboost.cbm", help="Путь к .cbm модели CatBoost")
    p.add_argument("--features", default="cb_feature_names.json", help="JSON с порядком/названиями фичей модели")
    p.add_argument("--calibrator", default="autobid_isotonic.joblib", help="Isotonic/Platt калибровщик вероятностей")
    p.add_argument("--te-maps", default="cb_te_maps.json", help="JSON с картами target-encoding/категориальных маппингов")
    p.add_argument("--debug-first", action="store_true", help="Вывести дебаг по первой обработанной строке")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    inp, outp = args.input, args.output
    model_path = args.model
    features_path = args.features
    calibrator_path = args.calibrator
    te_maps_path = args.te_maps

    if not os.path.exists(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    _log(f"Reading input CSV: {inp}")
    df = pd.read_csv(inp)
    _log(f"Input shape: {df.shape}")

    model, fns, calibrator, te_maps = load_artifacts(
        model_path,
        features_path,
        calibrator_path,
        te_maps_path,
    )

    n = len(df)
    base_iter = df.iterrows()
    use_tqdm = False
    try:
        from tqdm import tqdm
        it = tqdm(base_iter, total=n, ncols=80, leave=False)
        use_tqdm = True
    except Exception:
        it = base_iter

    rows = []
    debug_printed = False

    for i, (idx, row) in enumerate(it):
        try:
            best = recommend_bid_for_row(row, model, fns, calibrator, te_maps)

            price = float(best.get("price", float("nan")))
            p = float(best.get("p_accept", float("nan")))
            er = float(best.get("er", float("nan")))

            rows.append({"price": price, "p_accept": p, "er": er})

            if args.debug_first and not debug_printed:
                print(f"[debug] first row: price={price:.2f}, p={p:.4f}, er={er:.2f}")
                debug_printed = True

        except Exception as e:
            _log(f"Row {i}: ERROR {type(e).__name__}: {e}")
            rows.append({"price": float("nan"), "p_accept": float("nan"), "er": float("nan")})

        if not use_tqdm:
            pct = (i + 1) * 100 // max(1, n)
            print(f"\rProgress: {i+1}/{n} ({pct}%)", end="", flush=True)

    if not use_tqdm:
        print()

    out_df = pd.DataFrame({
        "order_id": df["order_id"] if "order_id" in df.columns else df.index,
        "recommended_price_bid_local": [r["price"] for r in rows],
        "p_accept": [r["p_accept"] for r in rows],
        "expected_revenue": [r["er"] for r in rows],
    })

    out_df = out_df[["order_id", "recommended_price_bid_local", "p_accept", "expected_revenue"]]

    out_dir = os.path.dirname(os.path.abspath(outp)) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(outp, index=False)

    _log(f"Saved predictions to: {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
