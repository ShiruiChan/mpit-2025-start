#!/usr/bin/env bash
set -euo pipefail
INP=${1:-final_test.csv}
OUT=${2:-predictions.csv}
python batch_predict.py "$INP" "$OUT"
echo "✅ Saved $OUT"