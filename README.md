# AutoBid — оптимальная ставка для заказа (FastAPI + CatBoost)

**Тизер.** AutoBid — сервис, который подбирает оптимальную цену для заявки, максимизируя ожидаемую выручку `ER = price × P(accept)`. 
За вероятность принятия отвечает модель CatBoost, доступна инференция через REST API, CLI/батч и интерактивная витрина (Streamlit).

---

## 🚀 Как запустить локально

### 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Быстрый smoke‑тест (CLI)
```bash
python demo_cli.py '{"price_start_local":300,"pickup_in_meters":400,"order_hour":18,"order_dow":4}'
```

### 3) REST API (FastAPI + Uvicorn)
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8080
# Тест:
curl -s -X POST http://localhost:8080/recommend -H "Content-Type: application/json"      -d '{"price_start_local":300,"pickup_in_meters":400,"order_hour":18,"order_dow":4}' | jq .
```
Открыть документацию: http://localhost:8080/docs

### 4) Батч‑инференция (для финального тестового набора)
```bash
python batch_predict.py path/to/final_test.csv predictions.csv
```

### 5) Streamlit демо (необязательно)
```bash
streamlit run streamlit_app.py
```

> ❗️Если требуется переобучить модель на своём датасете:
```bash
python catboost_train.py
```

---

## 🧩 Структура репозитория (по загруженному архиву)
```
├── app_fastapi.py
├── README_AUTOBID.md
├── train.csv
├── catboost_info/catboost_training.json
├── catboost_info/learn/events.out.tfevents
├── catboost_info/learn_error.tsv
├── catboost_info/time_left.tsv
├── __pycache__/autobid_utils.cpython-311.pyc
├── Аналитика/accept_by_hour.csv
├── Аналитика/accept_by_uplift.csv
├── Аналитика/accept_by_zone.csv
├── Аналитика/demo_recs.csv
├── Аналитика/feature_importances.csv
├── Аналитика/heatmap_hour_dow.png
├── Аналитика/hist_pickup.png
├── Аналитика/recommended_uplift_sample.csv
├── Аналитика/zone_hour_counts.csv
├── autobid_catboost.cbm
├── cb_feature_names.json
├── streamlit_app.py
├── requirements.txt
├── demo_cli.py
├── shap_explain.py
├── catboost_train.py
├── autobid_acceptance_model.joblib
├── batch_predict.py
├── autobid_utils.py
```

Ключевые файлы:
- `app_fastapi.py` — REST API, эндпоинт **POST /recommend**;
- `batch_predict.py` — пакетная инференция `input.csv → predictions.csv`;
- `demo_cli.py` — проверка модели и оптимума в консоли;
- `catboost_train.py` — обучение CatBoost, сохранение `.cbm` и `cb_feature_names.json`;
- `shap_explain.py` — фиче‑импортанс и графики SHAP;
- `streamlit_app.py` — интерактивная витрина;
- `autobid_catboost.cbm`, `cb_feature_names.json`, `autobid_acceptance_model.joblib` — артефакты модели;
- `requirements.txt` — зависимости.

Переменные окружения (не обязательны, есть дефолты):
- `MODEL_CBM` (по умолчанию `autobid_catboost.cbm`)
- `FNAMES_JSON` (по умолчанию `cb_feature_names.json`)
- `TRAIN_PATH` (по умолчанию `train.csv`)

---

## 📦 Формат итогового файла `predictions.csv`
Скрипт `batch_predict.py` формирует колонки:
- `order_id` — идентификатор заказа (если нет в входном файле — будет 0..N-1);
- `recommended_price_bid_local` — рекомендованная цена;
- `p_accept` — вероятность принятия;
- `expected_revenue` — ожидаемая выручка `ER`.

> Если организаторы согласовали другой формат — переименуйте/добавьте нужные столбцы в `batch_predict.py` и перезапустите.

---

## 🧪 CI (GitHub Actions)
В `.github/workflows/ci.yml` добавлен пайплайн, который:
1) Устанавливает зависимости;
2) Делает smoke‑тест CLI;
3) Поднимает API и проверяет `POST /recommend`;
4) Прогоняет батч на мини‑семпле и сохраняет `predictions.csv` как артефакт сборки.

---

## 🐳 Docker (опционально)
Собрать и запустить API в контейнере:
```bash
docker build -t autobid-api .
docker run --rm -p 8080:8080 autobid-api
```

---

## 📑 Что должно быть в репозитории к стоп‑коду
1. `README.md` (этот файл): тизер, инструкции по развёртыванию, структура.
2. **Презентация**: `presentation.pdf` или `presentation.pptx` (см. `PRESENTATION_OUTLINE.md`).
3. **Скринкаст** (2–3 мин): ссылка в `SCREencast_SCRIPT.md` (не забудьте открыть доступ «по ссылке»).
4. **Демо**: ссылка на работающее демо (например, Streamlit/ FastAPI), добавить в README.
5. **predictions.csv** — в корне репозитория (см. раздел «Батч‑инференция» или `scripts/run_final.sh`).

> Важно: коммиты после стоп‑кода — дисквалификация. Сверьтесь с `CHECKLIST.md` перед финальной загрузкой.

---

## 🧠 Полезные команды
```bash
# 1) Быстрый прогон батча
python batch_predict.py final_test.csv predictions.csv

# 2) SHAP-графики важности признаков
python shap_explain.py  # сохранит shap_summary.png и shap_waterfall.png

# 3) Проверка API локально
uvicorn app_fastapi:app --host 0.0.0.0 --port 8080
```

**Контакты и поддержка:** добавьте e-mail/телеграм для связи.

