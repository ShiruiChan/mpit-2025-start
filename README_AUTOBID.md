# AutoBid (Drivee) — локальный прототип

## Установка
```bash
pip install -r requirements.txt
```

## Быстрый прогон (CLI)
```bash
python demo_cli.py '{"price_start_local":300,"pickup_in_meters":400,"order_hour":18,"order_dow":4}'
```

## API
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8080 --reload
# POST http://localhost:8080/recommend
# Body JSON:
# {"price_start_local": 300, "pickup_in_meters": 400, "order_hour": 18, "order_dow": 4}
```

## Пакетная инференция
```bash
python batch_predict.py input.csv predictions.csv
```

## Примечания
- Модель не использует `driver_rating` в вероятностной части.
- Оптимизация цены: ищем максимум ER = price × P(accept).
- Весовой скор можно менять в `autobid_utils.compute_score_for_row`.
