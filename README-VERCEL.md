# Autobid: Full Integration (Vercel-ready)

Этот репозиторий объединяет *весь фронтенд* (из вашего `public.zip`) и *весь бэкенд* (из `autobid_utils_robust.zip`) без вырезаний.
- **Frontend** — в корне (Vite/React/TS или ваш стек из архива).
- **Backend** — полностью сохранён внутри `api/backend/`; на Vercel запускается как Python Serverless Function через `api/index.py` (ASGI).

## Локально (dev)
- Фронтенд: `npm i && npm run dev` (порт 5173 по умолчанию)
- API локально можно поднять отдельно как у вас было (если есть `uvicorn` и `app_fastapi.py`). На Vercel локально серверлес поднимается командой `vercel dev`.

## Деплой на Vercel
1. В корне:
   ```bash
   npm i
   # если есть билд-скрипты фронта: npm run build
   ```
2. Войти и развернуть:
   ```bash
   npm i -g vercel
   vercel
   # либо vercel --prod
   ```
3. Все запросы на `/api/*` идут в `api/index.py`, который динамически импортирует ваш `app` из `api/backend/*`.
   - Требования для API в `api/requirements.txt` собираются из ваших `requirements.txt` + базовые зависимости (`fastapi`, `pydantic`, `uvicorn`, ...).
   - Если у вас модель/артефакты, положите их в `api/backend/` (они попадут в функцию). Для крупных файлов используйте Vercel KV/Blob/внешнее хранилище.

## Клиент API во фронте
Фронт настроен на относительный путь `/api` — это корректно и в dev (через прокси/переменные), и на Vercel.
Если у вас был `VITE_API_BASE=http://localhost:8080`, он заменён на `/api` для продакшена на Vercel.

## Полезно
- `vercel.json` содержит `rewrites` для API.
- В `api/index.py` автоматически добавляется CORS на все источники (сузьте в проде при желании).
- Если ваша FastAPI-приложение объявлено не как `app = FastAPI(...)`, а иначе — убедитесь, что модуль экспортирует переменную `app`.