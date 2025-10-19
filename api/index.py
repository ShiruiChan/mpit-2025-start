# Vercel ASGI entrypoint that dynamically loads the user's FastAPI app
import os, sys, pathlib, importlib

BASE = pathlib.Path(__file__).parent
BACKEND = BASE / "backend"
sys.path.insert(0, str(BACKEND))

# Try common module names first, then scan
candidates = ["app_fastapi", "main", "api", "app"]
app = None
for name in candidates:
    try:
        m = importlib.import_module(name)
        if hasattr(m, "app"):
            app = getattr(m, "app")
            break
    except Exception:
        pass

if app is None:
    # brute-force scan for any module exposing `app`
    for p in BACKEND.rglob("*.py"):
        modname = p.relative_to(BACKEND).with_suffix("").as_posix().replace("/", ".")
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "app"):
                app = getattr(m, "app")
                break
        except Exception:
            continue

if app is None:
    raise RuntimeError("FastAPI `app` not found in backend. Ensure your main module defines `app = FastAPI(...)`.")

# Attach CORS in case it's missing (safe no-op if already present)
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass