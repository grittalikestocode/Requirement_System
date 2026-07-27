from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.pdf_routes import router as pdf_router
from backend.app.api.ambiguity_routes import router as ambiguity_router
from backend.app.core.llm_service import llm_service
from backend.app.core.ml_service import ml_service

app = FastAPI(
    title="Unified Requirement Engineering API",
    version="1.0"
)

# ----------------------------------------------------
# Frontend configuration
# ----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve everything inside frontend/
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


# ----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Initializing Core Models...")

    try:
        llm_service.initialize()
        print("LLM Service ready.")
    except Exception as e:
        print(e)

    try:
        ml_service.initialize()
        print("ML Service ready.")
    except Exception as e:
        print(e)

app.include_router(pdf_router, prefix="/api", tags=["PDF Extractor"])
app.include_router(ambiguity_router, prefix="/api", tags=["Ambiguity Checker"])

@app.get("/health")
def health():
    return {"status": "ok"}