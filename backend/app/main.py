from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.pdf_routes import router as pdf_router
from backend.app.api.ambiguity_routes import router as ambiguity_router
from backend.app.core.llm_service import llm_service
from backend.app.core.ml_service import ml_service

app = FastAPI(title="Unified Requirement Engineering API", version="1.0")

@app.get("/", include_in_schema=False)
def root():
    # Automatically redirect anyone visiting the base URL to /docs
    return RedirectResponse(url="/docs")
    
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
        print(f"LLM Service initialization failed: {e}")

    try:
        ml_service.initialize()
        print("ML Service ready.")
    except Exception as e:
        print(f"ML Service initialization failed: {e}")

app.include_router(pdf_router, prefix="/api", tags=["PDF Extractor"])
app.include_router(ambiguity_router, prefix="/api", tags=["Ambiguity Checker"])

@app.get("/health")
def health():
    return {"status": "ok"}
