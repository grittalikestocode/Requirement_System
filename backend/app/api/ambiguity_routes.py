"""
Ambiguity Analysis Routes
-------------------------
API endpoints for requirement ambiguity checks, exposing endpoints to process 
text inputs using both LLM and traditional ML approaches for comparison.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.app.core.llm_service import llm_service
from backend.app.core.ml_service import ml_service

router = APIRouter()

class CheckRequest(BaseModel):
    requirement: str
    doc_id: Optional[str] = None  # <--- Added optional doc_id context

@router.post("/check-llm")
def check_llm(body: CheckRequest):
    try:
        return llm_service.analyze_ambiguity(body.requirement, doc_id=body.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-ml")
def check_ml(body: CheckRequest):
    try:
        return ml_service.predict(body.requirement)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))