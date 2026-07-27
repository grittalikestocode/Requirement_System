from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from backend.app.core.pdf_extractor import extract_requirements_from_file
from backend.app.core.vector_service import vector_service

router = APIRouter()

class ExtractedSentence(BaseModel):
    index: int
    sentence: str
    is_requirement: bool
    confidence: float

class PipelineResponse(BaseModel):
    doc_id: str
    filename: str
    total_sentences: int
    requirement_count: int
    sentences: List[ExtractedSentence]

@router.post("/upload-pdf", response_model=PipelineResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        res = await extract_requirements_from_file(file)
        # 1. Generate unique document ID
        doc_id = f"{file.filename}_{uuid.uuid4().hex[:6]}"
        
        # 2. Collect extracted requirement texts
        req_texts = [s["sentence"] for s in res["sentences"] if s["is_requirement"]]
        
        # 3. Auto-index requirements into ChromaDB Vector DB
        vector_service.ingest_document(doc_id=doc_id, requirements=req_texts)

        res["doc_id"] = doc_id
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")