from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from backend.app.core.pdf_extractor import extract_requirements_from_file

router = APIRouter()

class ExtractedSentence(BaseModel):
    index: int
    sentence: str
    is_requirement: bool
    confidence: float

class PipelineResponse(BaseModel):
    filename: str
    total_sentences: int
    requirement_count: int
    sentences: List[ExtractedSentence]

@router.post("/upload-pdf", response_model=PipelineResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        res = await extract_requirements_from_file(file)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")