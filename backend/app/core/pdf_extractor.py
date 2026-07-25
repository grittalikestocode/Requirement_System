import sys
import os
import tempfile
from pathlib import Path

# Setup path for RAPR module
RAPR_DIR = Path(__file__).resolve().parents[2] / "RAPR"

if str(RAPR_DIR) not in sys.path:
    sys.path.insert(0, str(RAPR_DIR))

try:
    from backend.RAPR.segmenter import segment_document
    from backend.RAPR.classifier import classify_document
except ImportError as e:
    print(f"Warning: RAPR modules failed to load: {e}")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

async def extract_requirements_from_file(file):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and DOCX files are supported.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments = segment_document(tmp_path, threshold=0.5)
        if not segments:
            raise ValueError("No segments could be extracted from the document.")

        combined_text = "\n\n".join(segments)
        classified = classify_document(text=combined_text, threshold=0.5, include_non_req=True)

        sentences = []
        for i, item in enumerate(classified):
            sentences.append({
                "index": i + 1,
                "sentence": item.text,
                "is_requirement": item.is_requirement,
                "confidence": round(float(item.requirement_probability), 3),
            })

        requirement_count = sum(1 for s in sentences if s["is_requirement"])
        return {
            "filename": file.filename,
            "total_sentences": len(sentences),
            "requirement_count": requirement_count,
            "sentences": sentences
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)