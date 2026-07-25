"""
ML Segmenter using trained segment-start model
=============================================
"""

import re
import unicodedata
import argparse
import pickle
from typing import List
from pathlib import Path

from docx import Document
import fitz  # PyMuPDF
from pathlib import Path

# Base directory points to `backend/`
BASE_DIR = Path(__file__).resolve().parents[1]

# RAPR models folder: backend/RAPR/models/
RAPR_MODELS_DIR = BASE_DIR / "RAPR" / "models"

MODEL_PATH = RAPR_MODELS_DIR / "segmenter_model.pkl"


def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Run train_segmenter.py first")

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    return bundle["vectoriser"], bundle["model"]

def clean_pdf(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    suffix_pattern = r'(on|ons|ous|es|al|ble|ing|vity|ive|ic|ment|ces)'
    text = re.sub(r'([a-zA-Z])\s+' + suffix_pattern + r'\b', r'\1ti\2', text)
    text = re.sub(r'([a-zA-Z])\s+(ion|ions)\b', r'\1t\2', text)
    text = re.sub(r'([a-zA-Z])1([a-zA-Z])', r'\1tt\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_block(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def extract_pdf_blocks(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    blocks = []

    for page in doc:
        for b in page.get_text("blocks"):
            text = clean_block(b[4])

            if text:
                text = clean_pdf(text)
                blocks.append(text)

    return blocks


def extract_docx_blocks(docx_path: str) -> List[str]:
    document = Document(docx_path)
    blocks = []

    for paragraph in document.paragraphs:
        text = clean_block(paragraph.text)
        if text:
            blocks.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [clean_block(cell.text) for cell in row.cells]
            text = " | ".join(cell for cell in cells if cell)
            if text:
                blocks.append(text)

    return blocks


def extract_blocks(input_path: str) -> List[str]:
    suffix = Path(input_path).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_blocks(input_path)
    if suffix == ".docx":
        return extract_docx_blocks(input_path)

    raise ValueError("Unsupported file type. Use a .pdf or .docx file.")


# ============================================================
# 3. PREDICT STARTS
# ============================================================

def predict_starts(blocks, vectoriser, model):
    X = []

    for i, line in enumerate(blocks):
        prev = blocks[i - 1] if i > 0 else ""
        combined = prev + " || " + line
        X.append(combined)

    X_vec = vectoriser.transform(X)
    probs = model.predict_proba(X_vec)

    class_idx = list(model.classes_).index(1)
    return probs[:, class_idx]


def build_segments(blocks, probs, threshold=0.5):
    segments = []
    current = ""

    for line, prob in zip(blocks, probs):
        if prob >= threshold:
            if current:
                segments.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        segments.append(current.strip())

    return segments


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def segment_document(input_path: str, threshold=0.5):
    vectoriser, model = load_model()

    blocks = extract_blocks(input_path)
    probs = predict_starts(blocks, vectoriser, model)

    segments = build_segments(blocks, probs, threshold)

    return segments


def segment_pdf(pdf_path: str, threshold=0.5):
    return segment_document(pdf_path, threshold)


# ============================================================
# 6. SAVE
# ============================================================

def save_segments(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(s + "\n")


# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_txt")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    segments = segment_document(args.input_file, args.threshold)

    save_segments(segments, args.output_txt)

    print(f"Saved {len(segments)} segments")


if __name__ == "__main__":
    main()
