# app.py
import json, re
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM

# >> venv\Scripts\activate
# >> uvicorn requirement_app:app --reload --port 3000

app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Load your model once at startup ──────────────────────────────
model_path = r"C:\Users\HP PAVILION\Downloads\mistral-7b-instruct-v0.2.Q3_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="mistral",
    gpu_layers=0,
    context_length=2048,  # Reduced for faster loading
    threads=os.cpu_count(),
    batch_size=256,  # Reduced batch size
)

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load dataset as the example pool
# examples_df = prompt_dev_df.copy()
examples_df = pd.read_csv("requirements_dataset.csv")
# If columns are "Context" and "DetectedAs" rename first
examples_df = examples_df.rename(columns={
    "Context"    : "requirement",
    "DetectedAs" : "label"
})
example_embeddings = embedder.encode(
    examples_df["requirement"].tolist(),
    show_progress_bar=True
)
# (paste your existing llm + embedder + examples_df setup here)
# e.g.
# from llama_cpp import Llama
# llm = Llama(model_path="your_model.gguf", ...)
# embedder = SentenceTransformer(...)
# examples_df = pd.read_csv("your_examples.csv")
# example_embeddings = embedder.encode(examples_df["requirement"].tolist(), batch_size=32)

class CheckRequest(BaseModel):
    requirement: str

@app.post("/check")
def check(body: CheckRequest):
    try: 
        req = body.requirement

        # 1. Embed the single query
        query_emb = embedder.encode([req])
        sim_scores = (query_emb @ example_embeddings.T)[0]
        top_indices = sim_scores.argsort()[-5:][::-1]
        similar = examples_df.iloc[top_indices]

        # 2. Build prompt (copy from your existing code)
        examples_block = ""
        for _, row in similar.iterrows():
            lbl = row["label"].strip().capitalize()
            examples_block += f"Requirement: {row['requirement']}\nClassification: {lbl}\n\n"

        prompt = f"""You are an expert in software requirement analysis.

    Examples:
    {examples_block}

    Analyse this requirement step by step:
    "{req}"

    Step 1 - Pronouns: List any pronouns. Is referent 100% clear?
    Step 2 - Vague terms: Any unmeasurable terms?
    Step 3 - Interpretations: Could two engineers build different things?
    Step 4 - Verdict: AMBIGUOUS or UNAMBIGUOUS.

    Return ONLY JSON:
    {{"classification": "ambiguous or unambiguous", "confidence": 0.0, "explanation": "one sentence"}}"""

        full_prompt = f"""<|im_start|>system
    You are an expert in software requirement analysis.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """

        # 3. Run model
        out = llm(full_prompt, max_new_tokens=100, temperature=0.0,
                stop=["<|im_end|>"])
        raw = out.strip()

        # 4. Parse JSON
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            lbl = parsed.get("classification", "").strip().lower()
            verdict = "ambiguous" if "ambiguous" in lbl and "unambiguous" not in lbl else "clear"
            confidence = float(parsed.get("confidence", 0.0))
            explanation = parsed.get("explanation", "")
        else:
            verdict = "ambiguous" if "AMBIGUOUS" in raw.upper() else "clear"
            confidence = 0.0
            explanation = raw

        # 5. Return in the format the HTML already expects
        return {
            "verdict": verdict,
            "confidence": confidence,
            "ambiguous_phrases": [],        # your model doesn't return these yet — leave empty
            "reasons": [explanation],
            "suggested_rewrite": ""         # optional, leave blank for now
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())   # this prints the REAL error in your terminal
        raise