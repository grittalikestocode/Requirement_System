import json, re
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# >> uvicorn llm_app:app --reload --port 3000

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ollama config ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  

def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"]

# ── Load embedder + examples once at startup ─────────────────
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

examples_df = pd.read_csv("requirements_dataset.csv").rename(columns={
    "Context"    : "requirement",
    "DetectedAs" : "label"
})
example_embeddings = embedder.encode(
    examples_df["requirement"].tolist(),
    show_progress_bar=True
)

AMBIGUITY_TAXONOMY = """\
Types of ambiguity in software requirements:
1. PRONOUN AMBIGUITY — pronouns with unclear referents
2. VAGUE / WEAK LANGUAGE — unmeasurable terms (fast, adequate, sufficient)
3. MISSING SUBJECT / ACTOR — unclear who performs the action
4. COORDINATION AMBIGUITY — and/or allows multiple groupings
5. SCOPE / QUANTIFIER AMBIGUITY — all/each/any with undefined scope
6. TEMPORAL AMBIGUITY — timing with no defined threshold
7. CONDITIONAL AMBIGUITY — conditions with multiple interpretations
8. NEGATION AMBIGUITY — negations with unclear scope
"""

# ── Warmup ────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    try:
        embedder.encode(["Warmup"], show_progress_bar=False)
        call_ollama("Say OK")
        print("Models warmed up ✓")
    except Exception as e:
        print(f"Warmup failed (Ollama may not be running): {e}")

# ── Endpoint ──────────────────────────────────────────────────
class CheckRequest(BaseModel):
    requirement: str

@app.post("/check")
def check(body: CheckRequest):
    try:
        req = body.requirement

        # Retrieve similar examples
        query_emb  = embedder.encode([req])
        sim_scores = (query_emb @ example_embeddings.T)[0]
        top_indices = sim_scores.argsort()[-2:][::-1]
        similar = examples_df.iloc[top_indices]

        examples_block = ""
        for _, row in similar.iterrows():
            lbl = row["label"].strip().capitalize()
            examples_block += f"Requirement: {row['requirement']}\nClassification: {lbl}\n\n"

        prompt = f"""You are an expert software requirements engineer.

{AMBIGUITY_TAXONOMY}

NOTE: Passive voice is STANDARD — do NOT flag it.
Technical acronyms are NOT ambiguous on their own.

Examples:
{examples_block}

Requirement: "{req}"

Before classifying ask: can I point to one specific word that would cause
two developers to implement this differently? If yes → ambiguous. If no → unambiguous.

Return ONLY JSON:
{{"classification": "ambiguous or unambiguous", "ambiguity_type": "pronoun/vague_language/missing_actor/coordination/scope_quantifier/temporal/conditional/negation/none", "confidence": 0.0, "explanation": "one sentence"}}"""

        # ── Call Ollama instead of llm() ──────────────────────
        raw = call_ollama(prompt)

        # Parse JSON
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            parsed      = json.loads(match.group())
            lbl         = parsed.get("classification", "").strip().lower()
            verdict     = "ambiguous" if "ambiguous" in lbl and "unambiguous" not in lbl else "clear"
            confidence  = float(parsed.get("confidence", 0.0))
            explanation = parsed.get("explanation", "")
            ambiguity_type = parsed.get("ambiguity_type", "none")
        else:
            verdict        = "ambiguous" if "AMBIGUOUS" in raw.upper() else "clear"
            confidence     = 0.0
            explanation    = raw
            ambiguity_type = "none"

        return {
            "verdict"        : verdict,
            "confidence"     : confidence,
            "ambiguity_type" : ambiguity_type,
            "reasons"        : [explanation]
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise
