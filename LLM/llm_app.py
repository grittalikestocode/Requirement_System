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

1. PRONOUN AMBIGUITY
   Pronouns (it, they, their, its, this, these) whose referents are unclear.
   Example: "The system shall notify the user when it detects an error." → What does 'it' refer to?

2. VAGUE / WEAK LANGUAGE
   Words like 'fast', 'easy', 'appropriate', 'sufficient', 'adequate', 'user-friendly',
   'simple', 'flexible', 'robust' that have no measurable definition.
   Example: "The system shall respond quickly." → How quickly?

3. MISSING SUBJECT / ACTOR
   It is unclear WHO or WHAT performs an action.
   Example: "Errors shall be logged." → Logged by whom? When?

4. COORDINATION AMBIGUITY
   'and' / 'or' that can be grouped in multiple ways.
   Example: "The system shall support PDF and Word or Excel exports." → (PDF and Word) or Excel? or PDF and (Word or Excel)?

5. SCOPE / QUANTIFIER AMBIGUITY
   Unclear scope: 'all', 'each', 'any', 'every', 'some', 'most'.
   Example: "Each user shall have access to all reports." → Does 'all' include restricted reports?

6. TEMPORAL AMBIGUITY
   Unclear timing: 'when', 'after', 'before', 'during', 'immediately' with no defined threshold.
   Example: "The system shall update the record after approval." → How long after?

7. CONDITIONAL AMBIGUITY
   Conditions that can be interpreted in multiple ways.
   Example: "If the user is logged in and idle or disconnected, the session shall expire."

8. NEGATION AMBIGUITY
   Negations that create unclear scope.
   Example: "The system shall not allow users without admin rights and supervisors to delete records."
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
══ TASK ══
Analyse the given requirement. Think step-by-step:
  Step 1 — Identify every word, phrase, pronoun, or structure that could be unclear.
  Step 2 — Decide which ambiguity type(s) from the taxonomy apply (if any).
  Step 3 — Classify as "ambiguous" or "unambiguous".
  Step 4 — Write one concise explanation sentence.

══ WHAT IS NOT AMBIGUOUS ══
- Passive voice ("shall be conditioned", "shall be qualified", "shall be provided",
  "shall be avoided") is STANDARD in requirements engineering — do NOT flag as missing_actor.
- Technical acronyms and domain terms (MPAS, ERMS, CITIS, GSTSIM, EME, DMS, FTP, SPI)
  are NOT ambiguous on their own.
- A requirement with multiple clauses is NOT automatically ambiguous.
- Words like "other", "related", "associated", "respective" are NOT ambiguous
  unless their referent is genuinely unclear.

══ WHAT IS AMBIGUOUS ══
- Pronouns (it, they, their, its, this, these) where the referent is genuinely unclear.
- Unmeasurable terms (fast, soon, adequate, sufficient, appropriate, user-friendly)
  with no defined metric or threshold.
- Coordination (and/or) that allows multiple groupings with different meanings.
- Missing actor ONLY when it is impossible to infer who performs the action
  even from context — passive voice alone is NOT enough.
- Temporal conditions with no defined time threshold.

══ LABELLED EXAMPLES (similar to the test requirement) ══
{examples_block}

══ RULES ══
- Output ONLY valid JSON — no markdown, no extra text before or after.
- "classification" must be exactly "ambiguous" or "unambiguous".
- "ambiguity_type" must be one of: pronoun, vague_language, missing_actor, coordination,
  scope_quantifier, temporal, conditional, negation, none.
- "explanation" must be one clear sentence stating the specific issue found
  or confirming why the requirement is clear.
- BEFORE outputting, ask yourself: "Can I point to one specific word or phrase
  that would cause two developers to implement this differently?"
  If yes → ambiguous. If no → unambiguous.

Requirement to classify: "{req}"
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
