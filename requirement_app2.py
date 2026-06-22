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
model_path = r"C:\Users\HP PAVILION\Downloads\mistral-7b-instruct-v0.2.Q4_0.gguf"
llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="mistral",
    gpu_layers=0,
    context_length=2048,  # Reduced for faster loading
    threads=os.cpu_count(),
    batch_size=256,  # Reduced batch size
)

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def warmup_models():
    try:
        _ = embedder.encode(["Warmup query"], show_progress_bar=False)
        warmup_prompt = """<|im_start|>system
You are a warmup assistant.<|im_end|>
<|im_start|>user
Warm up the model.<|im_end|>
<|im_start|>assistant
"""
        llm(warmup_prompt, max_new_tokens=1, temperature=0.0, stop=["<|im_end|>"])
    except Exception:
        pass

@app.on_event("startup")
def on_startup():
    warmup_models()

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
        top_indices = sim_scores.argsort()[-2:][::-1]
        similar = examples_df.iloc[top_indices]

        # 2. Build prompt (copy from your existing code)
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
            # "ambiguous_phrases": [],        # your model doesn't return these yet — leave empty
            "reasons": [explanation]
            # "suggested_rewrite": ""         # optional, leave blank for now
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())   # this prints the REAL error in your terminal
        raise