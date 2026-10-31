import re
import json
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# Resolves to `backend/`
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "requirements_dataset.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_EXAMPLES = 5            # few-shot examples per prompt (v1 used 2)
AMBIG_LABEL = "ambiguous"
UNAMBIG_LABEL = "unambiguous"

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

#-----EXAMPLE BUILDER ----#
# ── v2: keyword heuristics used to write explanations for few-shot examples 
_PRONOUN_RE   = re.compile(r'\b(it|they|their|its|this|these|those|them)\b', re.I)
_VAGUE_RE     = re.compile(r'\b(fast|quick|quickly|easy|easily|appropriate|sufficient|adequate|user.friendly|simple|flexible|robust|efficient|reasonable|timely|high.quality|good|bad|large|small|many|few)\b', re.I)
_COORD_RE     = re.compile(r'\b(and|or)\b.*\b(and|or)\b', re.I)
_TEMPORAL_RE  = re.compile(r'\b(immediately|soon|quickly|eventually|when|before|after|during|within|upon)\b', re.I)
_SCOPE_RE     = re.compile(r'\b(all|every|each|any|some|most|several|multiple)\b', re.I)
_NEGATION_RE  = re.compile(r'\b(not|without|except|unless|no)\b', re.I)
# Used only inside few-shot examples. The model must NOT return this as its own answer.
CANNED_CLEAR = (
    "The requirement is specific and measurable; "
    "a developer can implement exactly one behaviour."
)

def infer_ambiguity_type(text: str) -> str:
    """Return a descriptive explanation of why a requirement is ambiguous."""
    reasons = []
    if _PRONOUN_RE.search(text):
        pronoun = _PRONOUN_RE.search(text).group(0)
        reasons.append(f"pronoun '{pronoun}' has an unclear referent")
    if _VAGUE_RE.search(text):
        word = _VAGUE_RE.search(text).group(0)
        reasons.append(f"vague/unmeasurable term '{word}' is used")
    if _COORD_RE.search(text):
        reasons.append("coordination with 'and/or' allows multiple groupings")
    if _TEMPORAL_RE.search(text) and not _VAGUE_RE.search(text):
        reasons.append("temporal condition has no defined threshold")
    if _SCOPE_RE.search(text):
        word = _SCOPE_RE.search(text).group(0)
        reasons.append(f"quantifier '{word}' is undefined in scope")
    if _NEGATION_RE.search(text) and not reasons:
        reasons.append("negation creates unclear scope")

    if reasons:
        return "Ambiguous because " + "; ".join(reasons) + "."
    else:
        return "The requirement is ambiguous due to vague or imprecise language that admits multiple interpretations."

def build_examples(examples_df: pd.DataFrame) -> str:
    """Build the few-shot block (requirement + classification + explanation)."""
    parts = []
    for _, row in examples_df.iterrows():
        explanation = (
            infer_ambiguity_type(row["requirement"])
            if row["label"] == AMBIG_LABEL else CANNED_CLEAR
        )
        parts.append(
            f"Requirement: {row['requirement']}\n"
            f"Classification: {row['label']}\n"
            f"Explanation: {explanation}"
        )
    return "\n\n".join(parts)

#---PROMPT BUILDER ---#
def build_prompt(query: str, examples_text: str) -> str:
    return f"""\
You are an expert software requirements engineer specialising in ambiguity analysis.

══ AMBIGUITY TAXONOMY ══
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
{examples_text}

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

══ OUTPUT FORMAT ══
{{"classification": "ambiguous or unambiguous", "ambiguity_type": "type or none", "explanation": "one sentence"}}

══ REQUIREMENT TO CLASSIFY ══
"{query}"
"""

def extract_json(raw: str) -> dict | None:
    """Try multiple strategies to parse JSON from LLM output."""
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except Exception:
        pass

    # Strategy 2: extract first {...} block
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Strategy 3: look for JSON inside markdown code fences
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    return None

def keyword_fallback(raw: str) -> str | None:
    """Last resort when JSON parsing fails: look for the label in the raw text."""
    raw_lower = raw.lower()
    if '"unambiguous"' in raw_lower or 'unambiguous' in raw_lower:
        return UNAMBIG_LABEL
    if '"ambiguous"' in raw_lower or 'ambiguous' in raw_lower:
        return AMBIG_LABEL
    return None

class LLMService:
    def __init__(self):
        self.embedder = None
        self.examples_df = None
        self.example_embeddings = None

    def initialize(self, examples_df: pd.DataFrame | None = None):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.embedder.encode(["Warmup"], show_progress_bar=False)

        if examples_df is None:
            examples_df = pd.read_csv(DATASET_PATH)

        df = examples_df.rename(columns={
            "Context": "requirement",
            "DetectedAs": "label"
        }).copy()
        required_columns = {"requirement", "label"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Example data is missing required columns: {sorted(missing_columns)}"
            )
        # v2 cleaning: drop blanks, trim text, lowercase labels
        df = df.dropna(subset=["requirement", "label"])
        df["requirement"] = df["requirement"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df["label"] = df["label"].replace({"clear": UNAMBIG_LABEL, "not ambiguous": UNAMBIG_LABEL})
        present = set(df["label"])
        if present != {AMBIG_LABEL, UNAMBIG_LABEL}:
            raise ValueError(
        f"Examples must contain both '{AMBIG_LABEL}' and '{UNAMBIG_LABEL}' labels; found {sorted(present)}"
            )   
        self.examples_df = df.reset_index(drop=True)


        self.example_embeddings = self.embedder.encode(
            self.examples_df["requirement"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def call_ollama(self, prompt: str) -> str:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def _explain_clear(self,req: str) -> str:
        """Second, short LLM call: write a real explanation for a requirement judged clear."""
        prompt = (
            "A software requirement was judged UNAMBIGUOUS.\n"
            "In ONE sentence, explain specifically why it is clear: point to the concrete "
            "detail (a metric, named actor, defined condition, or exact scope) that removes "
            "doubt. Do not use generic wording. Output only the sentence.\n\n"
            f'Requirement: "{req}"'
        )
        try:
            text = self.call_ollama(prompt).strip()
            first_line = text.split("\n")[0].strip().strip('"') if text else ""
            return first_line or CANNED_CLEAR
        except Exception:
            return CANNED_CLEAR

    def _retrieve(self, req: str):
        """Return (balanced few-shot examples, indices of the raw top-k neighbours, all scores)."""
        q_vec = self.embedder.encode(req, convert_to_tensor=True)
        scores = util.cos_sim(q_vec, self.example_embeddings)[0]
        order = scores.argsort(descending=True).cpu().numpy()
        scores = scores.cpu().numpy()
 
        # Wider candidate pool, then balance ambiguous / unambiguous
        candidates = self.examples_df.iloc[order[:K_EXAMPLES * 4]]
        half = K_EXAMPLES // 2
        ambig = candidates[candidates["label"] == AMBIG_LABEL].head(half)
        unambig = candidates[candidates["label"] == UNAMBIG_LABEL].head(K_EXAMPLES - half)
 
        # Shuffle so the model doesn't see a pattern in ordering
        balanced = (
            pd.concat([ambig, unambig])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        return balanced, order[:K_EXAMPLES], scores

    
    # def analyze_ambiguity(self, req: str):
        # query_emb = self.embedder.encode([req])
        # sim_scores = (query_emb @ self.example_embeddings.T)[0]
        # top_indices = sim_scores.argsort()[-2:][::-1]
        # similar = self.examples_df.iloc[top_indices]

#         # ── Compute confidence from retrieval similarity ──────────
#         top_sim_scores = sim_scores[top_indices]
#         retrieval_confidence = float(top_sim_scores.mean())  # 0.0 to 1.0
#         # Blend retrieval similarity + label agreement between top examples
#         labels = [row["label"].upper() for _, row in similar.iterrows()]
#         label_agreement = 1.0 if len(set(labels)) == 1 else 0.5  # both same label = more confident
        
#         examples_block = ""
#         for _, row in similar.iterrows():
#             lbl = row["label"].strip().capitalize()
#             examples_block += f"Requirement: {row['requirement']}\nClassification: {lbl}\n\n"

#         prompt = f"""You are an expert software requirements engineer.

# {AMBIGUITY_TAXONOMY}

# ══ TASK ══
# Analyse the given requirement. Think step-by-step:
#   Step 1 — Identify every word, phrase, pronoun, or structure that could be unclear.
#   Step 2 — Decide which ambiguity type(s) from the taxonomy apply (if any).
#   Step 3 — Classify as "ambiguous" or "unambiguous".
#   Step 4 — Write one concise explanation sentence.

# ══ WHAT IS NOT AMBIGUOUS ══
# - Passive voice ("shall be conditioned", "shall be qualified", "shall be provided",
#   "shall be avoided") is STANDARD in requirements engineering — do NOT flag as missing_actor.
# - Technical acronyms and domain terms (MPAS, ERMS, CITIS, GSTSIM, EME, DMS, FTP, SPI)
#   are NOT ambiguous on their own.
# - A requirement with multiple clauses is NOT automatically ambiguous.
# - Words like "other", "related", "associated", "respective" are NOT ambiguous
#   unless their referent is genuinely unclear.

# ══ WHAT IS AMBIGUOUS ══
# - Pronouns (it, they, their, its, this, these) where the referent is genuinely unclear.
# - Unmeasurable terms (fast, soon, adequate, sufficient, appropriate, user-friendly)
#   with no defined metric or threshold.
# - Coordination (and/or) that allows multiple groupings with different meanings.
# - Missing actor ONLY when it is impossible to infer who performs the action
#   even from context — passive voice alone is NOT enough.
# - Temporal conditions with no defined time threshold.

# ══ LABELLED EXAMPLES (similar to the test requirement) ══
# {examples_block}

# ══ RULES ══
# - Output ONLY valid JSON — no markdown, no extra text before or after.
# - "classification" must be exactly "ambiguous" or "unambiguous".
# - "ambiguity_type" must be one of: pronoun, vague_language, missing_actor, coordination,
#   scope_quantifier, temporal, conditional, negation, none.
# - "explanation" must be one clear sentence stating the specific issue found
#   or confirming why the requirement is clear.
# - BEFORE outputting, ask yourself: "Can I point to one specific word or phrase
#   that would cause two developers to implement this differently?"
#   If yes → ambiguous. If no → unambiguous.

# Requirement to classify: "{req}"
# Return ONLY JSON:
# {{"classification": "ambiguous or unambiguous", "ambiguity_type": "pronoun/vague_language/missing_actor/coordination/scope_quantifier/temporal/conditional/negation/none", "explanation": "one sentence"}}"""
    def analyze_ambiguity(self, req: str):
        
        examples, top_idx, scores = self._retrieve(req)
        prompt = build_prompt(req, build_examples(examples))
 
        raw = self.call_ollama(prompt)
        parsed = extract_json(raw)

        label = None
        explanation = ""
        ambiguity_type = "none"
 
        if parsed:
            lbl = str(parsed.get("classification", "")).strip().lower()
            if lbl == UNAMBIG_LABEL or ("unambiguous" in lbl):
                label = UNAMBIG_LABEL
            elif "ambiguous" in lbl:
                label = AMBIG_LABEL
            explanation = parsed.get("explanation", "")
            ambiguity_type = parsed.get("ambiguity_type", "none")
 
        if label is None:
            label = keyword_fallback(raw)
            if not parsed:
                explanation = raw[:200].replace("\n", " ")
 
        if label is None:
            # Model output was unusable → flag for manual review instead of crashing
            return {
                "verdict": "ambiguous",
                "confidence": 0.0,
                "ambiguity_type": "none",
                "reasons": ["Model output could not be parsed; flagged for manual review."],
            }

        # The model sometimes parrots the canned example sentence for clear requirements
        if label == UNAMBIG_LABEL and (
            not explanation.strip()
            or explanation.strip().lower().startswith("the requirement is specific and measurable")
        ):
            explanation = self._explain_clear(req)

        # ── Confidence: retrieval similarity + how many of the raw top-k
        #    neighbours agree with the model's verdict ──────────────────────
        similarity = float(scores[top_idx].mean())
        neighbour_labels = self.examples_df["label"].iloc[top_idx]
        agreement = float((neighbour_labels == label).mean())
        confidence = max(0.0, min(1.0, similarity * 0.7 + agreement * 0.3))
        
        # match = re.search(r'\{.*?\}', raw, re.DOTALL)
        # if match:
        #     parsed = json.loads(match.group())
        #     lbl = parsed.get("classification", "").strip().lower()
        #     verdict = "ambiguous" if "ambiguous" in lbl and "unambiguous" not in lbl else "clear"
        #     # confidence = float(parsed.get("confidence", 0.0))
        #     explanation = parsed.get("explanation", "")
        #     ambiguity_type = parsed.get("ambiguity_type", "none")
        # else:
        #     verdict = "ambiguous" if "AMBIGUOUS" in raw.upper() else "clear"
        #     # confidence = 0.0
        #     explanation = raw
        #     ambiguity_type = "none"

        # return {
        #     "verdict": verdict,
        #     "confidence": round(retrieval_confidence * 0.7 + label_agreement * 0.3, 2),
        #     "ambiguity_type": ambiguity_type,
        #     "reasons": [explanation]
        # }
        return {
            "verdict": "ambiguous" if label == AMBIG_LABEL else "clear",
            "confidence": round(confidence, 2),
            "ambiguity_type": ambiguity_type,
            "reasons": [explanation],
        }

llm_service = LLMService()