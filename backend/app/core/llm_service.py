import re
import json
import requests
from backend.app.core.vector_service import vector_service


# # Resolves to `backend/`
# BASE_DIR = Path(__file__).resolve().parents[2]
# DATASET_PATH = BASE_DIR / "data" / "requirements_dataset.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

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

class LLMService:

    def initialize(self):
        print("LLM Service initialized.")

    def call_ollama(self, prompt: str) -> str:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]

    def analyze_ambiguity(self, req: str, doc_id: str = None) -> dict:
        # 1. Fetch document context from Vector DB if doc_id is provided
        context_clauses = vector_service.retrieve_context(doc_id, req, top_k=5) if doc_id else []
        
        context_block = "\n".join([f"- {c}" for c in context_clauses]) if context_clauses else "No prior document context provided."

        # 2. Context-Aware Prompt
        prompt = f"""You are an expert software requirements engineer.

{AMBIGUITY_TAXONOMY}

DOCUMENT CONTEXT (Other clauses in the project):
{context_block}

TARGET REQUIREMENT TO ANALYZE:
"{req}"

Task:
1. Check if the TARGET REQUIREMENT directly contradicts or conflicts with any clause in the DOCUMENT CONTEXT.
2. Check for linguistic or general ambiguity using the taxonomy above.
Note: 
- Named entities (Canada, DND, RCN, Contractor) are NOT ambiguous — 
  they refer to defined parties in the contract
- Acronyms defined in the contract (HCCS, CPP, EGPP, ISS, WBS) 
  are NOT ambiguous on their own
  
Return ONLY JSON:
{{"classification": "ambiguous or unambiguous", "ambiguity_type": "pronoun/vague_language/missing_actor/coordination/scope_quantifier/temporal/conditional/negation/none","conflict_with": "quote context requirement if applicable, else 'none'", "confidence": 0.9, "explanation": "Provide a brief explanation of your reasoning."}}"""

        raw = self.call_ollama(prompt)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            lbl = parsed.get("classification", "").strip().lower()
            # Fixed ordering check:
            if "contradict" in lbl:
                verdict = "contradiction"
            elif "unambiguous" in lbl or "clear" in lbl:  # <--- Check unambiguous FIRST!
                verdict = "clear"
            elif "ambiguous" in lbl:
                verdict = "ambiguous"
            else:
                verdict = "clear"

            return {
                "verdict": verdict,
                "confidence": float(parsed.get("confidence", 0.0)),
                "ambiguity_type": parsed.get("ambiguity_type", "none"),
                "conflict_with": parsed.get("conflict_with", "none"),
                "reasons": [parsed.get("explanation", "")]
            }
        else:
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "ambiguity_type": "none",
                "conflict_with": "none",
                "reasons": [raw]
            }

llm_service = LLMService()