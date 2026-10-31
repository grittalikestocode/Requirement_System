# LLM-Based Requirements Ambiguity Classifier

A work-in-progress research pipeline for classifying ambiguity in 
software requirements using open-source LLMs — built as part of an 
AI internship at a project management consultancy.

## What this does
Classifies natural language software requirements as ambiguous or 
unambiguous using locally-deployed open-source LLMs, without sending 
data to external APIs (GDPR/EU AI Act compliant).

## Why it exists
Ambiguous requirements are a leading cause of project failure. This 
tool helps project managers flag unclear requirements early using AI — 
running entirely on standard laptops with no GPU required.

## Models evaluated so far
| Model | Parameters | Best strategy | Accuracy |
|---|---|---|---|
| Mistral 7B | 7B | Few-shot | 56.8% |
| Phi-3.5 Mini | 3.8B | Zero-shot | ~51% |
| Qwen 2.5 | 7B | RAG | ~54% |
| Llama 3 | 8B | Few-shot | ~53% |
| EuroLLM | 1.7B | Zero-shot | ~38% |

*Evaluation ongoing — prompt engineering experiments in progress.*

## Prompting strategies compared
- **Zero-shot**: Direct classification with no examples
- **Few-shot**: 3–5 labelled examples in context
- **RAG**: Retrieved similar requirements as context

Key finding so far: prompt strategy outweighs model size as a 
performance factor. A well-prompted 3B model outperforms a poorly 
prompted 7B model consistently.

## Technical stack
- `llama.cpp` for CPU inference
- 4-bit GGUF quantisation for local deployment
- Python pipeline for prompt formatting, inference, and output parsing
- Mistral API for synthetic training data generation

## Deployment constraint
Designed to run on standard customer laptops (no GPU). Entire 
inference pipeline runs on-premise — no data leaves the machine.

## Status
🔄 Active — currently optimising prompting strategies to push 
accuracy beyond 60% on the held-out test set.

## Note on data
The industrial requirements dataset is proprietary. Experiments on 
a public requirements dataset (PROMISE NFR / ESA) coming soon.

## Evaluating the backend LLM service

From the repository root, run the evaluator with Ollama running and the
dataset available at `backend/data/requirements_dataset.csv`:

```bash
python -m Experiments.evaluate_llm_service --sample-size 30
```

The evaluator creates a reproducible stratified split: 25% of the rows are
used as the service's retrieval examples and 75% are held out for evaluation.
It saves each prediction, confidence, ambiguity type, explanation, and any
row-level error to `dev_results_llm_service.csv`. Accuracy, per-class
precision/recall/F1, and a confusion matrix are printed for successful
predictions; errors are counted separately and excluded from those metrics.

Pass `--dataset path/to/file.csv` to use another CSV, and `--output
path/to/results.csv` to choose where results are saved. The CSV must contain
either `Context`/`DetectedAs` or `requirement`/`label` columns, with labels
`ambiguous`, `unambiguous`, or `clear`.

## Evaluating the backend ML classifier

From the repository root, run:

```bash
python -m Experiments.evaluate_ml_service --sample-size 30
```

This uses the same reproducible stratified 25%/75% split and writes
per-requirement predictions and confidence scores to
`dev_results_ml_service.csv`. It clones the vectorizer and SVC settings from
`backend/models/`, then fits fresh instances on only the 25% training split
before scoring the held-out rows. This is important because the current
`models_training.py` saves its SVC after fitting on `X_eval`; evaluating those
saved fitted artifacts directly on that same split would not be a valid
held-out evaluation.

As with the LLM evaluator, pass `--dataset path/to/file.csv` and
`--output path/to/results.csv` to customize the input and output paths.
