# Copilot instructions

## Run and evaluate

Run commands from the repository root.

```bash
python -m venv .venv
pip install -r requirements.txt
ollama pull mistral:7b
uvicorn backend.app.main:app --reload --port 3000
```

Activate `.venv` before installing or running Python commands (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux). Ollama must be running locally for LLM analysis. The API docs are at `http://127.0.0.1:3000/docs`; the app serves the frontend at `/`, or `frontend/index.html` can be opened directly.

For the documented held-out service evaluations, provide the local dataset at `backend/data/requirements_dataset.csv`:

```bash
python -m Experiments.evaluate_llm_service --sample-size 30
python -m Experiments.evaluate_ml_service --sample-size 30
```

Both evaluators accept `--dataset path/to/file.csv` and `--output path/to/results.csv`. The dataset is local and ignored by Git. There is currently no configured automated test suite or lint command, so no single-test command is available.

## Architecture

- `backend/app/main.py` is the FastAPI entry point. It mounts the static `frontend/` files, initializes the shared LLM and ML service instances at startup, and registers API routers under `/api`.
- `backend/app/api/ambiguity_routes.py` exposes `/api/check-llm` and `/api/check-ml`. Both accept JSON with a `requirement` field and return service results with a `verdict` of `ambiguous` or `clear`.
- `backend/app/core/llm_service.py` retrieves similar labelled examples from the local dataset using sentence embeddings, then asks the local Ollama model to classify the requirement. `backend/app/core/ml_service.py` uses the saved TF-IDF vectorizer and SVC artifacts in `backend/models/`; its text preprocessing is in `backend/app/core/utils/`.
- `backend/app/api/pdf_routes.py` and `backend/app/core/pdf_extractor.py` handle PDF/DOCX uploads. The `backend/RAPR/` segmenter and classifier extract document segments and classify them as requirements or non-requirements using the saved artifacts in `backend/RAPR/models/`.
- `frontend/` is a static HTML/CSS/JavaScript client. Its API URLs target port 3000 and use the `/api` prefix; there is no frontend package manager or build step.
- `Experiments/` contains research notebooks and scripts for evaluating the backend services. Evaluations are Python modules, hence the `python -m Experiments...` invocation from the repository root.

## Repository-specific conventions

- Keep router paths unprefixed inside each router; `main.py` supplies `/api`. Keep the browser client's endpoint paths in sync with those route registrations.
- Preserve the JSON shapes consumed by `frontend/app.js`: ambiguity results use `verdict` and `confidence` (plus service-specific explanation fields), while document extraction returns `filename`, `total_sentences`, `requirement_count`, and `sentences`.
- Resolve model and dataset locations relative to the Python source files rather than the process working directory. The LLM service expects `backend/data/requirements_dataset.csv`, which is deliberately not checked in.
- Keep the ML preprocessing used by `MLService.predict` aligned with the evaluation pipeline. The service loads existing trained artifacts; changes to feature generation or model formats can break compatibility with those artifacts.
- The frontend and backend are served together by FastAPI, but the static frontend can also be opened directly. When changing API routes or payloads, update the corresponding fetch and result-rendering logic in `frontend/app.js`.
