# Requirement Analyzer

An intelligent requirement analysis tool built with **FastAPI**, **Ollama (Mistral 7B)**, and **Scikit-Learn (SVC)**. This application processes contract and agreement PDFs to extract actual requirements and evaluates requirement text for ambiguity using both local LLM and classic Machine Learning models.

---

##  Features

* **PDF Requirement Extraction:** Automatically parses contract/agreement PDFs and classifies text segments into actionable requirements vs. non-requirements.
* **Ambiguity Detection:** Analyzes requirement statements for ambiguity using two distinct approaches:
  * **LLM Analysis:** Quantized Mistral 7B via local Ollama deployment. Deep contextual ambiguity detection with taxonomy-based explanations.
  * **ML Classifier:** TF-IDF Vectorizer + Support Vector Classifier (SVC). Fast, pattern-based ambiguity classification.
* **Model Comparison:** Allows users to run single-text checks or batch uploads and side-by-side compare LLM vs. ML performance.
* **Local & Private Execution:** Designed for privacy and strict specs—runs fully offline without third-party API dependencies.

---

## 🏗️ Project Architecture

```text
llm-requirement-analyzer/
│
├── backend/                  # Python backend core
│   ├── app/
│   │   ├── api/              # API routes (PDF parsing & Ambiguity checks)
│   │   ├── core/             # Business logic (LLM, ML services & PDF extraction)
│   │   └── main.py           # FastAPI entry point
│   └── models/               # Saved ML model artifacts (.joblib)
│   ├── RAPR/                 # Requirement Extraction parser dependencies
│
├── frontend/                 # Web interface (HTML, CSS, JS)
├── notebooks/                # Experimental research & Jupyter notebooks
├── .gitignore
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
---

## 🚀 Getting Started

### Prerequisites

* **Python:** 3.10 or higher
* **Ollama:** Installed locally with the `mistral` model pulled (`ollama pull mistral`)

---

### Set Up Virtual Environment & Install Dependencies
**Clone the repository and set up a Python virtual environment**:

```bash
git clone [https://github.com/your-username/llm-requirement-analyzer.git](https://github.com/your-username/llm-requirement-analyzer.git)
cd llm-requirement-analyzer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## 🏃 Running the Application
###  Start the FastAPI backend server from the project root directory:

```bash
uvicorn backend.app.main:app --reload --port 3000
```
The API will be available at http://127.0.0.1:3000
Interactive API Docs (Swagger): http://127.0.0.1:3000/docs

 Launch the Frontend
You can open ```frontend/index.html``` directly in your browser or run it using VS Code Live Server.
