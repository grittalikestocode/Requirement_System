"""
ML Classifier Service
--------------------------
Loads the trained SVC model and TF-IDF vectorizer artifacts to handle text 
preprocessing, feature extraction, and ambiguity predictions.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix


# Ensure parent directory is in sys.path so modules inside app/core/utils can import each other
UTILS_DIR = Path(__file__).resolve().parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

# Import Processing and Imports safely
try:
    from app.core.utils import imports as lib
    from app.core.utils import processing as preprocess
except ImportError:
    # Fallback if running directly from utils folder context
    import imports as lib
    import processing as preprocess
class MLService:
    def __init__(self):
        self.vectorizer = None
        self.svc_model = None
        self.feature_gap = 0

    def initialize(self):
        # Resolves to `backend/models/`
        models_path = Path(__file__).resolve().parents[2] / "models"
        self.vectorizer = joblib.load(models_path / "vectorizer.joblib")
        self.svc_model = joblib.load(models_path / "svc_model.joblib")

        expected_features = self.svc_model.n_features_in_
        vectorizer_features = len(self.vectorizer.vocabulary_)
        self.feature_gap = expected_features - vectorizer_features

    def predict(self, req: str):
        df = pd.DataFrame({'Requirement': [req], 'Comment': ['NA'], 'isAmbiguous': ['NO']})
        df = preprocess.preprocess_data(df)
        cleaned_req = df['Requirement'].iloc[0]

        X = self.vectorizer.transform([cleaned_req])
        if self.feature_gap > 0:
            X = hstack([X, csr_matrix((1, self.feature_gap))])

        prediction = self.svc_model.predict(X)[0]
        confidence_scores = self.svc_model.predict_proba(X)[0]

        verdict = "ambiguous" if prediction == 1 else "clear"
        confidence = float(confidence_scores[prediction])

        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "ambiguous_phrases": [],
            "reasons": ["SVC classifier prediction (TF-IDF features)"]
        }

ml_service = MLService()