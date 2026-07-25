import sys
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

sys.path.append('.')
try:
    import Imports as lib
    import Processing as preprocess
except ImportError:
    pass

class MLService:
    def __init__(self):
        self.vectorizer = None
        self.svc_model = None
        self.feature_gap = 0

    def initialize(self):
        base_path = Path(__file__).resolve().parents[2] / "models"
        self.vectorizer = joblib.load(base_path / "vectorizer.joblib")
        self.svc_model = joblib.load(base_path / "svc_model.joblib")

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