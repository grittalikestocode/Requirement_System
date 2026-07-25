"""
Requirements Classifier
=======================
Trains an NLP + ML pipeline on labelled sentence pairs
(full_text.txt vs requirements_only.txt) to learn what distinguishes a
real requirement from noise, then classifies any new document into
requirements vs non-requirements with per-sentence probability scores
and explainability terms.

Methodology mirrors Beqiri et al. (2024) "Classifying Ambiguous
Requirements: An Explainable Approach in Railway Industry":
  - Porter stemming  (custom implementation, no NLTK download needed)
  - Stop-word removal (bundled list)
  - TF-IDF vectorisation with smoothing
  - SVM  (primary classifier, best F-score)
  - Naive Bayes and Decision Tree (comparison + explainability)
  - 10-fold stratified group cross-validation (each source file stays together)
  - Decision-Tree feature importance for per-sentence key terms

Usage
-----
  # Train and evaluate on the provided dataset pairs:
  python classifier.py --train

  # Classify a new document after training:
  python classifier.py --classify my_spec.txt out.csv

  # Both at once:
  python classifier.py --train --classify my_spec.txt out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.base import clone


# ==============================================================================
# TEXT PREPROCESSING: STOPWORDS, STEMMING, TOKENIZATION
# ==============================================================================

_BASE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "may", "might", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "then", "that", "these", "those",
    "this", "which", "who", "whom", "whose", "what", "where", "when",
    "how", "why", "all", "any", "both", "each", "few", "more", "most",
    "own", "same", "too", "very", "just", "because", "while", "although",
    "however", "therefore", "thus", "hence", "its", "it", "their", "they",
    "he", "she", "we", "you", "i", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "its", "into", "through", "during",
    "before", "after", "above", "below", "between", "up", "down", "out",
    "off", "over", "under", "again", "further", "once", "here", "there",
    "about", "against", "around", "along", "also", "per",
}

# Domain noise words that appear in document structure but not in requirements
_STRUCTURE_STOPWORDS = {
    "see", "figure", "fig", "table", "section", "chapter", "appendix",
    "note", "revision", "history", "version", "document", "page",
    "prepared", "reviewed", "approved", "date", "author", "owner",
    "ref", "reference", "regarding", "subject", "status", "draft",
    "meeting", "attendees", "action", "item", "follow", "next",
    "end", "beginning", "above", "below", "following", "previous",
    "given", "then", "where", "case", "example", "e.g", "i.e",
    "etc", "inc", "vs", "rev", "ver", "jan", "feb", "mar", "apr",
    "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}


STOPWORDS = _BASE_STOPWORDS | _STRUCTURE_STOPWORDS


# ==============================================================================
# PORTER STEMMER (custom implementation - no external dependencies)
# ==============================================================================

class PorterStemmer:
    """
    Lightweight Porter stemmer ported from the original algorithm.
    Sufficient for English RE text; avoids any external dependency.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def stem(self, word: str) -> str:
        if word in self._cache:
            return self._cache[word]
        w = word.lower()
        if len(w) <= 2:
            return w
        result = self._step1a(w)
        result = self._step1b(result)
        result = self._step1c(result)
        result = self._step2(result)
        result = self._step3(result)
        result = self._step4(result)
        result = self._step5(result)
        self._cache[word] = result
        return result

    # ---- helpers ----
    @staticmethod
    def _cons(word: str, i: int) -> bool:
        c = word[i]
        if c in "aeiou":
            return False
        if c == "y":
            return i == 0 or not PorterStemmer._cons(word, i - 1)
        return True

    @classmethod
    def _m(cls, word: str) -> int:
        n = len(word)
        i, count = 0, 0
        while i < n and cls._cons(word, i):
            i += 1
        while i < n:
            while i < n and not cls._cons(word, i):
                i += 1
            while i < n and cls._cons(word, i):
                i += 1
                count += 1
        return count

    @classmethod
    def _vowel_in_stem(cls, word: str) -> bool:
        return any(not cls._cons(word, i) for i in range(len(word)))

    @classmethod
    def _double_c(cls, word: str) -> bool:
        return (len(word) >= 2 and word[-1] == word[-2]
                and cls._cons(word, len(word) - 1))

    @classmethod
    def _cvc(cls, word: str) -> bool:
        n = len(word)
        return (n >= 3
                and cls._cons(word, n - 3)
                and not cls._cons(word, n - 2)
                and cls._cons(word, n - 1)
                and word[-1] not in "wxy")


    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        flag = False
        if word.endswith("eed"):
            if self._m(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith("ed"):
            if self._vowel_in_stem(word[:-2]):
                word, flag = word[:-2], True
        elif word.endswith("ing"):
            if self._vowel_in_stem(word[:-3]):
                word, flag = word[:-3], True
        if flag:
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif self._double_c(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif self._m(word) == 1 and self._cvc(word):
                word += "e"
        return word

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and self._vowel_in_stem(word[:-1]):
            word = word[:-1] + "i"
        return word

    def _step2(self, word: str) -> str:
        suffixes = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suf, rep in suffixes:
            if word.endswith(suf) and self._m(word[: -len(suf)]) > 0:
                return word[: -len(suf)] + rep
        return word

    def _step3(self, word: str) -> str:
        suffixes = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suf, rep in suffixes:
            if word.endswith(suf) and self._m(word[: -len(suf)]) > 0:
                return word[: -len(suf)] + rep
        return word

    def _step4(self, word: str) -> str:
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize",
        ]
        for suf in suffixes:
            if word.endswith(suf):
                stem = word[: -len(suf)]
                if self._m(stem) > 1:
                    if suf == "ion" and stem and stem[-1] in "st":
                        return stem
                    elif suf != "ion":
                        return stem
        return word

    def _step5(self, word: str) -> str:
        if word.endswith("e"):
            m = self._m(word[:-1])
            if m > 1 or (m == 1 and not self._cvc(word[:-1])):
                word = word[:-1]
        if (len(word) > 1 and word.endswith("ll")
            and self._m(word[:-1]) > 1):
            word = word[:-1]
        return word


# Global stemmer instance and tokenizer pattern
_stemmer = PorterStemmer()
_TOKENISE = re.compile(r"[a-zA-Z]+")


# ==============================================================================
# TEXT PREPROCESSING PIPELINE
# ==============================================================================

def preprocess(text: str) -> str:
    """
    Tokenise → lowercase → remove stop words → Porter-stem.
    Returns a single space-joined string for TF-IDF.
    """
    tokens = _TOKENISE.findall(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)



def split_sentences(text: str) -> List[str]:
    """
    Split a document into individual sentences.
    Handles soft-wrapped lines (no blank line between continuation lines)
    and preserves bullet / numbered list items.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text)

    sentences: List[str] = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        buf = ""
        for line in lines:
            is_bullet = bool(re.match(r"^[\-\*\•]\s+", line))
            is_numbered = bool(re.match(r"^\d+[\.\)]\s+", line))
            if is_bullet or is_numbered:
                if buf:
                    sentences.append(buf.strip())
                    buf = line
                continue
            if not buf:
                buf = line
                continue
            ends_sent = buf.rstrip().endswith((".", "!", "?", "]"))
            starts_cap = bool(re.match(r"^[A-Z\(\"\[]", line))
            if ends_sent and starts_cap:
                sentences.append(buf.strip())
                buf = line
            else:
                buf = buf + " " + line
        if buf.strip():
            sentences.append(buf.strip())

    # Second pass: split prose sentences within a block
    final: List[str] = []
    for sent in sentences:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", sent)
        final.extend(p.strip() for p in parts if len(p.split()) >= 3)
    return final


#Build the entire dataset:

@dataclass
class LabelledSentence:
    text: str
    label: int          # 1 = requirement, 0 = non-requirement
    source: str         # which file pair it came from


def _normalize_sentence(s: str) -> str:
    """Normalize sentence: collapse whitespace, lowercase for fuzzy matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def build_dataset(pairs: List[Tuple[str, str]]) -> List[LabelledSentence]:
    """
    Build a labelled dataset from (full_text_path, requirements_only_path) pairs.

    For each pair:
      - Split both files into sentences
      - Match sentences: if a sentence from full_text appears in requirements_only, label it 1
      - Otherwise label it 0 (non-requirement)
      - Filter out noise (short sentences, headers, metadata)
    """
    rows: List[LabelledSentence] = []

    _noise_re = re.compile(
        r"^(figure|fig\.?|table|tab\.?|section|chapter|appendix|"
        r"note:|n\.b|warning|caution|see |rev |v\d|end of|"
        r"next review|document owner|action:|follow.up|\d+\.\d+\s|"
        r"[\=\-\_\*]{4,}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    )

    for full_path, req_path in pairs:
        full_text = Path(full_path).read_text(encoding="utf-8")
        req_text  = Path(req_path).read_text(encoding="utf-8")

        full_sents = split_sentences(full_text)
        req_sents  = split_sentences(req_text)

        # Build set of normalized requirement sentences for fast O(1) lookup
        req_norms = {_normalize_sentence(s) for s in req_sents}

        source_name = Path(full_path).stem

        for sent in full_sents:
            if len(sent.split()) < 4:
                continue
            if _noise_re.match(sent):
                continue

            norm = _normalize_sentence(sent)
            # Check if this sentence exists in the requirements set
            label = 1 if norm in req_norms else 0
            rows.append(LabelledSentence(text=sent, label=label, source=source_name))

    return rows



# ==============================================================================
# MACHINE LEARNING MODELS
# ==============================================================================
# Three classifiers for comparison: SVM (primary), Decision Tree (explainability), Naive Bayes

MODELS = {
    "SVM":          SVC(C=0.5, kernel="linear", gamma="scale",
                        probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(
        min_samples_split=6, min_samples_leaf=2,
        max_features="sqrt", criterion="gini",
        random_state=42),
    "NaiveBayes":   ComplementNB(alpha=0.5),
}

# MODEL_SAVE_PATH = Path("models/trained_classifier.pkl")
MODEL_SAVE_PATH = Path("C:/Users/HP PAVILION/Downloads/trained_classifier.pkl")

def calculate_per_file_metrics(
        test_indices: np.ndarray,
        sources: List[str],
        y_test: np.ndarray,
        y_pred: np.ndarray,
) -> Dict[str, Dict[str, any]]:
    """
    Calculate performance metrics broken down by source file.
    Returns metrics: TP, FP, FN, precision, recall, F1 for each file in the test set.
    """
    per_file = {}
    test_sources = [sources[i] for i in test_indices]
    unique_files = sorted(set(test_sources))
    
    for file in unique_files:
        # Filter to only sentences from this file
        file_mask = np.array([test_sources[j] == file for j in range(len(test_sources))])
        y_true_file = y_test[file_mask]
        y_pred_file = y_pred[file_mask]
         
        # Count true requirements and predicted requirements
        actual_reqs = np.sum(y_true_file == 1)
        found_reqs = np.sum(y_pred_file == 1)
        tp = np.sum((y_true_file == 1) & (y_pred_file == 1))
        fp = np.sum((y_true_file == 0) & (y_pred_file == 1))
        fn = np.sum((y_true_file == 1) & (y_pred_file == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_file[file] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "found": int(found_reqs),
            "actual": int(actual_reqs),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    return per_file


def train_and_evaluate_models(
        dataset: List[LabelledSentence],
        n_folds: int = 10,
        verbose: bool = True,
        force_svm: bool = True,
        threshold: float = 0.5, 
) -> Tuple[Pipeline, TfidfVectorizer]:
    """
    Train and evaluate three classifiers: SVM, Decision Tree, and Naive Bayes.
    Uses stratified group k-fold cross-validation (keeps source files together to prevent leakage).
    Returns the best model and the fitted vectorizer for use in classification.
    """
    texts = [preprocess(s.text) for s in dataset]
    labels = [s.label for s in dataset]
    sources = [s.source for s in dataset]

    y = np.array(labels)
    sources_array = np.array(sources)
    pos = sum(labels); neg = len(labels) - pos

    if len(dataset) == 0:
        if verbose:
            print("\nERROR: No training data found!")
            print("Could not discover training pairs. Check that the data directory exists")
            print("and contains files matching the pattern: *_requirements_only.txt")
        return None, None

    if verbose:
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"  Total sentences : {len(dataset)}")
        print(f"  Requirements    : {pos}  ({100 * pos / len(dataset):.1f}%)")
        print(f"  Non-requirements: {neg}  ({100 * neg / len(dataset):.1f}%)")
        source_counts = Counter(s.source for s in dataset)
        for src, cnt in sorted(source_counts.items()):
            req_cnt = sum(1 for s in dataset if s.source == src and s.label == 1)
            print(f"    {src:<40s}  {cnt:>4} sentences, {req_cnt} requirements")

    # TF-IDF vectorisation config (same for all models)
    vectoriser_config = dict(
        max_features=800,
        max_df=0.90,
        min_df=2,
        smooth_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )

    if force_svm:
        if verbose:
            print("\nForcing SVM training on full dataset (no CV)")
        svm_model = MODELS["SVM"]
        vectoriser = TfidfVectorizer(**vectoriser_config)
        X_full = vectoriser.fit_transform(texts)

        svm_model.fit(X_full, y)

        bundle = {
            "vectoriser": vectoriser,
            "svm": svm_model,
            "feature_names": vectoriser.get_feature_names_out().tolist(),
        }
        with open(MODEL_SAVE_PATH, "wb") as fh:
            pickle.dump(bundle, fh)

        if verbose:
            print(f"\n  SVM model saved to: {MODEL_SAVE_PATH}")

        return svm_model, vectoriser

    # Otherwise perform cross-validated comparison with per-file metrics
    # StratifiedGroupKFold: keeps each source file together (train or test), prevents data leakage
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    if verbose:
        print(f"\n10-fold stratified group cross-validation (per-source-file) with per-file breakdown")
        print("=" * 100)

    # Store results per model
    model_fold_results = {name: [] for name in MODELS.keys()}
    best_model = None
    best_f1 = -1.0
    best_model_name = None

    # Perform CV for each model
    for model_name, clf_template in MODELS.items():
        if verbose:
            print(f"\n{model_name}")
            print("-" * 100)

        fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []
        all_y_true_cv = []
        all_y_pred_cv = []

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(texts, y, groups=sources_array)):
            # Vectorise for this fold
            vectoriser = TfidfVectorizer(**vectoriser_config)

            # Fit on train fold, transform both train and test
            X_train = vectoriser.fit_transform([texts[i] for i in train_idx])
            X_test = vectoriser.transform([texts[i] for i in test_idx])

            # Get labels and sources for this fold
            y_train, y_test = y[train_idx], y[test_idx]
            sources_test = sources_array[test_idx]

            # Clone model for this fold
            clf = clone(clf_template)
            clf.fit(X_train, y_train)

            # Predict using custom threshold (e.g., 0.330)
            if hasattr(clf, 'predict_proba'):
                prob_arr = clf.predict_proba(X_test)
                classes = list(clf.classes_)
                prob_req = prob_arr[:, classes.index(1)]
                y_pred = (prob_req >= threshold).astype(int)  # Change 0.330 to your desired threshold
            else:
                y_pred = clf.predict(X_test) 

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recs.append(rec)
            fold_f1s.append(f1)

            # Per-file breakdown - understand which documents are challenging
            per_file = calculate_per_file_metrics(test_idx, sources_array, y_test, y_pred)

            # Print per-file results for this fold
            if verbose:
                if fold_idx == 0:
                    print(f"  {'Fold':<6} {'File':<45} {'Found':>5} {'Actual':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
                    print(f"  {'-'*111}")

                for file in sorted(per_file.keys()):
                    metrics = per_file[file]
                    print(f"  {fold_idx+1:<6} {file:<45} {metrics['found']:>5} {metrics['actual']:>6} "
                          f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4} "
                          f"{metrics['precision']:>7.3f} {metrics['recall']:>7.3f} {metrics['f1']:>7.3f}")

            # Store fold data for classification report
            all_y_true_cv.extend(y_test)
            all_y_pred_cv.extend(y_pred)

            model_fold_results[model_name].append({
                "fold": fold_idx,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "per_file": per_file,
            })

        # Print fold summary and classification report
        if verbose:
            print(f"  {'-'*111}")
            print(f"  Fold Averages:")
            print(f"    Accuracy :  {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
            print(f"    Precision:  {np.mean(fold_precs):.3f} ± {np.std(fold_precs):.3f}")
            print(f"    Recall   :  {np.mean(fold_recs):.3f} ± {np.std(fold_recs):.3f}")
            print(f"    F1       :  {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
            
            # Print full classification report
            print(f"\n  Full Classification Report (aggregated across all folds):")
            print(f"  {'-'*111}")
            print(classification_report(all_y_true_cv, all_y_pred_cv, 
                                       target_names=["Non-Requirement", "Requirement"],
                                       digits=3))

        # Track best model
        avg_f1 = np.mean(fold_f1s)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_name = model_name
            best_model = clone(clf_template)

    if verbose:
        print("\n" + "=" * 100)
        print(f"Best model: {best_model_name}  (Average F1={best_f1:.3f})")
        print("=" * 100)

    # Refit best model on full dataset for final deployment
    final_vectoriser = TfidfVectorizer(**vectoriser_config)
    X_full_final = final_vectoriser.fit_transform(texts)
    best_model.fit(X_full_final.toarray(), y)

    # Also train Decision Tree on full data (used for feature importance in explainability)
    # This is always trained regardless of whether it was the best CV model
    dt = MODELS["DecisionTree"]
    dt.fit(X_full_final.toarray(), y)

    # Save
    bundle = {
        "vectoriser": final_vectoriser,
        "svm": best_model,
        "dt": dt,
        "feature_names": final_vectoriser.get_feature_names_out().tolist(),
    }
    with open(MODEL_SAVE_PATH, "wb") as fh:
        pickle.dump(bundle, fh)

    if verbose:
        print(f"\n  Model saved to: {MODEL_SAVE_PATH}")

    return best_model, final_vectoriser


# ==============================================================================
# EXPLAINABILITY: KEY TERM EXTRACTION
# ==============================================================================

def get_key_terms(
        raw_text: str,
        vectoriser: TfidfVectorizer,
        dt: DecisionTreeClassifier,
        top_n: int = 5,
) -> List[str]:
    """
    Extract the most influential terms in a sentence for explainability.
    Weights TF-IDF values by Decision Tree feature importance to find key terms.
    """
    processed = preprocess(raw_text)
    vec = vectoriser.transform([processed]).toarray()[0]

    importances = dt.feature_importances_
    # Weight each TF-IDF value by its global importance in the Decision Tree
    weighted = vec * importances
    top_idx = np.argsort(weighted)[::-1][:top_n * 3]

    feature_names = vectoriser.get_feature_names_out()
    terms = []
    for i in top_idx:
        if weighted[i] > 0:
            terms.append(feature_names[i])
        if len(terms) >= top_n:
            break
    return terms



@dataclass
class ClassifiedSentence:
    seq_id: int
    req_id: str
    text: str
    predicted_class: str
    requirement_probability: float
    is_requirement: bool
    key_terms: str


def classify_document(
        text: str,
        model_path: Path = MODEL_SAVE_PATH,
        threshold: float = 0.330,
        include_non_req: bool = True,
) -> List[ClassifiedSentence]:
    """
    Classify all sentences in a document as requirement or non-requirement.
    Returns list of ClassifiedSentence with probabilities and key terms.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run with --train first."
        )
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)

    # Load trained models and vectorizer from bundle
    vectoriser: TfidfVectorizer       = bundle["vectoriser"]
    svm:        SVC                   = bundle["svm"]
    dt:         DecisionTreeClassifier = bundle["dt"]

    sentences = split_sentences(text)
    results: List[ClassifiedSentence] = []
    req_counter = 1

    for idx, sent in enumerate(sentences):
        processed = preprocess(sent)
        vec = vectoriser.transform([processed]).toarray()

        # Get probability that this sentence is a requirement
        prob_arr = svm.predict_proba(vec)[0]
        classes = list(svm.classes_)
        prob_req = float(prob_arr[classes.index(1)])
        is_req = prob_req >= threshold

        if not include_non_req and not is_req:
            continue

        key_terms = get_key_terms(sent, vectoriser, dt)

        # Assign IDs: REQ-0001, REQ-0002, etc. for requirements; NON-#### for others
        req_id = f"REQ-{req_counter:04d}" if is_req else f"NON-{idx+1:04d}"
        if is_req:
            req_counter += 1

        results.append(ClassifiedSentence(
            seq_id=idx + 1,
            req_id=req_id,
            text=sent,
            predicted_class="Requirement" if is_req else "Non-Requirement",
            requirement_probability=round(prob_req, 4),
            is_requirement=is_req,
            key_terms=", ".join(key_terms) if key_terms else "—",
        ))

    return results


def save_results_csv(results: List[ClassifiedSentence], output_path: str) -> None:
    fields = [
        "seq_id", "req_id", "text", "predicted_class",
        "requirement_probability", "is_requirement", "key_terms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seq_id":                   r.seq_id,
                "req_id":                   r.req_id,
                "text":                     r.text,
                "predicted_class":          r.predicted_class,
                "requirement_probability":  r.requirement_probability,
                "is_requirement":           r.is_requirement,
                "key_terms":                r.key_terms,
            })
    print(f"\nClassification written to: {output_path}")


def print_classification_summary(results: List[ClassifiedSentence]) -> None:
    # Separate results into requirements and non-requirements for summary
    reqs = [r for r in results if r.is_requirement]
    non  = [r for r in results if not r.is_requirement]
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total sentences classified : {len(results)}")
    print(f"  Requirements found         : {len(reqs)}")
    print(f"  Non-requirements filtered  : {len(non)}")
    if reqs:
        avg_p = sum(r.requirement_probability for r in reqs) / len(reqs)
        min_p = min(r.requirement_probability for r in reqs)
        max_p = max(r.requirement_probability for r in reqs)
        print(f"  Avg requirement probability: {avg_p:.3f}")
        print(f"  Probability range          : {min_p:.3f} – {max_p:.3f}")
        print(f"\n  {'ID':<12} {'Prob':>6}  Text[:68]")
        print(f"  {'-'*88}")
    for r in results[:30]:
        tag = "✓" if r.is_requirement else "✗"
        print(f"  {tag} {r.req_id:<11} {r.requirement_probability:>6.3f}  {r.text[:68]}")
    if len(results) > 30:
        print(f"  ... and {len(results)-30} more rows in the CSV.")




# ==============================================================================
# DATA DISCOVERY
# ==============================================================================

def discover_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Find training data pairs in data_dir: (full_text.txt, full_text_requirements_only.txt).
    Automatically matches files by name pattern for streamlined dataset building.
    """
    base = Path(data_dir).resolve()
    if not base.exists():
        print(f"  Warning: Data directory not found: {base}")
        return []
    all_files = list(base.glob("*.txt"))
    req_files = [f for f in all_files if "requirements_only" in f.name.lower()]
    pairs: List[Tuple[str, str]] = []

    for req in req_files:
        # Derive full_text.txt name by removing the "_requirements_only" suffix
        base_name = req.name.replace("_requirements_only.txt", ".txt")
        full = base / base_name
        
        if full.exists() and full != req:
            pairs.append((str(full), str(req)))
    
    if len(pairs) > 0:
        print(f"  Found {len(pairs)} training pairs in {base}")
            
    return pairs


# ==============================================================================
# MAIN ENTRY POINT AND CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NLP + ML Requirements Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on all dataset pairs and run 10-fold CV:
    python classifier.py --train

  Classify a new document (must train first):
    python classifier.py --classify my_spec.txt output.csv

  Train then immediately classify:
    python classifier.py --train --classify my_spec.txt output.csv

  Classify with custom probability threshold (default 0.50):
    python classifier.py --classify my_spec.txt output.csv --threshold 0.60

  Include non-requirement rows in CSV output:
    python classifier.py --classify my_spec.txt output.csv --all
        """,
    )
    parser.add_argument("--train",     action="store_true",
                        help="Train the model on all dataset pairs")
    parser.add_argument("--classify",  nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Classify INPUT file and write CSV to OUTPUT")
    parser.add_argument("--threshold", type=float, default=0.330,
                        help="Minimum probability to label as Requirement (default 0.330)")
    parser.add_argument("--all",       action="store_true",
                        help="Include non-requirement rows in the output CSV")
    parser.add_argument("--pairs",     nargs="+",
                        help="Custom training pairs: full1.txt req1.txt full2.txt req2.txt ...")

    args = parser.parse_args()

    # Determine data directory (support multiple possible locations)
    data_candidates = [
        "../../data",          # from src/core
        "../data",             # from src
        "data",                # from root
        Path(__file__).parent.parent.parent / "data",  # absolute fallback
    ]
    data_dir = None
    for candidate in data_candidates:
        if Path(candidate).exists():
            data_dir = candidate
            break
    
    if data_dir is None:
        data_dir = data_candidates[0]  # Default to most common path

    if not args.train and not args.classify:
        parser.print_help()
        sys.exit(0)

    if args.train:
        pairs = discover_pairs(data_dir)
        
        if args.pairs:
            if len(args.pairs) % 2 != 0:
                print("ERROR: --pairs must be an even number of files (full req full req ...)")
                sys.exit(1)
            else:
                pairs = [(args.pairs[i], args.pairs[i+1])
                     for i in range(0, len(args.pairs), 2)]

        print("Building labelled dataset from training pairs...")
        dataset = build_dataset(pairs)

        # Perform cross-validated comparison (not force_svm)
        train_and_evaluate_models(dataset, n_folds=10, verbose=True, force_svm=False, threshold=args.threshold)

    if args.classify:
        input_path, output_path = args.classify
        if not os.path.isfile(input_path):
            print(f"ERROR: Input file not found: {input_path}")
            sys.exit(1)

        text = Path(input_path).read_text(encoding="utf-8", errors="replace")
        print(f"\nClassifying: {input_path}  (threshold={args.threshold})")
        results = classify_document(
            text,
            threshold=args.threshold,
            include_non_req=args.all,
        )
        print_classification_summary(results)
        save_results_csv(results, output_path)


if __name__ == "__main__":
    main()
