# from re import search

import Imports as lib
import Processing as preprocess
import nltk
nltk.download('stopwords')

# ── Load & prepare data (same as Models_training.py) ─────────────────────
data = lib.pd.read_csv("requirements_dataset.csv")
data = data.rename(columns={"Context": "Requirement", "DetectedAs": "isAmbiguous"})
data["Comment"] = "NA"
data["isAmbiguous"] = data["isAmbiguous"].str.strip().str.upper()
data["isAmbiguous"] = data["isAmbiguous"].map({"AMBIGUOUS": "YES", "UNAMBIGUOUS": "NO"})
data = preprocess.preprocess_data(data)

y = (data["isAmbiguous"] == "YES").astype(int).values
X = data["Requirement"]   # single column — no Comment
# Split FIRST, tune only on train set
from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.75, stratify=y, random_state=42
)

# ── Search spaces ─────────────────────────────────────────────────────────
tfidf_grid = {
    'tfidf__max_features' : [5000, 10000],
    'tfidf__min_df'       : [1, 2],
    'tfidf__max_df'       : [0.85, 0.90, 0.95],
    'tfidf__ngram_range'  : [(1, 1), (1, 2)],
    'tfidf__norm'         : ['l2'],
    'tfidf__use_idf'      : [True],
    'tfidf__smooth_idf'   : [True],
}

svc_grid = {
    **tfidf_grid,
    'clf__C'      : [0.1, 1, 10, 100],
    'clf__kernel' : ['rbf', 'linear'],
    'clf__gamma'  : ['scale', 'auto'],
}

dt_grid = {
    **tfidf_grid,
    'clf__max_depth'        : [5, 10, 15, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf' : [1, 2, 5],
    'clf__max_features'     : ['sqrt', 'log2'],
    'clf__criterion'        : ['gini', 'entropy'],
}

nb_grid = {
    **tfidf_grid,
    'clf__alpha': [0.001, 0.01, 0.1, 1.0],
}

cv = lib.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune(name, classifier, param_grid):
    print(f"\nTuning {name}...")
    pipeline = lib.Pipeline([
        ('tfidf', lib.TfidfVectorizer()),
        ('clf',   classifier)
    ])
    search = lib.GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',   # optimise for F1 — better than accuracy on imbalanced data
        n_jobs=-1,               # use all CPU cores
        verbose=1
    )
     # Tune only on training data
    search.fit(X_train, y_train)   # ← not X, y
    # search.fit(X, y)
    # Evaluate on held-out eval
    search.best_estimator_.predict(X_eval)
    print(f"  Best Accuracy : {search.best_score_:.4f}")
    print(f"  Best params : {search.best_params_}")
    return search.best_params_

# ── Run tuning ────────────────────────────────────────────────────────────
best_svc = tune("SVC",           lib.SVC(),                    svc_grid)
best_dt  = tune("Decision Tree", lib.DecisionTreeClassifier(), dt_grid)
best_nb  = tune("Naive Bayes",   lib.MultinomialNB(),          nb_grid)

# ── Extract and save in the format Best_parameters.py expects ─────────────
def extract(prefix, params):
    return {k.replace(f"{prefix}__", ""): v 
            for k, v in params.items() 
            if k.startswith(prefix)}

tfidf_svm_best_parameters = {
    'tfidf': extract('tfidf', best_svc),
    'svc'  : extract('clf',   best_svc),
}
tfidf_dt_best_parameters = {
    'tfidf'        : extract('tfidf', best_dt),
    'decision_tree': extract('clf',   best_dt),
}

lib.joblib.dump(tfidf_svm_best_parameters, 'tfidf_svm_g_more.joblib')
lib.joblib.dump(tfidf_dt_best_parameters,  'tfidf_decisiontree_g_more.joblib')
print("\nSaved: tfidf_svm_g_more.joblib, tfidf_decisiontree_g_more.joblib")