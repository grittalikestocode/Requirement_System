import Imports as lib
import Processing as preprocess
import nltk
nltk.download('stopwords')
from Best_parameters import tf_idf_params, dt_params, svc_params

# # ── Hardcoded params (no Best_parameters.py needed yet) ──────────────────
# tf_idf_params = {
#     'max_features': 5000,
#     'max_df': 0.95,
#     'min_df': 2,
#     'smooth_idf': True,
#     'norm': 'l2',
#     'use_idf': True
# }
# svc_params  = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
# dt_params   = {
#     'max_depth': 10,
#     'min_samples_split': 5,
#     'min_samples_leaf': 2,
#     'max_features': 'sqrt',
#     'criterion': 'gini'
# }

# Load the data
# data = lib.pd.read_excel(r"C:\Users\HP PAVILION\Downloads\requirements_data.xlsx")  #Replace it with your data
# NEW
data = lib.pd.read_csv("requirements_dataset.csv")
data = data.rename(columns={"Context": "Requirement"})
data["Comment"] = "NA"           # dummy column — no comment data in your dataset
data = data.rename(columns={"DetectedAs": "isAmbiguous"})
data["isAmbiguous"] = data["isAmbiguous"].str.strip().str.capitalize()
# Now values are "Ambiguous" / "Unambiguous" — map to YES/NO for compatibility
data["isAmbiguous"] = data["isAmbiguous"].map({"Ambiguous": "YES", "Unambiguous": "NO"})
# Preprocess the data
data = preprocess.preprocess_data(data)

# ''' preprocess.get_dummies retrieves the data columns is_Ambiguous, changes the categorical values YES/NO to 0/1,
# and represents it with a single categorical feature: 1 = is ambiguous, 0 = is not ambiguous. '''
# # Get the labels
# y = lib.pd.get_dummies(data['isAmbiguous'])
# y = y.iloc[:, 1].values

# # Retrieve the data for training
# X = data[['Requirement', 'Comment']]
y = (data["isAmbiguous"] == "YES").astype(int).values   # 1 = ambiguous, 0 = not
X = data[['Requirement', 'Comment']]

#same split as LLM evaluation to keep the same eval set for both
from sklearn.model_selection import train_test_split
prompt_dev_df, eval_df = train_test_split(
    X.assign(label=y), test_size=0.75, stratify=y, random_state=42
)
X_eval = eval_df[['Requirement', 'Comment']]
y_eval = eval_df['label'].values

def model_evaluation(X, y):
    # Create empty lists to store the scores for the models that will be trained
    nb_scores = []  # naivebayes
    svc_scores = []  # svc
    dtree_scores = []  # decision tree
    nb_confusion = []
    svc_confusion = []
    dtree_confusion = []

    # initialize the vectorizer
    vectorizer_req = lib.TfidfVectorizer(**tf_idf_params)
    # vectorizer_comm = lib.TfidfVectorizer(**tf_idf_params)


    # Create a pipeLine that includes the Tf-idf vectorizer and Naive Bayes classification
    """pipeline = lib.Pipeline([
        ('tfidf', lib.TfidfVectorizer()),
        ('clf', lib.MultinomialNB()),
        ('svc', lib.SVC()),
    ])"""

    # Initialize the StratifiedKFold and split the data into training and test
    skf = lib.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_i, test_i in skf.split(X, y):
        """ Train test split"""
        # split data into training and test sets
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y[train_i], y[test_i]
        lib.np.random.seed(42)
        """TF_IDF"""
        # vectorize the data with TF-IDF using the best hyperparameters

        X_train_req_tfidf = vectorizer_req.fit_transform(X_train['Requirement'])
        X_test_req_tfidf = vectorizer_req.transform(X_test['Requirement'])
        #no comment column in dataset, so we can skip it 
        # X_train_comm_tfidf = vectorizer_comm.fit_transform(X_train['Comment'])
        # X_test_comm_tfidf = vectorizer_comm.transform(X_test['Comment'])
        # combine TF-IDF matrices into single matrix
        # X_train_tfidf = lib.hstack([X_train_req_tfidf, X_train_comm_tfidf])
        # X_test_tfidf = lib.hstack([X_test_req_tfidf, X_test_comm_tfidf])
        X_train_tfidf = X_train_req_tfidf
        X_test_tfidf  = X_test_req_tfidf
        """Naive Bayes"""
        accuracy_nb, precision_nb, recall_nb, f1_score_nb, confusion_nb_matrix = preprocess.get_score_and_confusion(
            lib.MultinomialNB(alpha=0.01), X_train_tfidf, X_test_tfidf, y_train, y_test)
        # Fill the nb_scores empty list with accuracy, precision, recall, and f1-score
        nb_scores.append((accuracy_nb, precision_nb, recall_nb, f1_score_nb))
        nb_confusion.append(confusion_nb_matrix)


        """Support Vector Classifier SVC"""
        accuracy_svc, precision_svc, recall_svc, f1_score_svc, confusion_svc_matrix = preprocess.get_score_and_confusion(
            lib.SVC(**svc_params), X_train_tfidf, X_test_tfidf, y_train, y_test)
        # Fill the svc_scores empty list with accuracy, precision, recall, and f1-score
        svc_scores.append((accuracy_svc, precision_svc, recall_svc, f1_score_svc))
        svc_confusion.append(confusion_svc_matrix)


        """Decision Tree"""
        accuracy_dtree, precision_dtree, recall_dtree, f1_score_dtree, confusion_dtree_matrix = preprocess.get_score_and_confusion(
            lib.DecisionTreeClassifier(**dt_params), X_train_tfidf, X_test_tfidf, y_train, y_test)
        # Fill the dtree_scores empty list with accuracy, precision, recall, and f1-score
        dtree_scores.append((accuracy_dtree, precision_dtree, recall_dtree, f1_score_dtree))
        dtree_confusion.append(confusion_dtree_matrix)



    # Calculate the average of the scores
    avg_scores_nb = preprocess.get_avg_evalaution(nb_scores)
    # Calculate the average of the scores for SVC
    avg_scores_svc = preprocess.get_avg_evalaution(svc_scores)
    # Calculate the average of the scores for decision Tree
    avg_scores_dTree = preprocess.get_avg_evalaution(dtree_scores)

    # Sum confusion matrices across folds
    total_confusion_nb = lib.np.sum(nb_confusion, axis=0)
    total_confusion_svc = lib.np.sum(svc_confusion, axis=0)
    total_confusion_dtree = lib.np.sum(dtree_confusion, axis=0)

    # Return the scores for each classifier: svc and naive bayes
    return {
            'SVC': {
                'SVC_Accuracy': '{:.4f}'.format(avg_scores_svc['Accuracy']),
                'SVC_Precision': '{:.4f}'.format(avg_scores_svc['Precision']),
                'SVC_Recall': '{:.4f}'.format(avg_scores_svc['Recall']),
                'SVC_F1-Score': '{:.4f}'.format(avg_scores_svc['F1-score']),
                'SVC_Confusion_Matrix': total_confusion_svc.tolist(),
            },
            '\nNaive Bayes': {
                'NB_Accuracy':  '{:.4f}'.format(avg_scores_nb['Accuracy']),
                'NB_Precision': '{:.4f}'.format(avg_scores_nb['Precision']),
                'NB_Recall': '{:.4f}'.format(avg_scores_nb['Recall']),
                'NB_F1-Score': '{:.4f}'.format(avg_scores_nb['F1-score']),
                'NB_Confusion_Matrix': total_confusion_nb.tolist(),
            },
            '\nDecision Tree': {
                'DTree_Accuracy': '{:.4f}'.format(avg_scores_dTree['Accuracy']),
                'DTree_Precision': '{:.4f}'.format(avg_scores_dTree['Precision']),
                'DTree_Recall': '{:.4f}'.format(avg_scores_dTree['Recall']),
                'DTree-F1-Score': '{:.4f}'.format(avg_scores_dTree['F1-score']),
                'DTree_Confusion_Matrix': total_confusion_dtree.tolist(),
            }
        }

# Invoke the function
# model_scores = model_evaluation(X, y)
model_scores = model_evaluation(X_eval, y_eval)
for model, scores in model_scores.items():
    print(model)
    print(scores)


#_________________________________________# Save the models_________________________________________________________
""" To store trained models in a .joblib file, use the lines below. 
Examples include vectorizer_requirement, joblib, etc.
When using new data, you may change the names.
"""
#lib.joblib.dump(vectorizer_req, 'vectorizer_requirement.joblib')
#lib.joblib.dump(vectorizer_comm, 'vectorizer_comment.joblib')
# save decision tree
#lib.joblib.dump(DecisionTreeClassifier, 'decisiontree_model.joblib')

vectorizer_final = lib.TfidfVectorizer(**tf_idf_params)
X_tfidf_final = vectorizer_final.fit_transform(X_eval['Requirement'])

# Train final SVC
svc_final = lib.SVC(**svc_params, probability=True)  # probability=True lets us get confidence scores
svc_final.fit(X_tfidf_final, y_eval)

# Save both
lib.joblib.dump(vectorizer_final, 'vectorizer.joblib')
lib.joblib.dump(svc_final, 'svc_model.joblib')
print("Models saved.")
