'''
The file contains the dictionary of the optimized parameters after using the HyperparameterTuning.
It is to be imported in the main file when the models are trained.
'''
import joblib
# Load the best model parameters (TF_IDF, svc ect.) from the tuned_parameters file - laudinas files
# tfidf_svm_best_parameters = joblib.load('tfidf_svm.joblib')
# dt_best_parameters = joblib.load('tfidf_decisiontree.joblib')

#grittas best parameters
# tfidf_svm_best_parameters = joblib.load('tfidf_svm_g.joblib') 
# dt_best_parameters = joblib.load('tfidf_decisiontree_g.joblib')

#grittas best parameters with more dataset in training
tfidf_svm_best_parameters = joblib.load('tfidf_svm_g_more.joblib')
dt_best_parameters = joblib.load('tfidf_decisiontree_g_more.joblib')
#tfidf_svm_best_parameters = joblib.load('svm(req).joblib')
#dt_best_parameters = joblib.load('decisiontree(req).joblib')

# Parameters for TF-IDF
tf_idf_params = {
    'max_features': tfidf_svm_best_parameters['tfidf']['max_features'],
    'max_df': tfidf_svm_best_parameters['tfidf']['max_df'],
    'min_df': tfidf_svm_best_parameters['tfidf']['min_df'],
    'smooth_idf': tfidf_svm_best_parameters['tfidf']['smooth_idf'],
    'norm': tfidf_svm_best_parameters['tfidf']['norm'],
    'use_idf': tfidf_svm_best_parameters['tfidf']['use_idf']
}
# Parameters for SVM
svc_params = {
    'C': tfidf_svm_best_parameters['svc']['C'],
    'gamma': tfidf_svm_best_parameters['svc']['gamma'],
    'kernel': tfidf_svm_best_parameters['svc']['kernel'],
}

# Parameters for Decision tress
dt_params = {
    'max_depth': dt_best_parameters['decision_tree']["max_depth"],
    'min_samples_split': dt_best_parameters['decision_tree']["min_samples_split"],
    'min_samples_leaf': dt_best_parameters['decision_tree']["min_samples_leaf"],
    'max_features': dt_best_parameters['decision_tree']["max_features"],
    'criterion': dt_best_parameters['decision_tree']["criterion"],
}
