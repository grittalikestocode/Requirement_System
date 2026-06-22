"""
This script contains all the functions required for the data processing,
and is to be imported from all the other scripts that require the functions.
"""
# Imports
import Imports as lib # contains all the necessary libraries

# Initialize the stemmer
ps = lib.PorterStemmer()
def clean_text(text):
    """
        Performs a series of data preprocessing on the text, such as
         lowercase, remove stopwords, remove special characters, stemming.
        :returns: cleaned text.
    """
    if text == 0:
        return '0'  # No comment, so return '0' as the cleaned text
    # Remove special chars and numbers
    text = lib.re.sub('[^a-zA-Z]', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    words = text.split()
    # Remove stop words
    stop_words = set(lib.stopwords.words('english'))
    # Add to the stop words the other_stopwords, take from the corpus
    other_stopwords = ["set", "given", "shall", "be", "such", "as", "a", "an", "in", "the", "na", "would"]
    stop_words.update(other_stopwords)
    # List with words not in stopword
    words = [word for word in words if word not in stop_words]
    # Stem words using PorterStemmer for English language
    stemmer = lib.PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    # Join words into a sentence
    cleaned_text = ' '.join(stemmed_words)
    return cleaned_text

def preprocess_data(df):
    """
    Performs a series of data preprocessing on the df (dataframe).
    Invokes the clean_text() function.
    :returns: the clean df.
    """
    # Reindex the columns if necessary
    if 'Requirement' in df.columns:
        if 'Comment' in df.columns and 'isAmbiguous' in df.columns:
            df = df.reindex(columns=['Requirement', 'Comment', 'isAmbiguous'])
        else:
            df = df.reindex(columns=['Requirement','Comment'])

    # Replace whitespace in the 'isAmbiguous' column
    if 'isAmbiguous' in df.columns:
        df['isAmbiguous'] = df['isAmbiguous'].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop rows where 'isAmbiguous' contains 'NoorYes' (if needed)
    if 'isAmbiguous' in df.columns:
        df = df[~df['isAmbiguous'].isin(['NoorYes'])]

    # Remove rows with missing values in 'isAmbiguous'
    if 'isAmbiguous' in df.columns:
        df = df.dropna(subset=['isAmbiguous'])

    # Fill missing values in the 'Comment' column with 'NA'
    if 'Comment' in df.columns:
        df['Comment'] = df['Comment'].fillna('NA')

    # Apply Stemming (assuming clean_text is a valid function)
    if 'Requirement' in df.columns:
        df['Requirement'] = df['Requirement'].apply(clean_text)

    if 'Comment' in df.columns:
        df['Comment'] = df['Comment'].apply(clean_text)

    # Return the cleaned DataFrame
    return df



def map_tfidf_terms_to_original(stemm_terms, req_original):
    """
    tfidf_terms: are the words resulting after applying the preprocessing to the words (stemming)
    req_original: the original requirements
    Returns:
    """
    stem = lib.PorterStemmer() # initialize the stemmer
    # Split/Tokenize the original requirement into words
    original_words = req_original.split()
    # Mapp stemmed to original words
    mapp = {stem.stem(original_word): original_word for original_word in original_words}
    # Map stemmed words to original words
    original_terms = [mapp.get(stemmed_word, stemmed_word) for stemmed_word in stemm_terms]
    return original_terms

def get_terms_for_requirements(classifier, vectorizer, sentence, n=5):
    """
        classifier: the model used for classification: exp Decision Tree
        vectorizer: the vectorizer used to apply tfidf
        sentence: the original requirement before stemming
        n: the desired numbers of terms to be extracted
    Returns: n number of words influencing the classification
    """
    if not hasattr(classifier, 'feature_importances_'):
        raise ValueError("The classifier cannot find the feature importance.")
    # Get feature importance for the classifier (decision tree)
    features = classifier.feature_importances_
    # Get feature names/terms from the TF-IDF vectorizer
    terms = vectorizer.get_feature_names()
    # Get the words in the requirement that are also in the feature names list
    words = [word for word in sentence.split() if word in terms]
    # Cals the importance for words in the requirements
    imp = [features[terms.index(word)] for word in words]
    # Create a list (word, importance)
    word_imp_set = list(zip(words, imp))
    # Sort the list in descending order
    word_imp_set.sort(key=lambda x: x[1], reverse=True)
    # Get the n important words
    n_words = [pair[0] for pair in word_imp_set[:n]]

    return n_words

def check_input_format(input_data):
    """
    This function checks if the input data is a single sentence or an Excel file. In both cases it check is the column
    Requirement and Comment exit, and creates a dataframe out of them
        input_data: the input provided: can be a sentence or an Excel.
    Returns: the dataframe created and the original_requirements, which saves the requirements before applying the stemming.
    """
    original_requirements = [] # variable to save the original requirements before stemming

    if isinstance(input_data, str) and input_data.endswith('.xlsx'):
        # If the input is an Excel file, read it into a DataFrame without header
        data = lib.pd.read_excel(input_data, header=None)
        # Check if 'Requirement' column is present, if not, rename it
        if 'Requirement' not in data.columns:
            # Rename the first column to 'Requirement'
            data = data.rename(columns={data.columns[0]: 'Requirement'})
        # Check if 'Comment' column is missing and create it and fill with zeros
        if 'Comment' not in data.columns:
            data['Comment'] = 0
        # Save original requirements after renaming the column
        original_requirements = data['Requirement'].tolist()
    elif isinstance(input_data, str):
        # If the input is a sentence Requirement, create a DataFrame with 'Requirement' and 'Comment' columns
        data = lib.pd.DataFrame({'Requirement': [input_data], 'Comment': [0]})
        original_requirements = [input_data]  # Save original Requirement
    else:
        raise ValueError("Unsupported input format. Provide an Excel file path or a text sentence.")
    return data, original_requirements

# Functions to get Training scores
def get_score(model, X_train, X_test, y_train, y_test):
    """
    Calculate the score (accuracy, precision, recall, f1_score) for a given model.
    :param:
        model: The classifier
        X_train: Data (df) to train the model
        X_test: Data to evaluate the model's performance
        y_train: numpy array that contains the labels for the training data.
        y_test: numpy array that contains the labels for the testing data.
     :return:
        The metrics for evaluating the performance of the model : accuracy, precision, recall, f1_score
    """
    model.fit(X_train, y_train) # train the model by learning the relationships in the data
    y_pred = model.predict(X_test) # use the trained model to predicted labels for the test data
    accuracy = model.score(X_test, y_test)
    precision, recall, f1_score, _ = lib.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1_score


def get_score_and_confusion(model, X_train, X_test, y_train, y_test):
    """
    Calculate the score and confusion matrix for a given model.
    :param:
        model: The classifier
        X_train: Data to train the model
        X_test: Data to evaluate the model
        y_train: numpy array labels for training
        y_test: numpy array labels for testing
    :return:
        accuracy, precision, recall, f1_score, confusion_matrix
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    precision, recall, f1_score, _ = lib.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    confusion = lib.confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1_score, confusion


def get_avg_evalaution(metrics):
    '''
    The function computes the mean of the metrics over all folds.
     :param metrics (list): contains each fold's evaluation score.
     :return: the average performance metrics across of the cross-validation folds.
    '''
    scores = lib.np.array(metrics, dtype= lib.np.float64)
    accuracy = lib.np.mean(scores[:, 0])
    precision = lib.np.mean(scores[:, 1])
    recall = lib.np.mean(scores[:, 2])
    f1_score = lib.np.mean(scores[:, 3])
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

def avg_prob_percent(model_probabilities, df):
    '''
    Calculates the average probabilities across multiple folds
    :param model_probabilities: list of probabilities for each fold,probabilities result of a classification model
    :param df: pandas dataframe
    :return: prob: list - each data point's average probability over all folds

    '''
    # Ensure all probabilities arrays (for each fold) have same shape
    c_shape = lib.np.max([arr.shape for arr in model_probabilities], axis=0)
    # Prune each array to the common shape
    model_probabilities = [lib.np.resize(arr, c_shape) for arr in model_probabilities]
    # Calculate the average probabilities among k folds
    avg_prob = lib.np.mean(model_probabilities, axis=0)
    # Normalize probabilities to add up to 1
    normalized_avg_prob = avg_prob / lib.np.sum(avg_prob, axis=1, keepdims=True)
    normalized_avg_prob_percent = normalized_avg_prob * 100
    prob = []
    for i, row in enumerate(normalized_avg_prob_percent):
        point = df[:, i]
        result_str = f"Requirement {i + 1}: {point} -> non-ambiguous : ambiguous -> {row[0]:.2f}% : {row[1]:.2f}%"
        prob.append(result_str)
    return prob
