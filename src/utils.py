from librerias import *
import os
def extract_numeric_part(flight_number):
    """
    Extracts the numeric part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        int: The numeric part of the flight number, or None if the flight_number is not a string or has no numeric part.
    """   
    if isinstance(flight_number, str):
        numeric_part = re.findall(r'\d+', flight_number)
        return int(numeric_part[0]) if numeric_part else None
    else:
        return None

def extract_alphabetic_part(flight_number):
    """
    Extracts the alphabetic part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        str: The alphabetic part of the flight number, or None if the flight_number is not a string or has no alphabetic part.
    """
    if isinstance(flight_number, str):
        alphabetic_part = re.findall(r'[a-zA-Z]+', flight_number)
        return alphabetic_part[0] if alphabetic_part else None
    else:
        return 0




        
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title='Confusion Matrix', cmap='Blues'):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    :param y_true: True labels.
    :type y_true: array-like, shape (n_samples,)
    :param y_pred: Predicted labels.
    :type y_pred: array-like, shape (n_samples,)
    :param labels: List of labels to index the matrix.
    :type labels: list of str, optional, default: None
    :param normalize: Whether to normalize the confusion matrix.
    :type normalize: bool, optional, default: False
    :param title: Title for the confusion matrix plot.
    :type title: str, optional, default: 'Confusion Matrix'
    :param cmap: Colormap to be used for the heatmap.
    :type cmap: str, optional, default: 'Blues'
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=labels, yticklabels=labels)

    # Set plot title and labels
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Show the plot
    plt.show()
    
    
def find_best_threshold(y_test, y_pred_prob):
    """
    Finds the best threshold for classification based on the highest F1 score.
    
    :param y_test: True target values.
    :type y_test: array-like
    :param y_pred_prob: Predicted probabilities for the positive class.
    :type y_pred_prob: array-like
    :return: Best threshold.
    :rtype: float
    """
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0
    best_f1_score = 0
    for t in thresholds:
        y_pred_t = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = t
    return best_threshold

def plot_roc_curve(fpr, tpr, roc_auc, classifier_name):
    """
    Plots the ROC curve for a given classifier.
    
    :param fpr: False positive rates.
    :type fpr: array-like
    :param tpr: True positive rates.
    :type tpr: array-like
    :param roc_auc: Area under the ROC curve.
    :type roc_auc: float
    :param classifier_name: Name of the classifier.
    :type classifier_name: str
    """
    plt.plot(fpr, tpr, label=f'{classifier_name} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

def train_with_cross_validation(clf, X, y, cv=5):
    """
    Trains a classifier using cross-validation and returns the average performance metrics and the trained classifier.

    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X: Training data.
    :type X: DataFrame
    :param y: Labels for the training data.
    :type y: Series
    :param cv: Number of cross-validation folds.
    :type cv: int
    :return: A dictionary containing average performance metrics and the trained classifier.
    :rtype: dict
    """

    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    avg_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred)
    }

    clf.fit(X, y)  # Train the classifier on the entire dataset

    return avg_metrics, clf

def optimize_hyperparameters(classifier, param_grid, X_train, y_train,cv):
    """
    Optimizes hyperparameters for a classifier using GridSearchCV.
    
    :param classifier: The classifier to optimize.
    :type classifier: estimator
    :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :type param_grid: dict
    :param X_train: Training data.
    :type X_train: array-like
    :param y_train: Training target values.
    :type y_train: array-like
    :return: Best estimator found by grid search.
    :rtype: estimator
    """
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1,verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def encode_categorical_columns(data, categorical_columns):
    """
    Encodes categorical columns in *data* using One-Hot Encoding.
    
    :param data: Data to be encoded.
    :type data: DataFrame
    :param categorical_columns: Categorical columns in *data*.
    :type categorical_columns: list of str
    :return: Encoded data.
    :rtype: DataFrame
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data = data[categorical_columns]
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    
    encoded_columns = []
    for cat_col, categories in zip(categorical_columns, encoder.categories_):
        encoded_columns.extend([f"{cat_col}_{category}" for category in categories])
    
    encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoded_columns, index=data.index)
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    
    return data

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a classifier using the specified data.
    
    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X_train: Training data.
    :type X_train: DataFrame
    :param y_train: Labels for the training data.
    :type y_train: Series
    :param X_test: Test data.
    :type X_test: DataFrame
    :param y_test: Labels for the test data.
    :type y_test: Series
    """
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-score: {f1_score(y_test, y_pred)}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_prob)}")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def generate_requirements_txt(filename="requirements.txt"):
    """
    Generates a `requirements.txt` file containing a list of all installed packages in the current environment.

    Parameters:
    -----------
    filename : str, optional (default="requirements.txt")
        The name of the file to be generated, without any path information.

    Returns:
    --------
    None
    """

    # run pip freeze command to get a list of installed packages
    installed_packages = subprocess.check_output(['pip', 'freeze']).decode('utf-8').split('\n')

    # create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # open the specified requirements.txt file in write mode within the data directory
    with open(f"data/{filename}", 'w') as f:
        # write each package to the file
        for package in installed_packages:
            f.write(package + '\n')

    print(f"Generated {filename} successfully in the data directory!")
    
def install_requirements(filepath='requirements.txt'):
    """
    Reads the packages listed in a `requirements.txt` file and installs them in the current environment.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    try:
        # read the packages listed in the requirements.txt file
        with open(f"data/{filepath}", 'r') as f:
            packages = f.read().splitlines()

        # install the packages using pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", *packages], capture_output=True)

        if result.returncode == 0:
            print("All packages already installed.")
        else:
            print("Installed packages successfully!")
    except FileNotFoundError:
        print("No requirements.txt file found. Continuing with the execution...")


    
