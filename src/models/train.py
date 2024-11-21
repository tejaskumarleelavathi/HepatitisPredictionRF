from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
import traceback as tb
from log.logger import get_logger


# Configure the logger
log = get_logger("train")  # Specify the log filename


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model using Grid Search for hyperparameter tuning.

    Parameters:
    X_train (pd.DataFrame or np.array): Training features.
    y_train (pd.Series or np.array): Training labels.

    Returns:
    sklearn model: The best estimator from GridSearchCV with the optimal
    hyperparameters.

    Raises:
    Exception: If an error occurs during model training
    or hyperparameter tuning.
    """
    best_model_rnf = None

    try:
        # Initialize Random Forest classifier with a fixed random state
        clf_rnf = RandomForestClassifier(random_state=46)

        # Define an expanded hyperparameter grid for tuning
        param_grid = {
            # Number of trees in forest
            'n_estimators': [
                int(x) for x in np.linspace(start=10, stop=80, num=10)],
            # Number of features to consider at every split
            'max_features': ['log2', 'sqrt'],
            'max_depth': [2, 4],  # Maximum depth of the tree
            # Minimum number of samples required to split a node
            'min_samples_split': [2, 5],
            # Minimum number of samples required at each leaf node
            'min_samples_leaf': [1, 2],
            # Method of selecting samples for training each tree
            'bootstrap': [True, False]
        }

        # Set up GridSearchCV with 6-fold cross-validation
        grid_forest = GridSearchCV(clf_rnf, param_grid, cv=6, n_jobs=-1)

        # Fit the grid search on the training data
        grid_forest.fit(X_train, y_train)

        # Output the best parameters
        print("Best parameters:", grid_forest.best_params_)

        # Assign the best estimator
        best_model_rnf = grid_forest.best_estimator_

    except Exception as e:
        log.error(f"Error during Random Forest training with Grid Search: {e}")
        log.error(tb.format_exc())  # Log the full traceback to the log file

    return best_model_rnf
