import traceback
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model using Grid Search for hyperparameter tuning.
    
    Parameters:
    X_train (pd.DataFrame or np.array): Training features.
    y_train (pd.Series or np.array): Training labels.
    
    Returns:
    sklearn model: The best estimator from GridSearchCV with the optimal hyperparameters.
    
    Raises:
    Exception: If an error occurs during model training or hyperparameter tuning.
    """
    try:
        # Initialize Random Forest classifier with a fixed random state
        clf_rnf = RandomForestClassifier(random_state=46)
        
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [3, 5, 7, 10],
            'max_depth': [2, 3, 4, 5, 6]
        }
        
        # Set up GridSearchCV with 6-fold cross-validation
        grid_forest = GridSearchCV(clf_rnf, param_grid, cv=6, n_jobs=-1)
        
        # Fit the grid search on the training data
        grid_forest.fit(X_train, y_train)
        
        # Output the best parameters
        print("Best parameters:", grid_forest.best_params_)
        
        # Return the best estimator
        best_model_rnf = grid_forest.best_estimator_
        return best_model_rnf
    except Exception as e:
        print(f"Error during Random Forest training with Grid Search: {e}")
        traceback.print_exc()
        return None
