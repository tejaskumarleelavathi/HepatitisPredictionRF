from library.utils import load_data
from library.utils import preprocess_data
from library.utils import feature_selection_with_smote  # Updated function with SMOTE integration
from models.train import train_random_forest
from evaluation.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from config import DATA_PATH 

def main():
    # Define file path
    data_path = DATA_PATH

    # Load and preprocess data
    data = load_data(data_path)
    print("Columns in dataset:", data.columns)
    data = preprocess_data(data)

    # Feature selection with SMOTE integration
    X, y = feature_selection_with_smote(data, num_features=11, k_neighbors=5, sampling_strategy="auto")  # Handles class imbalance

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45, stratify=y)

    # Train Random Forest model
    model_rnf = train_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(model_rnf, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
