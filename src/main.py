from dotenv import load_dotenv
import os

from src.library.utils import load_data
from src.library.utils import preprocess_data
# Updated function with SMOTE integration
from src.library.utils import feature_selection_with_smote
from src.models.train import train_random_forest
from src.evaluation.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    data_path = os.getenv("DATA_PATH")

    # Load and preprocess data
    data = load_data(data_path)
    print("Columns in dataset:", data.columns)
    data = preprocess_data(data)

    # Feature selection with SMOTE integration
    X, y = feature_selection_with_smote(
        # Handles class imbalance
        data, num_features=11, k_neighbors=5, sampling_strategy="auto")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=45, stratify=y)

    # Train Random Forest model
    model_rnf = train_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(model_rnf, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
