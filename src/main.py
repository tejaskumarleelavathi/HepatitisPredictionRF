from library.utils import load_data
from library.utils import preprocess_data
from library.utils import feature_selection
from models.train import train_random_forest
from evaluation.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Define file path
    data_path = '/Users/tejaskl/Documents/MLProject/HepatitisPrediction/data/HepatitisCdata.csv'

    # Load and preprocess data
    data = load_data(data_path)
    print("Columns in dataset:", data.columns)
    data = preprocess_data(data)

    # Feature selection
    X, y = feature_selection(data, num_features=11)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest model
    best_model_rnf = train_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(best_model_rnf, X_test, y_test)

if __name__ == "__main__":
    main()
