import matplotlib.pyplot as plt
import traceback as tb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from log.logger import get_logger


# Configure the logger
log = get_logger("evaluate")  # Specify the log filename


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model's performance using accuracy, classification report,
    and confusion matrix.

    Parameters:
    model (sklearn model): The trained model to evaluate.
    X_test (pd.DataFrame or np.array): Test features.
    y_test (pd.Series or np.array): True labels for the test data.

    Returns:
    None: Prints accuracy, classification report, and displays
    confusion matrix plot.

    Raises:
    Exception: If there is an error during model evaluation.
    """
    try:

        # Calculate training accuracy
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Training Accuracy: {train_accuracy}")

        # Predict labels for the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(" Testing Accuracy:", accuracy)

        # Print classification report
        # print("Classification Report:")
        # print(classification_report(y_test, y_pred))

        # Generate and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=cm,
                                        show_absolute=True,
                                        colorbar=True,
                                        cmap='cool',
                                        class_names=[True, False],
                                        figsize=(5, 3))
        plt.title("Confusion Matrix")
        plt.show()
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
    except Exception as e:
        log.error(f"Error during model evaluation: {e}")
        log.error(tb.format_exc())  # Log the full traceback to the log file
