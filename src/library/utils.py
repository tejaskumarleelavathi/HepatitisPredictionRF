import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback as tb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from log.logger import get_logger

# Configure the logger
log = get_logger("utils")  # Specify the log filename


def load_data(file_path):
    """
    Load the dataset from a specified CSV file path.
    
    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    data = None
    try:
        data = pd.read_csv(file_path)
    except:
        log.error(f"Error loading data from {file_path}: {e}")
        log.error(tb.format_exc())
    return data


def preprocess_data(data):
    """
    Preprocess the dataset by mapping categorical values, filling missing values, 
    and renaming columns for readability.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: A preprocessed DataFrame.
    """
    preprocessed_data = None
    try:
        # Map 'Category' and 'Sex' columns to binary values
        data['Category'] = data['Category'].map({
            '0=Blood Donor': 0,
            '0s=suspect Blood Donor': 0,
            '1=Hepatitis': 1,
            '2=Fibrosis': 1,
            '3=Cirrhosis': 1
        })
        data['Sex'] = data['Sex'].map({'m': 1, 'f': 0})
        
        # Fill missing values with column medians
        data.fillna(data.median(), inplace=True)
        
        # Rename columns for readability
        column_names = {
            'ALB': 'Albumin Blood Test (ALB) g/L',
            'ALP': 'Alkaline Phosphatase Test (ALP) IU/L',
            'ALT': 'Alanine Transaminase Test (ALT) U/L',
            'AST': 'Aspartate Transaminase Test (AST) U/L',
            'BIL': 'Bilirubin Blood Test (BIL) µmol/L',
            'CHE': 'Cholinesterase (CHE) kU/L',
            'CHOL': 'Cholesterol (CHOL) mmol/L',
            'CREA': 'Creatinine Blood Test (CREA) µmol/L',
            'GGT': 'Gamma-Glutamyl Transpeptidase Test (GGT) U/L',
            'PROT': 'Protein Blood Test (PROT) g/L'
        }
        data.rename(columns=column_names, inplace=True)

        preprocessed_data = data
    except:
        log.error(f"Error during data preprocessing: {e}")
        log.error(tb.format_exc())
    return preprocessed_data


def balance_data_with_smote(data, k_neighbors=5, sampling_strategy="auto"):
    """
    Balance the dataset using SMOTE with custom parameters.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset.
    k_neighbors (int): Number of nearest neighbors to use for generating synthetic samples.
    sampling_strategy (str or float or dict): Strategy to determine the class distribution after oversampling.
    
    Returns:
    pd.DataFrame, pd.Series: Balanced features (X) and target (y) datasets.
    """
    balanced_X, balanced_y = None, None
    try:
        # Separate features and target
        X = data.drop(["Category"], axis=1)
        y = data["Category"]
        
        # Apply SMOTE with custom parameters
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        balanced_X, balanced_y = smote.fit_resample(X, y)

        log.info(f"SMOTE applied. Class distribution after balancing: {balanced_y.value_counts().to_dict()}")
    except Exception as e:
        log.error(f"Error during data balancing with SMOTE: {e}")
        log.error(tb.format_exc())
    return balanced_X, balanced_y



def feature_selection_with_smote(data, num_features=11, k_neighbors=5, sampling_strategy="auto"):
    """
    Balance the dataset using SMOTE and perform feature selection using ANOVA F-values.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    num_features (int): The number of top features to select.
    k_neighbors (int): Number of nearest neighbors to use for generating synthetic samples in SMOTE.
    sampling_strategy (str or float or dict): Strategy to determine the class distribution after oversampling.
    
    Returns:
    pd.DataFrame, pd.Series: Balanced and selected features (X) and target (y).
    """
    selected_X, selected_y = None, None
    try:
        # Balance data with SMOTE
        balanced_X, balanced_y = balance_data_with_smote(data, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)

        # Select features
        selector = SelectKBest(f_classif, k=num_features)
        X_selected = selector.fit_transform(balanced_X, balanced_y)

        # Get the names of the selected features
        selected_features = balanced_X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]

        # Create a DataFrame of feature scores
        feature_scores_df = pd.DataFrame({'Feature': selected_features, 'Score': feature_scores})
        feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

        # Plot the feature scores
        plot_feature_scores(feature_scores_df)

        # Return the selected features and balanced target
        selected_X = balanced_X[selected_features]
        selected_y = balanced_y

    except:
        log.error(f"Error during feature selection with SMOTE: {e}")
        log.error(tb.format_exc())
    
    return selected_X, selected_y



def plot_feature_scores(feature_scores_df):
    """
    Plot the feature scores in a bar plot.
    
    Parameters:
    feature_scores_df (pd.DataFrame): A DataFrame containing features and their scores.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score', y='Feature', hue='Feature', data=feature_scores_df, palette='viridis', dodge=False, legend=False)
        plt.title('Feature Scores')
        plt.xlabel('Scores')
        plt.ylabel('Features')
        plt.show()
    except:
        log.error(f"Error during feature score plotting: {e}")
        log.error(tb.format_exc())




