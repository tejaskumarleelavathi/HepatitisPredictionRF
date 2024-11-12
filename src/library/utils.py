import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def load_data(file_path):
    """
    Load the dataset from a specified CSV file path.
    
    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    
    Raises:
    Exception: If there is an error while reading the file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        traceback.print_exc()
        return None


def preprocess_data(data):
    """
    Preprocess the dataset by mapping categorical values, filling missing values, 
    and renaming columns for readability.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: A preprocessed DataFrame.
    
    Raises:
    Exception: If an error occurs during preprocessing.
    """
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
        
        return data
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        traceback.print_exc()
        return None


def feature_selection(data, num_features=11):
    """
    Select the top features based on ANOVA F-values and display feature scores in a bar plot.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    num_features (int): The number of top features to select.
    
    Returns:
    Tuple[pd.DataFrame, pd.Series]: A tuple containing the selected feature DataFrame (X) 
                                    and target series (y).
    
    Raises:
    Exception: If an error occurs during feature selection.
    """
    try:
        X = data.drop(["Category"], axis=1)
        y = data["Category"]
        
        selector = SelectKBest(f_classif, k=num_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]

        # Create DataFrame of feature scores
        feature_scores_df = pd.DataFrame({'Feature': selected_features, 'Score': feature_scores})
        feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)
        
        # Plot the feature scores
        plot_feature_scores(feature_scores_df)
        
        return X[selected_features], y
    except Exception as e:
        print(f"Error during feature selection: {e}")
        traceback.print_exc()
        return None, None


def plot_feature_scores(feature_scores_df):
    """
    Plot the feature scores in a bar plot.
    
    Parameters:
    feature_scores_df (pd.DataFrame): A DataFrame containing features and their scores.
    
    Raises:
    Exception: If an error occurs during plotting.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score', y='Feature', data=feature_scores_df, palette='viridis')
        plt.title('Feature Scores')
        plt.xlabel('Scores')
        plt.ylabel('Features')
        plt.show()
    except Exception as e:
        print(f"Error during feature score plotting: {e}")
        traceback.print_exc()
