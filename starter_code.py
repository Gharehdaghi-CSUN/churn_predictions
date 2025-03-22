import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # TODO: Load the dataset
    pass

def preprocess_data(data):
    """
    Preprocess the data (encode categorical variables, scale numerical features).
    
    Args:
        data (pd.DataFrame): Raw dataset.
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    # TODO: Implement preprocessing steps
    pass

def train_models(X, y):
    """
    Split the data into train/test sets and train all three models.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        
    Returns:
        tuple: (models_dict, X_train, X_test, y_train, y_test)
        where models_dict contains the trained models.
    """
    # TODO: Implement train-test split and model training
    pass

def evaluate_models(models_dict, X_train, X_test, y_train, y_test):
    """
    Evaluate all three models using accuracy and AUROC scores.
    
    Args:
        models_dict (dict): Dictionary of trained models.
        X_train, X_test, y_train, y_test: Training and testing data.
        
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    # TODO: Implement evaluation
    pass

def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for all three models.
    
    Args:
        models_dict (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.
    """
    # TODO: Implement ROC curve plotting
    pass

def plot_probability_distributions(models_dict, X_test):
    """
    Plot probability distributions of predictions for all three models.
    
    Args:
        models_dict (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test feature matrix.
    """
    # TODO: Implement probability distribution plotting
    pass

if __name__ == "__main__":
    # Example usage (students can test their implementation here)
    data = load_data("customer_churn.csv")
    X, y = preprocess_data(data)
    models_dict, X_train, X_test, y_train, y_test = train_models(X, y)
    results = evaluate_models(models_dict, X_train, X_test, y_train, y_test)
    print("Model Evaluation Metrics:")
    print(results)
    plot_roc_curves(models_dict, X_test, y_test)
    plot_probability_distributions(models_dict, X_test)
