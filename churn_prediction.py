import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
    # Fixed: pd.load_csv() to pd.read_csv()
    data = pd.read_csv(file_path)

    return data

def preprocess_data(data):
    """
    Preprocess the data (encode categorical variables, scale numerical features).
    
    Args:
        data (pd.DataFrame): Raw dataset.
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    # 1. apply one-hot encoding
    data = pd.get_dummies(data, columns=['contract_type'], drop_first=True) # Fixed: drop_first = True to drop the first column

    # 2. apply standardization
    numerical_transformation = StandardScaler()
    data[["age", "tenure","monthly_charges"]] = numerical_transformation.fit_transform(data[["age", "tenure","monthly_charges"]])

    #3. divide to X and Y
    X = data.drop(columns = ['churn'])
    y = data['churn']

    return X, y


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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

    # Fixed: renamed to models_dict to match return statement
    models_dict = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

    return models_dict, X_train, X_test, y_train, y_test



def evaluate_models(models_dict, X_train, X_test, y_train, y_test):
    """
    Evaluate models using accuracy and AUROC scores.
    """
    results = []
    
    for name, model in models_dict.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        auroc = roc_auc_score(y_test, y_test_prob)
        
        results.append({
            "Model": name,
            "Training Accuracy": train_acc,
            "Testing Accuracy": test_acc,
            "AUROC Score": auroc
        })
    
    return pd.DataFrame(results)

def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        y_test_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        plt.plot(fpr, tpr, label=name)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Churn Prediction Models")
    plt.legend()
    plt.show()

def plot_probability_distributions(models_dict, X_test):
    """
    Plot probability distributions of predictions.
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        probs = model.predict_proba(X_test)[:, 1]
        sns.kdeplot(probs, label=name, fill=True)
    
    plt.xlabel("Predicted Probability of Churn")
    plt.ylabel("Density")
    plt.title("Probability Distributions of Model Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":  # Fixed: "_main" to "_main_"
    data = load_data("customer_churn.csv")  # Fixed: backslash to forward slash for path
    X, y = preprocess_data(data)
    models_dict, X_train, X_test, y_train, y_test = train_models(X, y)
    results = evaluate_models(models_dict, X_train, X_test, y_train, y_test)
    print("Model Evaluation Metrics:")
    print(results)
    plot_roc_curves(models_dict, X_test, y_test)
    plot_probability_distributions(models_dict, X_test)