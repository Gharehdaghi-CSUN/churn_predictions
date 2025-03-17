[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yGqBeinD)
# Customer Churn Prediction Assignment

## Overview
In this assignment, you will build and compare classification models to predict whether a customer will churn (leave the service) based on features like age, tenure, monthly charges, and contract type. You will use Python, scikit-learn, pandas, matplotlib, and seaborn to preprocess the data, train three different models (Logistic Regression, Decision Tree, and Random Forest), evaluate their performance, and visualize the results.




## Problem Statement
Customer churn is a critical issue for businesses, especially in subscription-based industries like telecommunications, streaming services, and financial services. Churn occurs when a customer discontinues their service, leading to loss of revenue and increased acquisition costs for replacing lost customers. The goal of this assignment is to build predictive models to identify customers at risk of churning based on their demographic and behavioral data, enabling proactive retention strategies.

## Dataset Description
A synthetic dataset (`customer_churn.csv`) is provided under data folder. The data include the following features:
- **age**: Age of the customer (integer, range: 18 to 80 years).
- **tenure**: Number of months the customer has been with the service (integer, range: 1 to 60 months).
- **monthly_charges**: Monthly bill amount for the customer (float, range: $20 to $120).
- **contract_type**: Type of contract the customer has (categorical: 'Month-to-Month', 'One-Year', 'Two-Year').
- **churn**: Target variable indicating whether the customer churned (binary: 0 = no churn, 1 = churn).


## Objective
The primary objective is to build and evaluate classification models to predict customer churn. Specifically:
- Develop and compare three classification models: Logistic Regression, Decision Tree, and Random Forest.
- Preprocess the data by encoding categorical variables and scaling numerical features.
- Evaluate model performance using accuracy and AUROC (Area Under the Receiver Operating Characteristic Curve) scores.
- Visualize model performance through ROC curves and probability distributions to understand prediction


## Tasks

1. **Data Preprocessing**:
   - Implement the `load_data` function in `churn_prediction.py` to load the dataset (`customer_churn.csv`).
   - Implement the `preprocess_data` function to:
     - Encode the categorical `contract_type` feature using one-hot encoding.
     - Scale numerical features (`age`, `tenure`, `monthly_charges`) using `StandardScaler`.
     - Split the data into features (X) and target (y).

2. **Model Training**:
   - Implement the `train_models` function to:
     - Split the data into training and testing sets (80-20 split).
     - Train three models: Logistic Regression, Decision Tree, and Random Forest.

3. **Model Evaluation**:
   - Implement the `evaluate_models` function to:
     - Compute accuracy and AUROC scores for each model.
     - Return a dictionary or DataFrame with the results.

4. **Visualization**:
   - Implement the `plot_roc_curves` function to plot ROC curves for all three models.
   - Implement the `plot_probability_distributions` function to plot the probability distributions of predictions.

5. **Testing**:
   - Run
