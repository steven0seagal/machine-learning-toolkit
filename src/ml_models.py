"""
Machine learning model wrapper functions for the educational platform.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Clustering
from sklearn.cluster import KMeans

# Dimensionality reduction
from sklearn.decomposition import PCA


def build_pipeline(model, use_imputer=False, imputer_strategy='mean',
                   use_scaler=False):
    """
    Build a sklearn pipeline with optional preprocessing steps.

    Args:
        model: sklearn estimator
        use_imputer: Whether to include imputation
        imputer_strategy: Strategy for SimpleImputer ('mean', 'median', 'most_frequent')
        use_scaler: Whether to include StandardScaler

    Returns:
        sklearn.pipeline.Pipeline
    """
    steps = []

    if use_imputer:
        steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))

    if use_scaler:
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', model))

    return Pipeline(steps)


def get_regression_model(model_name, **params):
    """
    Get a regression model by name with specified parameters.

    Args:
        model_name: Name of the model
        **params: Model-specific parameters

    Returns:
        sklearn estimator
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(**params),
        'Lasso': Lasso(**params),
    }

    return models.get(model_name, LinearRegression())


def get_classification_model(model_name, **params):
    """
    Get a classification model by name with specified parameters.

    Args:
        model_name: Name of the model
        **params: Model-specific parameters

    Returns:
        sklearn estimator
    """
    if model_name == 'Logistic Regression':
        return LogisticRegression(**params)
    elif model_name == 'k-NN':
        return KNeighborsClassifier(**params)
    elif model_name == 'SVM':
        return SVC(**params, probability=True)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(**params)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(**params)
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(**params)
    else:
        return LogisticRegression()


def get_clustering_model(model_name, **params):
    """
    Get a clustering model by name with specified parameters.

    Args:
        model_name: Name of the model
        **params: Model-specific parameters

    Returns:
        sklearn estimator
    """
    if model_name == 'K-Means':
        return KMeans(**params)
    else:
        return KMeans(**params)


def evaluate_regression(y_true, y_pred):
    """
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        dict: Dictionary of metrics
    """
    return {
        'R²': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }


def evaluate_classification(y_true, y_pred, average='binary'):
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multiclass ('binary', 'weighted', 'macro')

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
    }

    # Handle binary vs multiclass
    if len(np.unique(y_true)) == 2:
        metrics['Precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['Recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['F1-Score'] = f1_score(y_true, y_pred, average='binary')
    else:
        metrics['Precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['F1-Score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def evaluate_clustering(X, labels):
    """
    Calculate clustering metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}

    if len(np.unique(labels)) > 1:
        metrics['Silhouette Score'] = silhouette_score(X, labels)

    return metrics


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
