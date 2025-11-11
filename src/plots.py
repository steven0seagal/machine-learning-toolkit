"""
Plotly visualization helper functions for the ML educational platform.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary_2d(model, X, y, feature_names, resolution=0.02, padding=0.5):
    """
    Create a 2D decision boundary plot using Plotly.

    Args:
        model: Trained sklearn classifier
        X: Feature matrix (n_samples, 2)
        y: Target labels
        feature_names: List of 2 feature names
        resolution: Mesh resolution
        padding: Padding around data points

    Returns:
        plotly.graph_objects.Figure
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Add contour for decision boundary
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale='RdBu',
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    ))

    # Add scatter points
    for class_value in np.unique(y):
        mask = y == class_value
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_value}',
            marker=dict(size=8, line=dict(width=1, color='white'))
        ))

    fig.update_layout(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        hovermode='closest',
        width=700,
        height=500
    )

    return fig


def plot_probability_boundary_2d(model, X, y, target_names, feature_names, resolution=0.02, padding=0.5):
    """
    Create a 2D probability boundary plot for classification.

    Args:
        model: Trained sklearn classifier with predict_proba method
        X: Feature matrix (n_samples, 2)
        y: Target labels
        target_names: Names of target classes
        feature_names: List of 2 feature names
        resolution: Mesh resolution
        padding: Padding around data points

    Returns:
        plotly.graph_objects.Figure
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Add contour for probability
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale='RdYlBu_r',
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="P(Class 1)"),
        hovertemplate='%{z:.2f}<extra></extra>'
    ))

    # Add scatter points
    for i, class_value in enumerate(np.unique(y)):
        mask = y == class_value
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=target_names[i] if target_names is not None else f'Class {class_value}',
            marker=dict(
                size=8,
                line=dict(width=1, color='white'),
                color='red' if i == 0 else 'blue'
            )
        ))

    fig.update_layout(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        hovermode='closest',
        width=700,
        height=500
    )

    return fig


def plot_feature_importance(importances, feature_names, top_n=20):
    """
    Create a horizontal bar chart of feature importances.

    Args:
        importances: Array of feature importance scores
        feature_names: List of feature names
        top_n: Number of top features to display

    Returns:
        plotly.graph_objects.Figure
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importances',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )

    fig.update_layout(
        height=max(400, top_n * 20),
        showlegend=False
    )

    return fig


def plot_confusion_matrix(cm, class_names):
    """
    Create an interactive confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2D array)
        class_names: List of class names

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )

    return fig


def plot_pca_scree(explained_variance_ratio, cumulative=True):
    """
    Create a scree plot for PCA explained variance.

    Args:
        explained_variance_ratio: Array of explained variance ratios
        cumulative: Whether to show cumulative variance

    Returns:
        plotly.graph_objects.Figure
    """
    n_components = len(explained_variance_ratio)
    components = list(range(1, n_components + 1))

    if cumulative:
        variance = np.cumsum(explained_variance_ratio) * 100
        title = 'Cumulative Explained Variance'
        yaxis_title = 'Cumulative Variance Explained (%)'
    else:
        variance = explained_variance_ratio * 100
        title = 'Explained Variance per Component'
        yaxis_title = 'Variance Explained (%)'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=components,
        y=variance,
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2)
    ))

    if cumulative:
        # Add 90% reference line
        fig.add_hline(y=90, line_dash="dash", line_color="red",
                      annotation_text="90% threshold")

    fig.update_layout(
        title=title,
        xaxis_title='Principal Component',
        yaxis_title=yaxis_title,
        width=700,
        height=400
    )

    return fig


def plot_elbow_curve(inertias, k_range):
    """
    Create an elbow plot for K-Means.

    Args:
        inertias: List of inertia values
        k_range: Range of k values

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2)
    ))

    fig.update_layout(
        title='Elbow Method For Optimal k',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia (Within-Cluster Sum of Squares)',
        width=600,
        height=400
    )

    return fig


def plot_silhouette_scores(silhouette_scores, k_range):
    """
    Create a plot of silhouette scores for different k values.

    Args:
        silhouette_scores: List of silhouette scores
        k_range: Range of k values

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2)
    ))

    # Mark the maximum
    max_idx = np.argmax(silhouette_scores)
    fig.add_trace(go.Scatter(
        x=[list(k_range)[max_idx]],
        y=[silhouette_scores[max_idx]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Optimal k'
    ))

    fig.update_layout(
        title='Silhouette Score vs. Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        width=600,
        height=400
    )

    return fig
