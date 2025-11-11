"""
Data loading functions with caching for the ML educational platform.
All functions use @st.cache_data for performance optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from pathlib import Path


@st.cache_data
def load_qsar_fish_toxicity():
    """
    Load QSAR fish toxicity dataset.

    Returns:
        pd.DataFrame: DataFrame with 6 molecular descriptors and LC50 target
    """
    data_path = Path(__file__).parent.parent / "data" / "qsar_fish_toxicity.csv"

    if data_path.exists():
        df = pd.read_csv(data_path, sep=';', header=None)
        df.columns = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP', 'LC50']
    else:
        # Generate synthetic QSAR-like data if file doesn't exist
        np.random.seed(42)
        n_samples = 908
        df = pd.DataFrame({
            'CIC0': np.random.uniform(0, 5, n_samples),
            'SM1_Dz(Z)': np.random.uniform(0, 1, n_samples),
            'GATS1i': np.random.uniform(0, 2, n_samples),
            'NdsCH': np.random.randint(0, 10, n_samples),
            'NdssC': np.random.randint(0, 15, n_samples),
            'MLOGP': np.random.uniform(-2, 6, n_samples),
        })
        # Create synthetic LC50 based on some descriptors
        df['LC50'] = (
            2.5 +
            0.5 * df['MLOGP'] -
            0.3 * df['CIC0'] +
            np.random.normal(0, 0.5, n_samples)
        )

    return df


@st.cache_data
def load_breast_cancer_data():
    """
    Load Breast Cancer Wisconsin dataset from scikit-learn.

    Returns:
        tuple: (DataFrame with features, Series with target, feature_names, target_names)
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    return X, y, data.feature_names, data.target_names


@st.cache_data
def load_gene_expression_cancer():
    """
    Load Gene Expression Cancer RNA-Seq dataset.

    Returns:
        tuple: (DataFrame with features, Series with target labels)
    """
    data_path = Path(__file__).parent.parent / "data" / "gene_expression_cancer_rna_seq.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        # Assume last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        # Generate synthetic gene expression data if file doesn't exist
        np.random.seed(42)
        n_samples = 801
        n_genes = 100  # Reduced from 20531 for demo purposes

        # Create 5 cancer types
        cancer_types = ['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD']
        y = np.random.choice(cancer_types, n_samples)

        # Generate synthetic gene expression data with some structure
        X = pd.DataFrame()
        for i in range(n_genes):
            # Add some signal based on cancer type
            gene_expr = []
            for cancer in y:
                base_expr = np.random.lognormal(0, 1)
                # Add cancer-specific signal for some genes
                if i < 20:  # First 20 genes have signal
                    type_signal = cancer_types.index(cancer) * 0.5
                    base_expr += type_signal
                gene_expr.append(base_expr)
            X[f'gene_{i}'] = gene_expr

        y = pd.Series(y, name='cancer_type')

    return X, y


@st.cache_data
def load_uploaded_data(uploaded_file):
    """
    Load data from uploaded CSV file with automatic separator detection.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Try comma separator first
        df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
    except Exception:
        try:
            # Try semicolon separator
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except Exception:
            try:
                # Try with ISO-8859-1 encoding
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='ISO-8859-1')
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None

    return df
