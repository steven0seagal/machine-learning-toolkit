"""
Analizuj Własne Dane - Bring Your Own Data (BYOD) Universal Analysis Tool
Upload your CSV and run any ML algorithm!
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_uploaded_data
from src.ml_models import (
    build_pipeline, get_regression_model, get_classification_model, get_clustering_model,
    evaluate_regression, evaluate_classification, evaluate_clustering, train_test_split_data
)
from src.plots import plot_confusion_matrix, plot_feature_importance, plot_elbow_curve, plot_silhouette_scores

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from src.navigation import render_sidebar_navigation

st.set_page_config(page_title="Analizuj Własne Dane", page_icon="📂", layout="wide")

# Render sidebar navigation
render_sidebar_navigation()

st.title("📂 Analizuj Własne Dane (BYOD)")

st.markdown("""
## Uniwersalne narzędzie do analizy Machine Learning

Prześlij swój plik CSV i przeprowadź pełną analizę ML w 5 krokach:

1. 📤 **Upload Pliku** - Prześlij swoje dane w formacie CSV
2. 🎯 **Definicja Zmiennych** - Wybierz zmienną docelową (target) i cechy (features)
3. 🔧 **Preprocessing** - Uzupełnianie braków, skalowanie
4. 🤖 **Model i Trening** - Wybierz algorytm i hiperparametry
5. 📊 **Wyniki** - Metryki, wizualizacje, pobieranie wyników

---
""")

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("1️⃣ Upload Pliku CSV")

st.markdown("""
**Wymagania:**
- Format: CSV (separatory: `,` lub `;`)
- Kolumny: Jedna kolumna target + wiele kolumn features
- Dane numeryczne (lub kategoryczne - zostaną zakodowane)
""")

uploaded_file = st.file_uploader(
    "Wybierz plik CSV",
    type=['csv'],
    help="Prześlij plik CSV z danymi do analizy"
)

if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)

        if df is not None:
            st.session_state.df = df
            st.session_state.data_uploaded = True

            st.success(f"✅ Plik wczytany pomyślnie! {df.shape[0]} wierszy × {df.shape[1]} kolumn")

            # Preview data
            with st.expander("👀 Podgląd danych (pierwsze 10 wierszy)"):
                st.dataframe(df.head(10))

            # Basic statistics
            with st.expander("📊 Statystyki opisowe"):
                st.dataframe(df.describe())

            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                with st.expander("⚠️ Braki w danych"):
                    missing_df = pd.DataFrame({
                        'Kolumna': missing.index,
                        'Liczba braków': missing.values,
                        'Procent braków': (missing.values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Liczba braków'] > 0]
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                    st.warning("⚠️ Wykryto braki w danych. W kroku 3 (Preprocessing) możesz je uzupełnić.")

        else:
            st.error("Nie udało się wczytać pliku. Sprawdź format CSV.")

    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {str(e)}")
        st.session_state.data_uploaded = False

else:
    st.info("👆 Prześlij plik CSV aby rozpocząć analizę")
    st.session_state.data_uploaded = False

# ============================================================================
# STEP 2: VARIABLE DEFINITION
# ============================================================================

if st.session_state.data_uploaded:
    st.markdown("---")
    st.header("2️⃣ Definicja Zmiennych")

    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Zmienna Docelowa (Target)")

        task_type = st.radio(
            "Typ zadania:",
            options=['Klasyfikacja', 'Regresja', 'Klastrowanie', 'PCA (Redukcja wymiarowości)'],
            help="Wybierz typ analizy ML"
        )

        if task_type in ['Klasyfikacja', 'Regresja']:
            target_column = st.selectbox(
                "Wybierz kolumnę target:",
                options=df.columns.tolist(),
                help="Kolumna którą chcesz przewidywać"
            )

            # Show target distribution
            if target_column:
                st.markdown(f"**Rozkład zmiennej `{target_column}`:**")

                if task_type == 'Klasyfikacja':
                    target_counts = df[target_column].value_counts()
                    fig_target = px.bar(
                        x=target_counts.index.astype(str),
                        y=target_counts.values,
                        labels={'x': target_column, 'y': 'Liczba'},
                        title=f'Rozkład klas: {target_column}'
                    )
                    st.plotly_chart(fig_target, use_container_width=True)

                    n_classes = len(target_counts)
                    st.info(f"Liczba klas: **{n_classes}**")

                else:  # Regression
                    fig_target = px.histogram(
                        df,
                        x=target_column,
                        nbins=30,
                        title=f'Rozkład: {target_column}'
                    )
                    st.plotly_chart(fig_target, use_container_width=True)

                    st.info(f"Min: {df[target_column].min():.2f}, Max: {df[target_column].max():.2f}, Mean: {df[target_column].mean():.2f}")

        else:
            target_column = None
            st.info("Klastrowanie i PCA nie wymagają zmiennej target (unsupervised learning)")

    with col2:
        st.markdown("### 📊 Cechy (Features)")

        available_features = [col for col in df.columns if col != target_column] if target_column else df.columns.tolist()

        select_all_features = st.checkbox("Wybierz wszystkie kolumny jako features", value=True)

        if select_all_features:
            feature_columns = available_features
        else:
            feature_columns = st.multiselect(
                "Wybierz kolumny features:",
                options=available_features,
                default=available_features,
                help="Kolumny używane do predykcji"
            )

        if len(feature_columns) > 0:
            st.success(f"✅ Wybrano **{len(feature_columns)}** cech")

            # Show feature types
            with st.expander("🔍 Typy danych cech"):
                feature_types = df[feature_columns].dtypes
                type_df = pd.DataFrame({
                    'Cecha': feature_types.index,
                    'Typ': feature_types.values.astype(str)
                })
                st.dataframe(type_df, use_container_width=True, hide_index=True)

                non_numeric = [col for col in feature_columns if df[col].dtype == 'object']
                if len(non_numeric) > 0:
                    st.warning(f"⚠️ Kolumny tekstowe zostaną zakodowane: {', '.join(non_numeric)}")
        else:
            st.warning("Wybierz przynajmniej jedną cechę!")

    # Store in session state
    if len(feature_columns) > 0:
        st.session_state.task_type = task_type
        st.session_state.target_column = target_column
        st.session_state.feature_columns = feature_columns
        st.session_state.variables_defined = True
    else:
        st.session_state.variables_defined = False

# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================

if st.session_state.data_uploaded and st.session_state.get('variables_defined', False):
    st.markdown("---")
    st.header("3️⃣ Preprocessing")

    df = st.session_state.df
    feature_columns = st.session_state.feature_columns
    target_column = st.session_state.target_column
    task_type = st.session_state.task_type

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 Uzupełnianie Braków")

        missing_in_features = df[feature_columns].isnull().sum().sum()

        if missing_in_features > 0:
            st.warning(f"⚠️ Wykryto **{missing_in_features}** braków w cechach")

            use_imputer = st.checkbox("Uzupełnij braki", value=True)

            if use_imputer:
                imputer_strategy = st.selectbox(
                    "Strategia uzupełniania:",
                    options=['mean', 'median', 'most_frequent'],
                    help="mean: średnia, median: mediana, most_frequent: najczęstsza wartość"
                )
            else:
                imputer_strategy = None
                st.info("Wiersze z brakami zostaną usunięte")
        else:
            st.success("✅ Brak braków w danych")
            use_imputer = False
            imputer_strategy = None

    with col2:
        st.markdown("### 📏 Skalowanie Cech")

        use_scaler = st.checkbox(
            "Standaryzuj cechy (StandardScaler)",
            value=(task_type in ['Klastrowanie', 'PCA (Redukcja wymiarowości)']),
            help="Rekomendowane dla: k-NN, SVM, Klastrowanie, PCA"
        )

        if use_scaler:
            st.info("Cechy zostaną znormalizowane: mean=0, std=1")

    st.session_state.use_imputer = use_imputer
    st.session_state.imputer_strategy = imputer_strategy
    st.session_state.use_scaler = use_scaler
    st.session_state.preprocessing_defined = True

# ============================================================================
# STEP 4: MODEL SELECTION AND TRAINING
# ============================================================================

if st.session_state.data_uploaded and st.session_state.get('preprocessing_defined', False):
    st.markdown("---")
    st.header("4️⃣ Wybór Modelu i Trening")

    task_type = st.session_state.task_type

    # Model selection based on task type
    if task_type == 'Regresja':
        st.markdown("### 📈 Algorytmy Regresji")

        with st.form("regression_form"):
            algorithm = st.selectbox(
                "Wybierz algorytm:",
                options=['Linear Regression'],
                help="Linear Regression: Podstawowy model regresji liniowej"
            )

            test_size = st.slider(
                "Rozmiar zbioru testowego (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100

            submit_button = st.form_submit_button("🚀 Trenuj Model")

            if submit_button:
                st.session_state.algorithm = algorithm
                st.session_state.test_size = test_size
                st.session_state.model_params = {}
                st.session_state.start_training = True

    elif task_type == 'Klasyfikacja':
        st.markdown("### 🎯 Algorytmy Klasyfikacji")

        with st.form("classification_form"):
            algorithm = st.selectbox(
                "Wybierz algorytm:",
                options=['Logistic Regression', 'k-NN', 'SVM', 'Decision Tree', 'Random Forest'],
                help="Wybierz algorytm klasyfikacji"
            )

            # Algorithm-specific hyperparameters
            model_params = {}

            if algorithm == 'Logistic Regression':
                C = st.slider("C (Regularization):", 0.01, 10.0, 1.0, step=0.1)
                model_params['C'] = C
                model_params['max_iter'] = 1000

            elif algorithm == 'k-NN':
                n_neighbors = st.slider("k (Number of neighbors):", 1, 30, 5)
                model_params['n_neighbors'] = n_neighbors

            elif algorithm == 'SVM':
                kernel = st.selectbox("Kernel:", ['rbf', 'linear', 'poly'])
                C = st.slider("C (Regularization):", 0.01, 10.0, 1.0, step=0.1)
                model_params['kernel'] = kernel
                model_params['C'] = C

            elif algorithm == 'Decision Tree':
                max_depth = st.slider("Max Depth:", 1, 20, 5)
                criterion = st.selectbox("Criterion:", ['gini', 'entropy'])
                model_params['max_depth'] = max_depth
                model_params['criterion'] = criterion

            elif algorithm == 'Random Forest':
                n_estimators = st.slider("Number of Trees:", 10, 500, 100, step=10)
                max_depth_option = st.selectbox("Max Depth:", ['None', '5', '10', '20'])
                max_depth = None if max_depth_option == 'None' else int(max_depth_option)
                model_params['n_estimators'] = n_estimators
                model_params['max_depth'] = max_depth

            test_size = st.slider(
                "Rozmiar zbioru testowego (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100

            submit_button = st.form_submit_button("🚀 Trenuj Model")

            if submit_button:
                st.session_state.algorithm = algorithm
                st.session_state.test_size = test_size
                st.session_state.model_params = model_params
                st.session_state.start_training = True

    elif task_type == 'Klastrowanie':
        st.markdown("### 🎯 Algorytmy Klastrowania")

        with st.form("clustering_form"):
            algorithm = st.selectbox(
                "Wybierz algorytm:",
                options=['K-Means'],
                help="K-Means: Najpopularniejszy algorytm klastrowania"
            )

            n_clusters = st.slider("Liczba klastrów (k):", 2, 10, 3)

            submit_button = st.form_submit_button("🚀 Wykonaj Klastrowanie")

            if submit_button:
                st.session_state.algorithm = algorithm
                st.session_state.n_clusters = n_clusters
                st.session_state.start_training = True

    elif task_type == 'PCA (Redukcja wymiarowości)':
        st.markdown("### 📐 PCA - Principal Component Analysis")

        with st.form("pca_form"):
            n_components = st.slider(
                "Liczba głównych składowych:",
                2,
                min(20, len(st.session_state.feature_columns)),
                min(10, len(st.session_state.feature_columns))
            )

            submit_button = st.form_submit_button("🚀 Wykonaj PCA")

            if submit_button:
                st.session_state.algorithm = 'PCA'
                st.session_state.n_components = n_components
                st.session_state.start_training = True

# ============================================================================
# STEP 5: TRAINING AND RESULTS
# ============================================================================

if st.session_state.get('start_training', False):
    st.markdown("---")
    st.header("5️⃣ Wyniki Analizy")

    df = st.session_state.df
    feature_columns = st.session_state.feature_columns
    target_column = st.session_state.target_column
    task_type = st.session_state.task_type
    algorithm = st.session_state.algorithm
    use_imputer = st.session_state.use_imputer
    imputer_strategy = st.session_state.imputer_strategy
    use_scaler = st.session_state.use_scaler

    try:
        with st.spinner(f'Trenowanie modelu {algorithm}...'):
            # Prepare data
            X = df[feature_columns].copy()

            # Encode categorical features
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

            # Handle target
            if target_column:
                y = df[target_column].copy()

                # Encode categorical target for classification
                if task_type == 'Klasyfikacja' and y.dtype == 'object':
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))
                    target_classes = target_encoder.classes_
                else:
                    target_encoder = None
                    target_classes = None
            else:
                y = None

            # Handle missing values
            if use_imputer and X.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy=imputer_strategy)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            else:
                # Drop rows with missing values
                if X.isnull().sum().sum() > 0:
                    mask = ~X.isnull().any(axis=1)
                    if target_column:
                        mask = mask & ~y.isnull()
                    X = X[mask]
                    if target_column:
                        y = y[mask]

            # Scaling
            if use_scaler:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)

            # ============================================================================
            # REGRESSION
            # ============================================================================
            if task_type == 'Regresja':
                X_train, X_test, y_train, y_test = train_test_split_data(
                    X, y, test_size=st.session_state.test_size
                )

                model = get_regression_model(algorithm)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_metrics = evaluate_regression(y_train, y_train_pred)
                test_metrics = evaluate_regression(y_test, y_test_pred)

                # Display results
                st.success(f"✅ Model {algorithm} wytrenowany pomyślnie!")

                st.subheader("📊 Metryki Wydajności")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train R²", f"{train_metrics['R²']:.4f}")
                with col2:
                    st.metric("Test R²", f"{test_metrics['R²']:.4f}")
                with col3:
                    st.metric("Test MAE", f"{test_metrics['MAE']:.4f}")
                with col4:
                    st.metric("Test RMSE", f"{test_metrics['RMSE']:.4f}")

                # Visualization
                st.subheader("📈 Predykcje vs Rzeczywiste Wartości")

                fig = px.scatter(
                    x=y_test,
                    y=y_test_pred,
                    labels={'x': 'Rzeczywiste', 'y': 'Przewidywane'},
                    title='Test Set: Przewidywane vs Rzeczywiste'
                )
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Idealna predykcja',
                    line=dict(dash='dash', color='red')
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Residuals
                with st.expander("📊 Analiza Reszt"):
                    residuals = y_test - y_test_pred
                    fig_res = px.scatter(
                        x=y_test_pred,
                        y=residuals,
                        labels={'x': 'Przewidywane', 'y': 'Reszty'},
                        title='Wykres Reszt'
                    )
                    fig_res.add_hline(y=0, line_dash='dash', line_color='red')
                    st.plotly_chart(fig_res, use_container_width=True)

            # ============================================================================
            # CLASSIFICATION
            # ============================================================================
            elif task_type == 'Klasyfikacja':
                X_train, X_test, y_train, y_test = train_test_split_data(
                    X, y, test_size=st.session_state.test_size
                )

                model = get_classification_model(algorithm, **st.session_state.model_params)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Determine if binary or multiclass
                n_classes = len(np.unique(y))
                average = 'binary' if n_classes == 2 else 'weighted'

                train_metrics = evaluate_classification(y_train, y_train_pred, average=average)
                test_metrics = evaluate_classification(y_test, y_test_pred, average=average)

                # Display results
                st.success(f"✅ Model {algorithm} wytrenowany pomyślnie!")

                st.subheader("📊 Metryki Wydajności")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Accuracy", f"{train_metrics['Accuracy']:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{test_metrics['Accuracy']:.4f}")
                with col3:
                    st.metric("Test Precision", f"{test_metrics['Precision']:.4f}")
                with col4:
                    st.metric("Test Recall", f"{test_metrics['Recall']:.4f}")

                # Confusion Matrix
                st.subheader("📊 Macierz Pomyłek")

                cm = test_metrics['Confusion Matrix']
                class_names = target_classes if target_classes is not None else [str(i) for i in range(n_classes)]
                fig_cm = plot_confusion_matrix(cm, class_names)
                st.plotly_chart(fig_cm, use_container_width=True)

                # Feature Importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🔍 Feature Importances")
                    fig_imp = plot_feature_importance(
                        model.feature_importances_,
                        feature_columns,
                        top_n=min(20, len(feature_columns))
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Decision Tree visualization
                if algorithm == 'Decision Tree':
                    with st.expander("🌳 Wizualizacja Drzewa Decyzyjnego"):
                        fig_tree, ax = plt.subplots(figsize=(20, 10))
                        plot_tree(
                            model,
                            max_depth=4,
                            feature_names=feature_columns,
                            class_names=class_names,
                            filled=True,
                            rounded=True,
                            fontsize=10,
                            ax=ax
                        )
                        plt.tight_layout()
                        st.pyplot(fig_tree)

            # ============================================================================
            # CLUSTERING
            # ============================================================================
            elif task_type == 'Klastrowanie':
                model = get_clustering_model(algorithm, n_clusters=st.session_state.n_clusters, random_state=42)
                cluster_labels = model.fit_predict(X)

                metrics = evaluate_clustering(X, cluster_labels)

                st.success(f"✅ Klastrowanie {algorithm} zakończone pomyślnie!")

                st.subheader("📊 Metryki Klastrowania")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inertia", f"{model.inertia_:.2f}")
                with col2:
                    if 'Silhouette Score' in metrics:
                        st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
                with col3:
                    st.metric("Liczba Klastrów", st.session_state.n_clusters)

                # Visualization with PCA
                st.subheader("🎨 Wizualizacja Klastrów (PCA 2D)")

                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)

                df_plot = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': cluster_labels.astype(str)
                })

                fig_cluster = px.scatter(
                    df_plot,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title=f'Klastrowanie {algorithm} (k={st.session_state.n_clusters})',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

                # Elbow plot
                with st.expander("📉 Elbow Plot & Silhouette Analysis"):
                    k_range = range(2, min(11, len(X)))
                    inertias = []
                    silhouettes = []

                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42)
                        labels = km.fit_predict(X)
                        inertias.append(km.inertia_)
                        if k > 1:
                            silhouettes.append(silhouette_score(X, labels))

                    col_elbow1, col_elbow2 = st.columns(2)

                    with col_elbow1:
                        fig_elbow = plot_elbow_curve(inertias, k_range)
                        st.plotly_chart(fig_elbow, use_container_width=True)

                    with col_elbow2:
                        fig_sil = plot_silhouette_scores(silhouettes, list(k_range)[1:])
                        st.plotly_chart(fig_sil, use_container_width=True)

                # Cluster sizes
                cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
                st.markdown("**Rozmiary klastrów:**")
                for cluster_id, size in cluster_sizes.items():
                    st.markdown(f"- **Klaster {cluster_id}**: {size} próbek ({size/len(cluster_labels)*100:.1f}%)")

            # ============================================================================
            # PCA
            # ============================================================================
            elif task_type == 'PCA (Redukcja wymiarowości)':
                pca = PCA(n_components=st.session_state.n_components, random_state=42)
                X_pca = pca.fit_transform(X)

                st.success("✅ PCA zakończone pomyślnie!")

                st.subheader("📊 Explained Variance")

                cumsum = np.cumsum(pca.explained_variance_ratio_)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
                with col2:
                    st.metric("PC2 Variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
                with col3:
                    st.metric(f"Cumulative ({st.session_state.n_components} PC)", f"{cumsum[-1]*100:.1f}%")

                # Scree plot
                st.subheader("📉 Scree Plot")
                from src.plots import plot_pca_scree
                fig_scree = plot_pca_scree(pca.explained_variance_ratio_, cumulative=True)
                st.plotly_chart(fig_scree, use_container_width=True)

                # 2D Visualization
                st.subheader("🎨 Wizualizacja PCA (PC1 vs PC2)")

                df_pca = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1]
                })

                if target_column:
                    df_pca['Target'] = df[target_column].values

                    fig_pca = px.scatter(
                        df_pca,
                        x='PC1',
                        y='PC2',
                        color='Target',
                        title='PCA Visualization',
                        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
                    )
                else:
                    fig_pca = px.scatter(
                        df_pca,
                        x='PC1',
                        y='PC2',
                        title='PCA Visualization',
                        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
                    )

                st.plotly_chart(fig_pca, use_container_width=True)

                # Download transformed data
                st.subheader("💾 Pobierz Przekształcone Dane")

                df_pca_full = pd.DataFrame(
                    X_pca,
                    columns=[f'PC{i+1}' for i in range(st.session_state.n_components)]
                )

                if target_column:
                    df_pca_full.insert(0, target_column, df[target_column].values)

                csv = df_pca_full.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Pobierz dane PCA (CSV)",
                    data=csv,
                    file_name='pca_transformed_data.csv',
                    mime='text/csv'
                )

            # ============================================================================
            # DOWNLOAD RESULTS
            # ============================================================================
            if task_type in ['Regresja', 'Klasyfikacja']:
                st.markdown("---")
                st.subheader("💾 Pobierz Wyniki")

                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Index': range(len(y_test)),
                    'Actual': y_test.values,
                    'Predicted': y_test_pred
                })

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Pobierz predykcje (CSV)",
                    data=csv,
                    file_name=f'{algorithm.lower().replace(" ", "_")}_predictions.csv',
                    mime='text/csv'
                )

            st.session_state.model_trained = True
            st.session_state.start_training = False

    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu: {str(e)}")
        st.exception(e)
        st.session_state.start_training = False

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 💡 Wskazówki do użycia:

1. **Przygotuj dane**:
   - Upewnij się że plik CSV ma nagłówki kolumn
   - Usuń kolumny ID/indeksy (lub nie wybieraj ich jako features)
   - Jedna kolumna = target, pozostałe = features

2. **Preprocessing**:
   - **Zawsze** użyj StandardScaler dla k-NN, SVM, Klastrowanie, PCA
   - Uzupełnij braki danych (mean dla cech numerycznych, most_frequent dla kategorycznych)

3. **Wybór algorytmu**:
   - **Regresja**: Linear Regression dla prostych, liniowych zależności
   - **Klasyfikacja binarna**: Logistic Regression, SVM (RBF)
   - **Klasyfikacja wieloklasowa**: Random Forest, SVM
   - **Klastrowanie**: K-Means (sprawdź Elbow plot dla optymalnego k)
   - **Wizualizacja**: PCA (PC1 vs PC2)

4. **Interpretacja wyników**:
   - **R² > 0.7**: Dobry model regresji
   - **Accuracy > 0.9**: Dobry model klasyfikacji
   - **Silhouette > 0.5**: Dobre klastrowanie
   - **Cumulative Variance > 90%**: Wystarczająca liczba PC

5. **Eksperymentuj**:
   - Zmień hiperparametry i obserwuj wpływ na wyniki
   - Porównaj różne algorytmy
   - Próbuj różnych strategii preprocessingu
""")

st.info("""
📚 **Potrzebujesz pomocy?** Wróć do poprzednich stron (1-8) aby nauczyć się więcej
o każdym algorytmie, jego teorii i zastosowaniach w bioinformatyce!
""")
