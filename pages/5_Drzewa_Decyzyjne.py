"""
Drzewa Decyzyjne - Decision Trees
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_breast_cancer_data
from src.plots import plot_confusion_matrix
from src.navigation import render_sidebar_navigation

st.set_page_config(page_title="Drzewa Decyzyjne", page_icon="🌳", layout="wide")

# Render sidebar navigation
render_sidebar_navigation()

st.title("🌳 Drzewa Decyzyjne (Decision Trees)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym są Drzewa Decyzyjne?

    Drzewa decyzyjne to algorytm uczenia maszynowego nadzorowanego, który modeluje decyzje
    i ich możliwe konsekwencje w strukturze drzewa. Algorytm **dzieli przestrzeń cech**
    na regiony, wykonując sekwencję decyzji binarnych (tak/nie) w każdym węźle.

    ### Struktura Drzewa

    - **Węzeł główny (Root Node)**: Zawiera wszystkie dane
    - **Węzły wewnętrzne (Internal Nodes)**: Testy na cechy (np. "czy wiek > 50?")
    - **Liście (Leaf Nodes)**: Końcowe decyzje (klasy w klasyfikacji)
    - **Gałęzie (Branches)**: Reprezentują wynik testu (tak/nie)

    ### Jak działa budowa drzewa?

    1. **Wybór cechy do podziału**: Algorytm wybiera cechę i próg, który najlepiej dzieli dane
    2. **Podział danych**: Dane są dzielone na dwa podzbiory na podstawie testu
    3. **Rekursja**: Proces powtarza się dla każdego podzbioru
    4. **Warunek stop**: Proces zatrzymuje się gdy:
       - Wszystkie dane w węźle należą do jednej klasy
       - Osiągnięto maksymalną głębokość
       - Węzeł zawiera zbyt mało próbek

    ## 2. Kryteria Podziału: Gini vs Entropy

    Algorytm musi decydować **która cecha i próg** najlepiej dzielą dane. Używa do tego
    miar "nieczystości" (impurity).

    ### Gini Impurity (Indeks Giniego)
    """)

    st.latex(r"Gini = 1 - \sum_{i=1}^{C} p_i^2")

    st.markdown("""
    Gdzie $p_i$ to proporcja próbek klasy $i$ w węźle, $C$ to liczba klas.

    - **Gini = 0**: Węzeł jest czysty (wszystkie próbki jednej klasy)
    - **Gini = 0.5**: Maksymalna nieczystość (dla 2 klas z równymi proporcjami)
    - **Obliczeniowo szybsze** niż entropia

    ### Entropy (Entropia Shannona)
    """)

    st.latex(r"Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)")

    st.markdown("""
    - **Entropy = 0**: Węzeł jest czysty
    - **Entropy = 1**: Maksymalna nieczystość (dla 2 klas z równymi proporcjami)
    - Oparta na teorii informacji
    - **Wolniejsza** niż Gini

    ### Gini vs Entropy - co wybrać?

    - **Praktycznie dają bardzo podobne wyniki!**
    - **Gini**: Domyślne w scikit-learn, szybsze
    - **Entropy**: Może tworzyć nieco bardziej zrównoważone drzewa
    - Różnice są zazwyczaj minimalne

    ## 3. Wady i Zalety

    ### ✅ Zalety:

    - **Interpretowalność** - Model "white-box": można wizualnie zobaczyć i zrozumieć decyzje
    - **Brak wymogu skalowania** - Nie wymaga normalizacji/standaryzacji cech
    - **Nieliniowość** - Radzi sobie z nieliniowymi zależnościami
    - **Obsługa różnych typów danych** - Numeryczne i kategoryczne
    - **Odporność na outliery** - Nie wpływają silnie na podział
    - **Feature importance** - Automatycznie wskazuje ważne cechy

    ### ❌ Wady:

    - **Przeuczenie** - Skłonność do budowania zbyt złożonych drzew (high variance)
    - **Niestabilność** - Małe zmiany w danych mogą prowadzić do zupełnie innego drzewa
    - **Bias** - Faworyzuje cechy z wieloma wartościami
    - **Granice ortogonalne** - Dzieli przestrzeń tylko wzdłuż osi (axis-aligned splits)
    - **Gorsze generalizacje** niż ensembles (Random Forest, Gradient Boosting)

    ## 4. Kontrola Przeuczenia

    Drzewa decyzyjne łatwo się **przeuczają**, rosnąc głęboko aby idealnie dopasować się
    do danych treningowych. Kontrolujemy to poprzez:

    ### Parametry Pruning (Przycinania)

    - **max_depth**: Maksymalna głębokość drzewa (np. 3-10)
    - **min_samples_split**: Minimalna liczba próbek do podziału węzła (np. 2-20)
    - **min_samples_leaf**: Minimalna liczba próbek w liściu (np. 1-10)
    - **max_leaf_nodes**: Maksymalna liczba liści

    **Strategie:**
    - Początek: Pozwól drzewu rosnąć głęboko → zobaczysz przeuczenie
    - Następnie: Ogranicz głębokość (max_depth=5) → lepsze generalizowanie

    ## 5. Zastosowanie w Bioinformatyce: Selekcja Genów

    Drzewa decyzyjne są szeroko stosowane w bioinformatyce, szczególnie do **selekcji
    cech (gene selection)** w analizie ekspresji genów.

    ### Cel
    Identyfikacja **genów biomarkerowych** - genów, których ekspresja najlepiej odróżnia
    próbki biologiczne (np. zdrowe vs choroba).

    ### Jak to działa?

    1. **Dane wejściowe**: Macierz ekspresji genów
       - Wiersze: Próbki pacjentów (np. 100 pacjentów)
       - Kolumny: Geny (np. 20,000 genów)
       - Wartości: Poziomy ekspresji (z RNA-seq, mikromacierzy)

    2. **Trenowanie**: DecisionTreeClassifier(max_depth=5)
       - Klasyfikacja: zdrowy vs chory

    3. **Feature Importance**: Po trenowaniu, drzewo zwraca `feature_importances_`
       - Wartości 0-1 dla każdego genu
       - Suma = 1.0
       - **Wysokie wartości = ważne geny biomarkerowe**

    4. **Selekcja**: Wybierz top N genów (np. top 50)

    ### Przykład
    """)

    st.latex(r"\text{Gene Importance} = \frac{\text{Reduction in Gini/Entropy}}{\text{Total Reduction}}")

    st.markdown("""
    Geny które **najlepiej dzielą** pacjentów (zdrowi vs chorzy) na wczesnych poziomach
    drzewa mają **najwyższe importance**.

    ### Zastosowania

    - **Klasyfikacja typów nowotworów** - Identyfikacja genów biomarkerowych
    - **Diagnoza** - Przewidywanie choroby na podstawie profilu ekspresji
    - **Odkrywanie leków** - Identyfikacja genów docelowych
    - **Medycyna personalizowana** - Stratyfikacja pacjentów

    ### Dlaczego Drzewa Decyzyjne?

    - **Interpretowalność**: Lekarze/badacze mogą zobaczyć "dlaczego" pacjent został sklasyfikowany
    - **Feature Importance**: Automatyczna identyfikacja kluczowych genów
    - **Odporność**: Nie wymaga normalizacji, odporne na outliery

    **UWAGA**: W praktyce często używa się **Random Forest** (ensemble drzew) dla lepszej
    accuracy, ale pojedyncze drzewo jest najlepsze dla interpretowalności!

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
    - [Gene Selection with Decision Trees](https://www.ncbi.nlm.nih.gov/pmc/articles/)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Breast Cancer Classification")

    st.markdown("""
    Ten demo pokazuje wykorzystanie drzew decyzyjnych do klasyfikacji nowotworów piersi
    jako **złośliwe (malignant)** lub **łagodne (benign)** na podstawie cech komórek.

    **Zbiór danych**: Breast Cancer Wisconsin Dataset (569 próbek, 30 cech)
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Ustawienia Demo")

    criterion = st.sidebar.selectbox(
        "Kryterium podziału:",
        options=['gini', 'entropy'],
        format_func=lambda x: f"{x.capitalize()} Impurity"
    )

    max_depth = st.sidebar.slider(
        "Maksymalna głębokość drzewa:",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    min_samples_split = st.sidebar.slider(
        "Minimalna liczba próbek do podziału:",
        min_value=2,
        max_value=50,
        value=2,
        step=1
    )

    test_size = st.sidebar.slider(
        "Rozmiar zbioru testowego (%):",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    ) / 100

    st.sidebar.markdown("""
    ---
    **Wskazówki:**
    - Zwiększ `max_depth` → drzewo rosnie głębiej
    - Niskie `max_depth` (1-3) → prostsze, bardziej ogólne
    - Wysokie `max_depth` (>10) → przeuczenie!
    """)

    # Load data
    try:
        X, y, feature_names, target_names = load_breast_cancer_data()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train Decision Tree
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Overfitting indicator
        overfitting_gap = train_accuracy - test_accuracy

        # Display info
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Parametry Modelu:**
            - Kryterium: **{criterion}**
            - Max głębokość: **{max_depth}**
            - Min próbek do podziału: **{min_samples_split}**
            - Liczba liści: **{model.get_n_leaves()}**
            - Głębokość drzewa: **{model.get_depth()}**
            """)

        with col2:
            st.info(f"""
            **Podział Danych:**
            - Trening: **{len(X_train)} próbek**
            - Test: **{len(X_test)} próbek**
            - Klasy: **{target_names[0]} / {target_names[1]}**
            """)

        # Metrics
        st.subheader("📈 Metryki Wydajności")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Train Accuracy",
                f"{train_accuracy:.4f}",
                help="Dokładność na zbiorze treningowym"
            )
        with col2:
            st.metric(
                "Test Accuracy",
                f"{test_accuracy:.4f}",
                help="Dokładność na zbiorze testowym"
            )
        with col3:
            st.metric(
                "Test F1-Score",
                f"{test_f1:.4f}",
                help="F1-score na zbiorze testowym"
            )
        with col4:
            st.metric(
                "Overfitting Gap",
                f"{overfitting_gap:.4f}",
                delta=f"{-overfitting_gap:.4f}",
                delta_color="inverse",
                help="Różnica między Train i Test Accuracy"
            )

        # Overfitting warning
        if overfitting_gap > 0.1:
            st.warning(f"""
            ⚠️ **Wykryto przeuczenie!**

            Model ma znacznie wyższą accuracy na zbiorze treningowym ({train_accuracy:.2%})
            niż testowym ({test_accuracy:.2%}). Gap = {overfitting_gap:.2%}

            **Rozwiązanie**: Zmniejsz `max_depth` lub zwiększ `min_samples_split`
            """)
        elif overfitting_gap < 0.02:
            st.success(f"""
            ✅ **Model dobrze generalizuje!**

            Niewielka różnica między accuracy treningową a testową ({overfitting_gap:.2%}).
            Model nie jest przeuczony.
            """)

        # Decision Tree Visualization
        st.subheader("🌳 Wizualizacja Drzewa Decyzyjnego")

        st.markdown("""
        Drzewo pokazuje **sekwencję decyzji** podejmowanych przez model.
        - **Kolor**: Niebieski = Benign (0), Pomarańczowy = Malignant (1)
        - **Wartość Gini/Entropy**: Im niższa, tym czystsza klasa w węźle
        """)

        # Plot decision tree
        fig, ax = plt.subplots(figsize=(20, 10))

        # Limit displayed features for readability
        max_displayed_depth = min(max_depth, 4)

        plot_tree(
            model,
            max_depth=max_displayed_depth,
            feature_names=feature_names,
            class_names=target_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        plt.tight_layout()
        st.pyplot(fig)

        if max_depth > 4:
            st.info(f"""
            **Uwaga**: Drzewo ma głębokość {model.get_depth()}, ale wyświetlamy tylko
            pierwsze {max_displayed_depth} poziomy dla czytelności.
            """)

        # Feature Importances
        st.subheader("🔍 Feature Importances (Ważność Cech)")

        st.markdown("""
        **Feature importance** pokazuje, które cechy były najważniejsze dla modelu.
        Wartości sumują się do 1.0.
        """)

        # Get top 15 features
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)

        import plotly.express as px
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Feature Importances',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})

        st.plotly_chart(fig_importance, use_container_width=True)

        # Confusion Matrix
        with st.expander("📊 Macierz Pomyłek (Confusion Matrix)"):
            cm = confusion_matrix(y_test, y_test_pred)
            fig_cm = plot_confusion_matrix(cm, target_names)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("""
            **Interpretacja Macierzy Pomyłek:**
            - **True Negatives (TN)**: Prawidłowo sklasyfikowane jako Benign
            - **True Positives (TP)**: Prawidłowo sklasyfikowane jako Malignant
            - **False Positives (FP)**: Błędnie sklasyfikowane jako Malignant (Type I error)
            - **False Negatives (FN)**: Błędnie sklasyfikowane jako Benign (Type II error)

            W diagnostyce medycznej **FN jest gorszy niż FP** (lepiej źle zdiagnozować
            zdrowego jako chorego, niż przegapić chorego)!
            """)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Przeuczenie vs Niedouczenie**:
           - Ustaw `max_depth=1`: Zobaczysz **niedouczenie** (underfitting) - prosty model
           - Ustaw `max_depth=20`: Zobaczysz **przeuczenie** (overfitting) - Train accuracy ≈100%
           - Złoty środek: `max_depth=5-7`

        2. **Gini vs Entropy**:
           - Przełącz między `gini` i `entropy`
           - Zauważysz, że wyniki są bardzo podobne!

        3. **Feature Importances**:
           - Które cechy są najważniejsze?
           - W tym zbiorze często: `worst concave points`, `worst perimeter`, `mean concave points`
           - To potencjalne **biomarkery** dla diagnozy raka!

        4. **Głębokość Drzewa**:
           - Sprawdź jak rośnie gap między Train a Test accuracy gdy zwiększasz depth

        ### 🧬 Biomedyczne wnioski:

        Ten model pokazuje, że **cechy geometryczne komórek** (perimeter, area, concavity)
        są kluczowe dla rozróżnienia nowotworów łagodnych i złośliwych.

        W rzeczywistej diagnostyce, model Decision Tree z `max_depth=5` może być używany jako:
        - **Narzędzie decyzyjne** dla patologów
        - **Selektor cech** dla bardziej złożonych modeli
        - **Wstępny screening** przed biopsją
        """)

        # Data preview
        with st.expander("📋 Podgląd Danych (pierwsze 10 wierszy)"):
            df_display = X.head(10).copy()
            df_display['target'] = y.head(10).map({0: target_names[0], 1: target_names[1]})
            st.dataframe(df_display)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Upewnij się, że funkcja load_breast_cancer_data() działa poprawnie.")
