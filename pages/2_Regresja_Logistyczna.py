"""
Regresja Logistyczna - Logistic Regression
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_breast_cancer_data
from src.plots import plot_probability_boundary_2d, plot_confusion_matrix

st.set_page_config(page_title="Regresja Logistyczna", page_icon="🎯", layout="wide")

st.title("🎯 Regresja Logistyczna (Logistic Regression)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest Regresja Logistyczna?

    Regresja Logistyczna jest fundamentalnym algorytmem uczenia nadzorowanego używanym do
    **problemów klasyfikacyjnych**. Pomimo nazwy, nie służy do regresji, lecz do przewidywania
    **prawdopodobieństwa** przynależności do klasy.

    Domyślnie używana jest do **klasyfikacji binarnej** (2 klasy: np. "chory" vs "zdrowy").

    ### Funkcja Logistyczna (Sigmoid)
    Podstawą modelu jest funkcja sigmoid, która "ściska" wynik liniowy do zakresu (0, 1):
    """)

    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

    st.markdown("""
    Gdzie $z$ jest liniową kombinacją cech:
    """)

    st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p")

    st.markdown("""
    Wynik $\\sigma(z)$ jest interpretowany jako **prawdopodobieństwo** przynależności do klasy "1".
    Ustalając próg (zazwyczaj 0.5), model dokonuje klasyfikacji.

    ## 2. Kluczowe Założenia Modelu

    Regresja logistyczna ma mniej rygorystyczne założenia niż liniowa:

    1. **Liniowość Log-Szans** - Liniowa zależność między X a logarytmem szans (log-odds)
    2. **Brak Multikolinearności** - Predyktory nie powinny być silnie skorelowane
    3. **Niezależność Obserwacji** - Obserwacje muszą być niezależne
    4. **Odpowiednio Duża Próba** - Wymagana wystarczająco duża próba

    ## 3. Miary Ewaluacji (Klasyfikacja)

    Dla modeli klasyfikacyjnych używamy innych metryk niż w regresji:
    """)

    st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
    st.latex(r"\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}")
    st.latex(r"\text{F1-Score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}")

    st.markdown("""
    Gdzie:
    - **TP** (True Positives) - Poprawnie zidentyfikowane przypadki pozytywne
    - **TN** (True Negatives) - Poprawnie zidentyfikowane przypadki negatywne
    - **FP** (False Positives) - Błędnie zidentyfikowane jako pozytywne (Błąd typu I)
    - **FN** (False Negatives) - Błędnie zidentyfikowane jako negatywne (Błąd typu II)

    ### Metryki:
    - **Accuracy** - Ogólna dokładność (uwaga: myląca przy niezbalansowanych danych!)
    - **Precision** - Jak bardzo możemy ufać predykcji pozytywnej?
    - **Recall** - Jaki procent faktycznych przypadków pozytywnych wykryliśmy?
    - **F1-Score** - Średnia harmoniczna Precision i Recall (zbalansowana metryka)

    ## 4. Regularyzacja

    Parametr **C** kontroluje siłę regularyzacji:
    - **Niskie C** (silna regularyzacja): Prostsza granica, mniej przeuczenia, wyższy bias
    - **Wysokie C** (słaba regularyzacja): Bardziej złożona granica, więcej przeuczenia, wyższa wariancja

    ## 5. Zastosowanie w Genomice: GWAS i SNP

    **GWAS (Genome-Wide Association Studies)** - Badania Asocjacyjne Całego Genomu

    ### Cel
    Identyfikacja wariantów genetycznych (SNP - polimorfizmów pojedynczego nukleotydu),
    które są statystycznie powiązane z ryzykiem wystąpienia choroby.

    ### Jak to działa?

    1. **Zbieramy dane**:
       - Grupa "przypadków" (cases) - pacjenci z daną chorobą
       - Grupa "kontrolna" (controls) - osoby zdrowe

    2. **Genotypujemy** setki tysięcy lub miliony SNP dla każdego osobnika

    3. **Budujemy model**:
       - Zmienna zależna ($y$): 1 (case) lub 0 (control)
       - Zmienne niezależne ($X$): genotypy (0, 1, 2 - liczba alleli ryzyka)
         oraz zmienne zakłócające (wiek, płeć, pochodzenie)

    4. **Model** $P(Choroba | Genotyp)$ pozwala oszacować:
    """)

    st.latex(r"\text{Odds Ratio} = \frac{P(Choroba|Allel=1)}{P(Choroba|Allel=0)}")

    st.markdown("""
    Informując nas, o ile dany wariant genetyczny zwiększa lub zmniejsza ryzyko choroby.

    ### Przykład
    SNP rs123456 ma OR = 1.5 dla cukrzycy typu 2, co oznacza, że osoby z allelem ryzyka
    mają 50% wyższe ryzyko rozwoju choroby.

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
    - [GWAS Overview](https://www.genome.gov/genetics-glossary/Genome-Wide-Association-Studies)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Breast Cancer Classification")

    st.markdown("""
    Ten demo pokazuje wykorzystanie regresji logistycznej do klasyfikacji nowotworów piersi
    jako **złośliwych (Malignant)** lub **łagodnych (Benign)** na podstawie cech komórkowych.

    **Wizualizacja 2D** pokazuje granicę decyzyjną w przestrzeni dwóch wybranych cech.
    """)

    try:
        # Load data
        X, y, feature_names, target_names = load_breast_cancer_data()

        # Sidebar controls
        st.sidebar.header("⚙️ Ustawienia Demo")

        # Regularization parameter
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        C_labels = ['0.001', '0.01', '0.1', '1.0', '10.0', '100.0', '1000.0']
        C_index = st.sidebar.select_slider(
            "Parametr Regularyzacji C (siła odwrotna)",
            options=range(len(C_values)),
            value=3,
            format_func=lambda x: C_labels[x]
        )
        C = C_values[C_index]

        st.sidebar.markdown(f"""
        **Wybrane C**: {C}

        - **Niskie C**: Silna regularyzacja, prostsza granica
        - **Wysokie C**: Słaba regularyzacja, złożona granica
        """)

        # Feature selection for 2D visualization
        default_features = ['mean radius', 'mean texture']
        feature_x = st.sidebar.selectbox(
            "Cecha na osi X:",
            options=list(feature_names),
            index=list(feature_names).index(default_features[0])
        )

        feature_y = st.sidebar.selectbox(
            "Cecha na osi Y:",
            options=list(feature_names),
            index=list(feature_names).index(default_features[1])
        )

        # Get indices of selected features
        idx_x = list(feature_names).index(feature_x)
        idx_y = list(feature_names).index(feature_y)

        # Prepare 2D data
        X_2d = X.iloc[:, [idx_x, idx_y]].values
        y_array = y.values

        # Scale features (CRITICAL for regularization)
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)

        # Train model
        model = LogisticRegression(C=C, random_state=42, max_iter=1000)
        model.fit(X_2d_scaled, y_array)

        # Make predictions
        y_pred = model.predict(X_2d_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_array, y_pred)
        precision = precision_score(y_array, y_pred)
        recall = recall_score(y_array, y_pred)
        f1 = f1_score(y_array, y_pred)
        cm = confusion_matrix(y_array, y_pred)

        # Visualization
        st.subheader("📊 Wizualizacja Granicy Decyzyjnej")

        fig = plot_probability_boundary_2d(
            model, X_2d_scaled, y_array,
            target_names, [feature_x, feature_y]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretacja:**
        - Kolor tła pokazuje prawdopodobieństwo przynależności do klasy "Malignant"
        - Czerwone punkty: Malignant (złośliwe)
        - Niebieskie punkty: Benign (łagodne)
        - Granica decyzyjna znajduje się w miejscu, gdzie P = 0.5
        """)

        # Metrics
        st.subheader("📈 Metryki Wydajności")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")

        # Confusion Matrix
        st.subheader("🔢 Macierz Pomyłek")

        col_cm, col_explain = st.columns([1, 1])

        with col_cm:
            fig_cm = plot_confusion_matrix(cm, ['Benign', 'Malignant'])
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_explain:
            st.markdown(f"""
            **Elementy macierzy:**

            - **True Negatives (TN)**: {cm[0, 0]} - Poprawnie sklasyfikowane jako Benign
            - **False Positives (FP)**: {cm[0, 1]} - Błędnie jako Malignant
            - **False Negatives (FN)**: {cm[1, 0]} - Błędnie jako Benign
            - **True Positives (TP)**: {cm[1, 1]} - Poprawnie jako Malignant

            **Uwaga:** W diagnostyce medycznej FN (przeoczenie raka) jest
            często gorszy niż FP (fałszywy alarm).
            """)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Zmień parametr C**:
           - Ustaw C=0.001 (silna regularyzacja) - granica będzie prosta
           - Ustaw C=1000 (słaba regularyzacja) - granica będzie złożona
           - Obserwuj wpływ na metryki!

        2. **Zmień cechy**:
           - Wybierz różne pary cech
           - Które pary najlepiej separują klasy?
           - Czy 'worst' cechy są lepsze niż 'mean'?

        3. **Zwróć uwagę**:
           - Jak regularyzacja wpływa na kształt granicy?
           - Czy model się przeuczą przy wysokim C?
           - Czy wszystkie punkty są poprawnie klasyfikowane?

        ### 🔬 Zastosowanie w praktyce:
        W rzeczywistości użylibyśmy **wszystkich 30 cech**, a nie tylko 2.
        Wizualizacja 2D służy wyłącznie celom edukacyjnym.
        """)

        # Data preview
        with st.expander("📋 Podgląd Danych (pierwsze 5 wierszy)"):
            preview_df = X.iloc[:5, [idx_x, idx_y]].copy()
            preview_df['target'] = y.iloc[:5].map({0: 'Malignant', 1: 'Benign'})
            st.dataframe(preview_df)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Dataset Breast Cancer Wisconsin jest wbudowany w scikit-learn.")
