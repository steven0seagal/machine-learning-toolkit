"""
k-Najbliższych Sąsiadów (k-NN) - k-Nearest Neighbors
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_breast_cancer_data
from src.plots import plot_decision_boundary_2d

st.set_page_config(page_title="k-NN", page_icon="🎯", layout="wide")

st.title("🎯 k-Najbliższych Sąsiadów (k-NN)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest k-Najbliższych Sąsiadów (k-NN)?

    k-NN to jeden z najprostszych algorytmów uczenia maszynowego. Należy do rodziny
    **"leniwych" algorytmów** (lazy learners) lub **opartych na instancjach** (instance-based).

    ### Kluczowa Cecha: Brak Fazy Trenowania
    - k-NN **nie buduje** aktywnego modelu podczas treningu
    - Po prostu **zapamiętuje** cały zbiór treningowy w pamięci
    - Całe "uczenie" odbywa się podczas predykcji!

    ### Jak działa predykcja?

    Gdy pojawia się nowa obserwacja:
    1. **Oblicz odległość** (np. Euklidesową) do każdego punktu treningowego
    2. **Znajdź k najbliższych** sąsiadów
    3. **Klasyfikacja**: Głosowanie większościowe (najczęstsza klasa wśród k sąsiadów)
    4. **Regresja**: Średnia wartość target z k sąsiadów

    ## 2. Kluczowy Hiperparametr: Wybór k

    To **najważniejszy** hiperparametr. Jego wybór to balansowanie **kompromisu bias-wariancja**:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📉 Niskie k (np. k=1)
        - Bardzo **elastyczny** model
        - **Niski bias**, **wysoka wariancja**
        - Wrażliwy na szum i outliery
        - **Przeuczenie** (overfitting)
        - Postrzępione granice decyzyjne
        """)

    with col2:
        st.markdown("""
        ### 📈 Wysokie k (np. k=N)
        - Bardzo **sztywny** model
        - **Wysoki bias**, **niska wariancja**
        - Klasyfikuje wszystko do klasy większościowej
        - **Niedouczenie** (underfitting)
        - Gładkie granice decyzyjne
        """)

    st.markdown("""
    ### 💡 Wskazówki:
    - Dla klasyfikacji binarnej: używaj **nieparzystego k** (unikniesz remisów)
    - Typowe wartości: k ∈ {3, 5, 7, 9, 11}
    - Wybór k poprzez walidację krzyżową

    ## 3. Założenia i Wymagania

    ### ⚠️ KRYTYCZNE: Skalowanie Cech
    k-NN jest **ekstremalnie wrażliwy** na skalę cech:
    - Cechy o dużych zakresach (np. 0-10000) dominują nad małymi (0-1)
    - **ZAWSZE** standaryzuj dane przed użyciem k-NN!

    ### Metryka Odległości
    Najczęściej używane:
    - **Euklidesowa**: $d = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$
    - **Manhattan**: $d = \\sum_{i=1}^{n} |x_i - y_i|$
    - **Minkowski**: Uogólnienie powyższych

    ## 4. Wady i Zalety

    ### ✅ Zalety:
    - **Prostota**: Niezwykle łatwy do implementacji
    - **Adaptowalność**: Łatwo dodawać nowe dane (bez ponownego treningu)
    - **Nieliniowość**: Naturalne obsługuje nieliniowe granice
    - **Mało hiperparametrów**: Głównie k i metryka odległości

    ### ❌ Wady:
    - **Koszt obliczeniowy**: Predykcja wymaga porównania z KAŻDYM punktem treningowym
    - **Nie skaluje się**: Nie nadaje się do dużych zbiorów danych
    - **Klątwa wymiarowości**: Działa słabo w wysokowymiarowych przestrzeniach
    - **Wrażliwość na szum**: Szczególnie przy małym k

    ## 5. Zastosowanie w Bioinformatyce: Klasyfikacja Ekspresji Genów

    Pomimo wad, k-NN jest często używany w analizie danych z mikromacierzy lub RNA-Seq.

    ### Cel
    Klasyfikacja próbek biologicznych (np. typów nowotworów) na podstawie profilu ekspresji genów.

    ### Jak to działa?

    1. **Dane**: Każda próbka (pacjent) = wektor poziomów ekspresji (~20,000 genów)
    2. **Metryka**: Odległość między próbkami = różnica w profilach ekspresji
    3. **Predykcja**: Nowa próbka klasyfikowana na podstawie k najbardziej podobnych próbek

    ### Problem: Klątwa Wymiarowości
    - Gdy p (liczba genów) >> n (liczba próbek), odległości stają się nieinformatywne
    - **Rozwiązanie**: Selekcja cech lub redukcja wymiaru (PCA) przed k-NN

    ### Przykład
    - Mamy 100 próbek nowotworów (50 Typ A, 50 Typ B)
    - Każda próbka ma ekspresję 20,000 genów
    - Używamy PCA do redukcji do 10 komponentów
    - k-NN (k=5) klasyfikuje nową próbkę na podstawie 5 najbliższych sąsiadów

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn k-NN](https://scikit-learn.org/stable/modules/neighbors.html)
    - [k-NN in Bioinformatics](https://academic.oup.com/bioinformatics/)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Breast Cancer Classification")

    st.markdown("""
    Ten demo wizualizuje **kompromis bias-wariancja** w k-NN. Zobaczysz jak wartość k
    wpływa na kształt granicy decyzyjnej.

    **Wizualizacja 2D** pokazuje granice decyzyjne (mozaikę Voronoi) dla różnych wartości k.
    """)

    try:
        # Load data
        X, y, feature_names, target_names = load_breast_cancer_data()

        # Sidebar controls
        st.sidebar.header("⚙️ Ustawienia Demo")

        # k parameter
        k = st.sidebar.slider(
            "Liczba sąsiadów (k):",
            min_value=1,
            max_value=51,
            value=5,
            step=2,  # Force odd values
            help="Niskie k = przeuczenie, Wysokie k = niedouczenie"
        )

        st.sidebar.markdown(f"""
        **Wybrane k**: {k}

        - **k=1**: Maksymalne przeuczenie
        - **k=5**: Dobrze zbalansowane
        - **k=51**: Potencjalne niedouczenie
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

        # Get indices
        idx_x = list(feature_names).index(feature_x)
        idx_y = list(feature_names).index(feature_y)

        # Prepare 2D data
        X_2d = X.iloc[:, [idx_x, idx_y]].values
        y_array = y.values

        # CRITICAL: Scale features
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)

        # Train model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_2d_scaled, y_array)

        # Predictions
        y_pred = model.predict(X_2d_scaled)

        # Metrics
        accuracy = accuracy_score(y_array, y_pred)
        f1 = f1_score(y_array, y_pred)

        # Visualization
        st.subheader("📊 Wizualizacja Granicy Decyzyjnej")

        fig = plot_decision_boundary_2d(
            model, X_2d_scaled, y_array,
            [feature_x, feature_y]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretacja:**
        - Mozaika kolorów pokazuje regiony decyzyjne
        - Każdy region należy do jednej klasy
        - Granice są określone przez najbliższych k sąsiadów
        """)

        # Metrics
        st.subheader("📈 Metryki Wydajności")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("k (liczba sąsiadów)", k)
        with col2:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col3:
            st.metric("F1-Score", f"{f1:.4f}")

        # Bias-Variance explanation
        st.subheader("⚖️ Kompromis Bias-Wariancja")

        if k <= 3:
            st.warning(f"""
            **k={k}: Wysokie ryzyko przeuczenia!**

            - Granica decyzyjna jest bardzo **postrzępiona**
            - Model idealnie dopasowuje się do szumu w danych treningowych
            - **Wysoka wariancja** - małe zmiany danych → duże zmiany modelu
            - **Niski bias** - model jest bardzo elastyczny
            - Może słabo generalizować na nowe dane
            """)
        elif k >= 25:
            st.info(f"""
            **k={k}: Ryzyko niedouczenia**

            - Granica decyzyjna jest bardzo **gładka**
            - Model ignoruje lokalne struktury w danych
            - **Niska wariancja** - model jest stabilny
            - **Wysoki bias** - model jest zbyt sztywny
            - Może nie wychwytywać istotnych wzorców
            """)
        else:
            st.success(f"""
            **k={k}: Dobrze zbalansowane!**

            - Granica decyzyjna jest **umiarkowanie złożona**
            - Model balansuje dopasowanie i generalizację
            - **Średnia wariancja i bias**
            - Typowa "dobra" wartość k dla tego problemu
            """)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Eksperymentuj z k**:
           - Ustaw k=1 → Zobacz ekstremalnie postrzępione granice (każdy punkt tworzy własną wyspę)
           - Ustaw k=5 → Granice są bardziej gładkie, ale wciąż elastyczne
           - Ustaw k=51 → Bardzo gładkie, uogólnione granice

        2. **Obserwuj metryki**:
           - Przy k=1: Accuracy na danych treningowych będzie ~100% (przeuczenie!)
           - Przy optymalnym k: Najlepszy balans
           - Przy dużym k: Accuracy spada (niedouczenie)

        3. **Zmień cechy**:
           - Które pary cech dają najlepszą separację klas?
           - Czy potrzebujesz większego czy mniejszego k dla różnych cech?

        ### 🔬 W praktyce:
        - Użylibyśmy **walidacji krzyżowej** do wyboru optymalnego k
        - Użylibyśmy **wszystkich 30 cech**, nie tylko 2
        - Zawsze **standaryzujemy** dane przed k-NN!
        """)

        # Data preview
        with st.expander("📋 Informacje o Danych"):
            st.markdown(f"""
            - **Liczba próbek**: {len(X)}
            - **Liczba cech**: {len(feature_names)}
            - **Klasy**: {', '.join(target_names)}
            - **Rozkład klas**: Benign: {sum(y==1)}, Malignant: {sum(y==0)}
            """)

    except Exception as e:
        st.error(f"Błąd: {str(e)}")
