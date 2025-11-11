"""
Maszyny Wektorów Nośnych (SVM) - Support Vector Machines
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.plots import plot_decision_boundary_2d
from src.navigation import render_sidebar_navigation

st.set_page_config(page_title="SVM", page_icon="⚛️", layout="wide")

# Render sidebar navigation
render_sidebar_navigation()

st.title("⚛️ Maszyny Wektorów Nośnych (SVM)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym są Maszyny Wektorów Nośnych (SVM)?

    SVM to potężny algorytm uczenia nadzorowanego używany do klasyfikacji, regresji i wykrywania anomalii.
    **SVM jest szczególnie efektywny** w przestrzeniach wysokowymiarowych, nawet gdy liczba wymiarów
    (cech) jest większa niż liczba próbek!

    ### Podstawowa Idea

    Celem SVM jest znalezienie **optymalnej hiperpłaszczyzny** (linii w 2D, płaszczyzny w 3D),
    która najlepiej separuje klasy w zbiorze danych.

    **Optymalna hiperpłaszczyzna** = ta z **maksymalnym marginesem**
    - Margines = odległość do najbliższych punktów danych z obu klas
    - Najbliższe punkty = **wektory nośne** (support vectors)
    - Tylko wektory nośne "podtrzymują" hiperpłaszczyznę, inne punkty są ignorowane!

    ## 2. Kernel Trick (Sztuczka Jądrowa)

    Wiele rzeczywistych zbiorów danych **nie jest separowalna liniowo**. SVM radzi sobie z tym
    za pomocą "sztuczki jądrowej"!

    ### Idea
    Dane są transformowane z oryginalnej przestrzeni (np. 2D) do przestrzeni o wyższym wymiarze
    (np. 3D lub nieskończonym), gdzie stają się liniowo separowalne.

    ### Funkcje Jądrowe (Kernels)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 'linear' (Liniowe)
        - Dla danych liniowo separowalnych
        - Najszybsze obliczeniowo
        - Prosta hiperpłaszczyzna

        #### 'poly' (Wielomianowe)
        - Dla krzywoliniowych granic
        - Parametr: degree (stopień wielomianu)
        - Średnia złożoność

        """)

    with col2:
        st.markdown("""
        #### 'rbf' (Radial Basis Function)
        - **Najpopularniejsze!**
        - Dla złożonych, nieliniowych granic
        - Parametr: gamma
        - Bardzo elastyczne

        #### 'sigmoid'
        - Rzadziej używane
        - Podobne do sieci neuronowych
        """)

    st.markdown("""
    ## 3. Kluczowe Hiperparametry

    ### C (Parametr Regularyzacji)
    Kontroluje kompromis między maksymalizacją marginesu a minimalizacją błędu klasyfikacji.
    """)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        **Niskie C** (np. 0.01)
        - **Miękki margines**
        - Toleruje błędy klasyfikacji
        - Szerszy, bardziej ogólny margines
        - **Niska wariancja, wysoki bias**
        - Ryzyko niedouczenia
        """)

    with col_c2:
        st.markdown("""
        **Wysokie C** (np. 100)
        - **Twardy margines**
        - Stara się poprawnie sklasyfikować każdy punkt
        - Wąski margines
        - **Wysoka wariancja, niski bias**
        - Ryzyko przeuczenia
        """)

    st.markdown("""
    ### gamma (Dla jądra RBF i Poly)
    Definiuje jak daleko sięga wpływ pojedynczego wektora nośnego.
    """)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("""
        **Niskie gamma** (np. 0.01)
        - Szeroki wpływ
        - Gładka granica decyzyjna
        - **Niska wariancja, wysoki bias**
        - Model bardziej ogólny
        """)

    with col_g2:
        st.markdown("""
        **Wysokie gamma** (np. 10)
        - Wąski wpływ (tylko najbliższe punkty)
        - Pofałdowana granica decyzyjna
        - **Wysoka wariancja, niski bias**
        - Przeuczenie do pojedynczych punktów
        """)

    st.markdown("""
    ## 4. Zastosowanie w Bioinformatyce: Klasyfikacja Białek

    SVM jest jednym z **najskuteczniejszych** algorytmów w bioinformatyce, szczególnie w proteomice.

    ### Cel
    Przewidywanie funkcji, struktury drugorzędowej, lokalizacji subkomórkowej lub interakcji białek
    na podstawie sekwencji aminokwasowej.

    ### Jak to działa?

    1. **Kodowanie sekwencji**: Sekwencja białkowa (ciąg liter) → wektor liczbowy
       - Skład aminokwasowy
       - PseAAC (Pseudo-amino acid composition)
       - PSSM (Position-Specific Scoring Matrix) - profile ewolucyjne

    2. **Przestrzeń wysokowymiarowa**: Typowo p >> n (więcej cech niż próbek)
       - 1000+ wymiarów, 200 próbek
       - To jest **siła SVM**!

    3. **Trenowanie**: SVM (zazwyczaj RBF kernel) na zbiorze białek o znanej funkcji

    4. **Predykcja**: Klasyfikacja nowych białek

    ### Przykłady Zastosowań

    - **Przewidywanie struktury drugorzędowej**: α-helisa, β-kartka, pętla
    - **Lokalizacja subkomórkowa**: jądro, mitochondrium, cytoplazma
    - **Funkcja białka**: enzym, receptor, transporter
    - **Interakcje białko-białko**: czy dwa białka oddziałują?

    ### Dlaczego SVM?
    - Efektywność w przestrzeniach wysokowymiarowych (p > n)
    - Odporność na przeuczenie (dzięki maksymalizacji marginesu)
    - Elastyczność dzięki kernelom
    - Często osiąga najwyższą dokładność

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
    - [SVM in Bioinformatics](https://bmcbioinformatics.biomedcentral.com/)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Nieliniowe Granice Decyzyjne")

    st.markdown("""
    Ten demo pokazuje **moc jąder nieliniowych** (szczególnie RBF) w SVM.
    Używamy syntetycznych danych w kształcie księżyców, które **nie są liniowo separowalne**.

    **Cel**: Zobaczysz jak kernel i parametry C, gamma wpływają na granicę decyzyjną.
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Ustawienia Demo")

    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Typ danych syntetycznych:",
        options=['moons', 'circles'],
        format_func=lambda x: 'Księżyce (Moons)' if x == 'moons' else 'Koła (Circles)'
    )

    noise_level = st.sidebar.slider(
        "Poziom szumu:",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.05
    )

    # Kernel selection
    kernel = st.sidebar.selectbox(
        "Wybierz jądro (kernel):",
        options=['linear', 'rbf', 'poly'],
        index=1
    )

    # C parameter
    C_exp = st.sidebar.slider(
        "Parametr Regularyzacji (C) - skala log:",
        min_value=-2.0,
        max_value=3.0,
        value=0.0,
        step=0.5
    )
    C = 10 ** C_exp

    # Gamma parameter (only for RBF and poly)
    if kernel in ['rbf', 'poly']:
        gamma_exp = st.sidebar.slider(
            "Parametr Gamma - skala log:",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.5
        )
        gamma = 10 ** gamma_exp
    else:
        gamma = 'scale'

    st.sidebar.markdown(f"""
    ---
    **Aktualne wartości:**
    - C = {C:.3f}
    - Gamma = {gamma if isinstance(gamma, str) else f'{gamma:.3f}'}
    - Kernel = {kernel}
    """)

    # Generate synthetic data
    np.random.seed(42)
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=300, noise=noise_level, random_state=42)
    else:
        X, y = make_circles(n_samples=300, noise=noise_level, factor=0.5, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    if kernel in ['rbf', 'poly']:
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    else:
        model = SVC(kernel=kernel, C=C, random_state=42)

    model.fit(X_scaled, y)

    # Predictions
    y_pred = model.predict(X_scaled)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    n_support = model.n_support_

    # Visualization
    st.subheader("📊 Wizualizacja Granicy Decyzyjnej")

    fig = plot_decision_boundary_2d(
        model, X_scaled, y,
        ['Feature 1', 'Feature 2']
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.subheader("📈 Metryki Wydajności")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("F1-Score", f"{f1:.4f}")
    with col3:
        st.metric("Support Vectors (Class 0)", n_support[0])
    with col4:
        st.metric("Support Vectors (Class 1)", n_support[1])

    # Kernel explanation
    st.subheader("🎯 Interpretacja Wyników")

    if kernel == 'linear':
        st.warning("""
        **Kernel 'linear' - Niepowodzenie!**

        Dane w kształcie księżyców/kół **nie są liniowo separowalne**.
        Liniowa hiperpłaszczyzna nie może ich poprawnie rozdzielić.

        **Accuracy jest niska** (~50-60%), model działa słabo.

        💡 **Spróbuj**: Przełącz się na kernel 'rbf'!
        """)
    elif kernel == 'rbf':
        if accuracy > 0.95:
            st.success(f"""
            **Kernel 'rbf' - Doskonałe dopasowanie!**

            Accuracy = {accuracy:.2%} - model idealnie separuje klasy!

            Kernel RBF **transformuje dane do przestrzeni wyższego wymiaru**,
            gdzie stają się liniowo separowalne.

            **Support Vectors**: {sum(n_support)} punktów (z {len(X)}) podtrzymuje hiperpłaszczyznę.
            """)
        elif accuracy < 0.7:
            st.warning(f"""
            **Kernel 'rbf' - Niedouczenie lub przeuczenie**

            Accuracy = {accuracy:.2%} - model nie działa optymalnie.

            Możliwe przyczyny:
            - **C zbyt niskie** → model zbyt prosty (niedouczenie)
            - **Gamma zbyt wysokie/niskie** → nieprawidłowa skala transformacji

            💡 **Spróbuj**: C=1.0, Gamma=1.0
            """)
        else:
            st.info(f"""
            **Kernel 'rbf' - Dobre dopasowanie**

            Accuracy = {accuracy:.2%}

            Model radzi sobie dobrze. Możesz spróbować dostroić C i gamma
            dla jeszcze lepszych wyników.
            """)
    else:  # poly
        st.info("""
        **Kernel 'poly' - Wielomianowa transformacja**

        Kernel wielomianowy może również modelować nieliniowe granice,
        ale często RBF działa lepiej w praktyce.
        """)

    # Experimentation tips
    st.markdown("""
    ---
    ### 💡 Wskazówki do eksperymentowania:

    1. **Porównaj kernele**:
       - **linear**: Zobacz że kompletnie zawodzi na tych danych
       - **rbf**: Idealna separacja (przy dobrych parametrach)
       - **poly**: Również może działać, ale RBF często lepsze

    2. **Eksperymentuj z C**:
       - **C=0.01**: Bardzo miękki margines, może niedouczać
       - **C=1.0**: Dobry balans
       - **C=100**: Twardy margines, może przeuczać (granica bardzo postrzępiona)

    3. **Eksperymentuj z gamma** (dla RBF):
       - **gamma=0.01**: Bardzo gładka granica (może za prosta)
       - **gamma=1.0**: Umiarkowanie złożona (zazwyczaj dobra)
       - **gamma=100**: Absurdalnie pofałdowana (przeuczenie!)

    4. **Obserwuj support vectors**:
       - Im więcej support vectors, tym bardziej złożona granica
       - Idealne modele: niewiele support vectors, wysoka accuracy

    ### 🧬 Analogia do bioinformatyki:
    Tak jak RBF kernel znajduje nieliniową granicę dla księżyców,
    tak samo w klasyfikacji białek SVM znajduje **złożone wzorce**
    w przestrzeni wysokowymiarowej sekwencji aminokwasowych!
    """)
