"""
Regresja Liniowa - Linear Regression
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_qsar_fish_toxicity
from src.navigation import render_sidebar_navigation

st.set_page_config(page_title="Regresja Liniowa", page_icon="📈", layout="wide")

# Render sidebar navigation
render_sidebar_navigation()

st.title("📈 Regresja Liniowa (Linear Regression)")

# Create tabs for Theory and Demo
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest Regresja Liniowa?

    Regresja liniowa to fundamentalna metoda statystyczna i algorytm uczenia maszynowego nadzorowanego.
    Jej celem jest **modelowanie i estymowanie relacji** między skalarną zmienną zależną (target)
    a jedną lub wieloma zmiennymi niezależnymi (features).

    ### Prosta Regresja Liniowa
    W przypadku jednej zmiennej niezależnej, model przyjmuje postać:
    """)

    st.latex(r"y = \beta_0 + \beta_1 x + \epsilon")

    st.markdown("""
    Gdzie:
    - $y$ - zmienna zależna (target, np. toksyczność związku)
    - $x$ - zmienna niezależna (predyktor, np. masa cząsteczkowa)
    - $\\beta_0$ - wyraz wolny (intercept)
    - $\\beta_1$ - współczynnik nachylenia (slope)
    - $\\epsilon$ - błąd losowy

    ### Wielokrotna Regresja Liniowa
    Dla wielu zmiennych niezależnych (np. wielu deskryptorów chemicznych), model dopasowuje
    hiperpłaszczyznę do danych.

    Parametry ($\\beta_0$, $\\beta_1$, ...) są estymowane przy użyciu **Metody Najmniejszych Kwadratów**
    (Ordinary Least Squares, OLS), która minimalizuje sumę kwadratów błędów (RSS).

    ## 2. Kluczowe Założenia Modelu

    Aby model regresji liniowej był wiarygodny, muszą być spełnione następujące założenia:

    1. **Liniowość** - Relacja między X a y jest liniowa
    2. **Niezależność Reszt** - Błędy są niezależne od siebie
    3. **Homoskedastyczność** - Wariancja reszt jest stała
    4. **Normalność Reszt** - Reszty mają rozkład normalny
    5. **Brak Multikolinearności** - Predyktory nie są silnie skorelowane (w regresji wielokrotnej)

    ## 3. Wady i Zalety

    ### ✅ Zalety:
    - **Interpretowalność** - Współczynniki bezpośrednio pokazują wpływ cech
    - **Wydajność** - Działa dobrze dla danych liniowo separowalnych
    - **Ekstrapolacja** - Może przewidywać poza zakresem danych treningowych
    - **Prostota** - Łatwa implementacja i zrozumienie

    ### ❌ Wady:
    - **Założenie liniowości** - Nie radzi sobie z nieliniowymi zależnościami
    - **Wrażliwość na outliery** - Obserwacje odstające silnie wpływają na model
    - **Wrażliwość na multikolinearność** - Wysoka korelacja predyktorów destabilizuje współczynniki

    ## 4. Miary Ewaluacji

    Do oceny jakości modelu regresyjnego używamy:

    - **R² (Współczynnik Determinacji)** - Procent wariancji w y wyjaśnianej przez X (0-1, wyższy lepszy)
    - **MAE (Mean Absolute Error)** - Średni bezwzględny błąd (w jednostkach y)
    - **MSE (Mean Squared Error)** - Średnia kwadratów błędów (karze większe błędy)
    - **RMSE (Root Mean Squared Error)** - Pierwiastek z MSE (w jednostkach y)

    ## 5. Zastosowanie w Bioinformatyce: QSAR

    **QSAR (Quantitative Structure-Activity Relationship)** - Ilościowa Zależność między Strukturą a Aktywnością

    ### Cel
    Znalezienie statystycznej zależności między strukturą chemiczną związku a jego aktywnością
    biologiczną (np. toksycznością, zdolnością do inhibicji enzymu).

    ### Jak to działa?

    1. **Struktura** jest reprezentowana przez liczbowe deskryptory molekularne:
       - CIC0 - Information content index
       - SM1_Dz(Z) - Spectral moment
       - GATS1i - Geary autocorrelation
       - NdsCH, NdssC - Liczba atomów określonych typów
       - MLOGP - Molar log P (lipofilowość)

    2. **Aktywność** jest mierzona eksperymentalnie:
       - LC50 - stężenie powodujące śmierć 50% organizmów testowych

    3. **Model** przewiduje aktywność na podstawie struktury:
    """)

    st.latex(r"\text{Aktywność Biologiczna} = f(\text{Deskryptory Molekularne})")

    st.markdown("""
    ### Zastosowanie
    Modele QSAR pozwalają na przewidywanie aktywności (np. toksyczności) **nowych, nieprzetestowanych**
    związków chemicznych, co drastycznie obniża koszty i przyspiesza badania przesiewowe w procesie
    odkrywania leków.

    ---

    ## 📖 Dodatkowe Zasoby

    - [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
    - [QSAR in Drug Discovery](https://en.wikipedia.org/wiki/Quantitative_structure%E2%80%93activity_relationship)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: QSAR Fish Toxicity")

    st.markdown("""
    Ten demo pokazuje wykorzystanie regresji liniowej do przewidywania **toksyczności związków chemicznych
    dla ryb** na podstawie deskryptorów molekularnych.

    **Zadanie:** Przewidywanie LC50 (toksyczność) na podstawie cech chemicznych.
    """)

    # Load data
    try:
        df = load_qsar_fish_toxicity()

        # Sidebar controls
        st.sidebar.header("⚙️ Ustawienia Demo")

        descriptors = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
        selected_descriptor = st.sidebar.selectbox(
            "Wybierz deskryptor molekularny (Oś X):",
            options=descriptors,
            index=5  # Default to MLOGP
        )

        st.sidebar.markdown("""
        ---
        **Informacje o deskryptorach:**
        - **CIC0**: Information content index
        - **SM1_Dz(Z)**: Spectral moment
        - **GATS1i**: Geary autocorrelation
        - **NdsCH**: Liczba atomów ds-CH
        - **NdssC**: Liczba atomów dss-C
        - **MLOGP**: Molar log P (lipofilowość)
        """)

        # Prepare data
        X = df[[selected_descriptor]].values
        y = df['LC50'].values

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        # Display model equation
        st.subheader("📐 Równanie Modelu")
        beta_0 = model.intercept_
        beta_1 = model.coef_[0]

        st.latex(f"LC50 = {beta_0:.3f} + {beta_1:.3f} \\times {selected_descriptor}")

        st.markdown(f"""
        - **Wyraz wolny (β₀)**: {beta_0:.3f}
        - **Współczynnik nachylenia (β₁)**: {beta_1:.3f}

        **Interpretacja:** Gdy {selected_descriptor} wzrasta o 1 jednostkę,
        LC50 {'wzrasta' if beta_1 > 0 else 'maleje'} o {abs(beta_1):.3f}.
        """)

        # Visualization
        st.subheader("📊 Wizualizacja Regresji")

        fig = px.scatter(
            df,
            x=selected_descriptor,
            y='LC50',
            trendline="ols",
            title=f"Regresja Liniowa: LC50 vs {selected_descriptor}",
            labels={selected_descriptor: selected_descriptor, 'LC50': 'LC50 (Toksyczność)'},
            opacity=0.6
        )

        fig.update_traces(marker=dict(size=6))
        fig.update_layout(
            width=800,
            height=500,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        st.subheader("📈 Metryki Wydajności Modelu")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("MSE", f"{mse:.4f}")
        with col4:
            st.metric("RMSE", f"{rmse:.4f}")

        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Zmień deskryptor** w menu po lewej i obserwuj jak zmienia się R²
        2. **Porównaj deskryptory**: który ma najwyższy R²? (najlepiej przewiduje toksyczność)
        3. **Zwróć uwagę** na nachylenie linii - dodatnie czy ujemne?
        4. **Interpretuj**: czy związek o wyższym MLOGP (lipofilowości) jest bardziej czy mniej toksyczny?

        ### 🔍 Obserwacje:
        - **MLOGP** (lipofilowość) jest często dobrym predyktorem toksyczności
        - **R² < 0.5** sugeruje, że pojedynczy deskryptor nie wyjaśnia pełnej wariancji
        - Dla lepszych wyników potrzebowalibyśmy **wielokrotnej regresji liniowej** (wszystkie 6 deskryptorów)
        """)

        # Data preview
        with st.expander("📋 Podgląd Danych (pierwsze 10 wierszy)"):
            st.dataframe(df.head(10))

        # Additional analysis
        with st.expander("📊 Analiza Reszt (Residuals)"):
            residuals = y - y_pred

            fig_residuals = px.scatter(
                x=y_pred,
                y=residuals,
                title="Wykres Reszt",
                labels={'x': 'Przewidywane LC50', 'y': 'Reszty (Błędy)'},
                opacity=0.6
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(width=700, height=400)

            st.plotly_chart(fig_residuals, use_container_width=True)

            st.markdown("""
            **Analiza reszt** pozwala sprawdzić założenia modelu:
            - Reszty powinny być losowo rozrzucone wokół linii y=0
            - Brak widocznych wzorców sugeruje dobre dopasowanie modelu
            - Wzorce (np. lejek) sugerują heteroskedastyczność
            """)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Dane QSAR zostaną wygenerowane syntetycznie, jeśli plik nie istnieje.")
