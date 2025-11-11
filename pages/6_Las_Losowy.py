"""
Las Losowy - Random Forest
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_breast_cancer_data
from src.plots import plot_confusion_matrix, plot_feature_importance

st.set_page_config(page_title="Las Losowy", page_icon="🌲", layout="wide")

st.title("🌲 Las Losowy (Random Forest)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest Random Forest?

    Random Forest (Las Losowy) to **ensemble learning algorithm** - metoda, która łączy
    predykcje wielu modeli bazowych aby uzyskać lepsze wyniki niż pojedynczy model.

    ### Podstawowa Idea

    Random Forest = **Wiele Drzew Decyzyjnych + Głosowanie**

    Zamiast budować jedno drzewo decyzyjne, Random Forest buduje **las** setek lub tysięcy
    drzew, a następnie:
    - **Klasyfikacja**: Każde drzewo "głosuje" na klasę → zwracana jest klasa większościowa
    - **Regresja**: Zwracana jest średnia predykcji wszystkich drzew

    """)

    st.latex(r"\text{Predykcja}_{RF} = \text{majority vote}\{\text{Drzewo}_1, \text{Drzewo}_2, ..., \text{Drzewo}_N\}")

    st.markdown("""
    ### Dlaczego Random Forest jest lepszy niż pojedyncze drzewo?

    **Pojedyncze Drzewo Decyzyjne**:
    - **High Variance** - Niestabilne, małe zmiany w danych → zupełnie inne drzewo
    - **Przeuczenie** - Łatwo dopasowuje się do szumu w danych treningowych

    **Random Forest**:
    - **Averaging/Voting** - Uśrednianie wielu drzew redukuje wariancję
    - **Lepsza generalizacja** - Mniej podatny na przeuczenie
    - **Robust** - Stabilny na zmianach w danych

    Koncepcja: "Mądrość tłumu" (Wisdom of Crowds) - wiele niezależnych estymatorów
    razem podejmują lepsze decyzje!

    ## 2. Jak działa Random Forest? - Dwa Źródła Losowości

    Random Forest wprowadza **losowość** podczas trenowania każdego drzewa, aby zapewnić
    że drzewa są **różnorodne i niezależne**.

    ### 2.1. Bootstrap Aggregating (Bagging)
    """)

    st.latex(r"\text{Bagging} = \text{Bootstrap} + \text{Aggregating}")

    st.markdown("""
    **Bootstrap**: Każde drzewo jest trenowane na **losowej próbce** danych z powtórzeniami
    - Mamy N próbek treningowych
    - Dla każdego drzewa: losuj N próbek **z powtórzeniami** (sampling with replacement)
    - Różne drzewa widzą trochę inne dane!

    **Przykład**: Dane = [1, 2, 3, 4, 5]
    - Drzewo 1 może dostać: [1, 2, 2, 4, 5]
    - Drzewo 2 może dostać: [1, 3, 3, 3, 5]
    - Każda próbka ma ~63% oryginalnych danych, ~37% duplikatów

    **Out-of-Bag (OOB) samples**: Próbki niewykorzystane w trenowaniu danego drzewa (~37%)
    mogą być użyte do walidacji!

    ### 2.2. Feature Randomness (Losowość Cech)

    W każdym węźle każdego drzewa:
    - Zamiast rozważać **wszystkie** cechy do podziału
    - Losujemy **podzbiór** cech (np. √p dla klasyfikacji, gdzie p = liczba cech)
    - Wybieramy najlepszą cechę **z tego podzbioru**

    **Dlaczego?** - Decorrelation (dekorelacja drzew)
    - Gdyby wszystkie drzewa widziały wszystkie cechy, mogłyby być podobne
    - Niektóre cechy mogą dominować (bardzo informacyjne)
    - Feature randomness wymusza różnorodność

    ## 3. Kluczowe Hiperparametry

    ### n_estimators (Liczba Drzew)

    - **Definicja**: Liczba drzew w lesie
    - **Typowe wartości**: 100-500 (więcej = lepiej, ale wolniej)
    - **Efekt**:
      - Więcej drzew → lepsza stabilność, mniejsza wariancja
      - Po pewnym punkcie (np. 500) zyski są minimalne
      - **Nigdy nie powoduje przeuczenia!** (ale może zbędnie spowalniać)

    ### max_depth (Maksymalna Głębokość)

    - **Definicja**: Maksymalna głębokość każdego drzewa
    - **Typowe wartości**: None (bez limitu), lub 10-30
    - **Efekt**:
      - None → drzewa rosną do pełnej głębokości (powszechne w RF!)
      - Niskie wartości → prostsze drzewa, może niedouczać
      - Random Forest często używa **głębokich drzew** bez problemu przeuczenia

    ### max_features (Liczba Cech w Węźle)

    - **Definicja**: Liczba cech do rozważenia przy podziale węzła
    - **Typowe wartości**:
      - 'sqrt' lub 'auto': √p (dla klasyfikacji) - domyślne
      - 'log2': log₂(p)
      - Liczba lub procent
    - **Efekt**:
      - Mniej cech → większa różnorodność drzew (lepsza dekorelacja)
      - Więcej cech → silniejsze pojedyncze drzewa (ale mogą być podobne)

    ### min_samples_split, min_samples_leaf

    Podobnie jak w pojedynczym drzewie - kontrolują rozrost drzew.

    ## 4. Wady i Zalety

    ### ✅ Zalety:

    - **Wysoka Accuracy** - Często najlepszy standardowy algorytm ML
    - **Odporność na przeuczenie** - Dzięki averaging/voting
    - **Feature Importance** - Automatyczna selekcja cech
    - **Brak wymogu skalowania** - Nie wymaga normalizacji
    - **Obsługa Missing Values** - Może radzić sobie z brakami (w implementacjach)
    - **Out-of-Bag (OOB) Error** - Darmowa walidacja
    - **Paralelizacja** - Drzewa można trenować równolegle

    ### ❌ Wady:

    - **Black Box** - Mniej interpretowalny niż pojedyncze drzewo (setki drzew!)
    - **Rozmiar modelu** - Setki drzew = duży model (pamięć, storage)
    - **Czas predykcji** - Wolniejszy niż pojedyncze drzewo (musi zapytać wszystkie drzewa)
    - **Gorszy dla regresji na ekstrapolacji** - Nie przewiduje poza zakresem danych

    ## 5. Feature Importance w Random Forest

    Random Forest dostarcza **uśrednione feature importances** z wszystkich drzew!
    """)

    st.latex(r"\text{Importance}_{feature} = \frac{1}{N} \sum_{i=1}^{N} \text{Importance}_{feature, tree_i}")

    st.markdown("""
    - Im częściej cecha jest używana do podziału (i im większa redukcja Gini/Entropy)
    - Tym wyższa importance
    - Bardziej **stabilne** niż w pojedynczym drzewie!

    ## 6. Zastosowanie w Bioinformatyce: DTI Prediction

    **DTI (Drug-Target Interaction)** - Przewidywanie czy lek będzie oddziaływał z białkiem docelowym.

    ### Problem

    Odkrywanie nowych leków jest:
    - **Kosztowne**: 2.6 miliarda USD na 1 lek
    - **Czasochłonne**: 10-15 lat
    - **Ryzykowne**: 90% kandydatów na leki odpada

    **Rozwiązanie**: Computational screening - przewidywanie interakcji in silico (na komputerze)
    zamiast testować wszystko eksperymentalnie!

    ### Jak to działa?

    **Dane wejściowe**:
    1. **Deskryptory leku**:
       - Fingerprint molekularny (np. ECFP4) - 1024-wymiarowy wektor binarny
       - Cechy fizykochemiczne (MW, LogP, HBA, HBD)
       - Struktura 2D/3D

    2. **Deskryptory białka**:
       - Sekwencja aminokwasowa → deskryptory (pseudo-AAC, CTD)
       - Struktura 3D (jeśli dostępna)
       - Domeny funkcyjne, motywy

    3. **Target**: Czy lek i białko oddziałują? (1 = Tak, 0 = Nie)

    **Model**: Random Forest Classifier
    """)

    st.latex(r"P(\text{Interaction}) = \text{RandomForest}(\text{Drug Features}, \text{Protein Features})")

    st.markdown("""
    **Trenowanie**:
    - Dane: znane pary Drug-Target z baz danych (ChEMBL, DrugBank)
    - Pozytywne: potwierdzone interakcje
    - Negatywne: brak interakcji (ostrożnie z nieznanymi!)
    - Model: Random Forest z n_estimators=500

    **Predykcja**:
    - Input: nowy lek × znane białko (lub odwrotnie)
    - Output: Prawdopodobieństwo interakcji (0-1)
    - Top kandydaci → walidacja eksperymentalna

    ### Przykładowe Wyniki

    Random Forest w DTI prediction osiąga typowo:
    - **Accuracy**: 85-95%
    - **AUC-ROC**: 0.90-0.98
    - **Przewaga nad**: pojedynczym drzewem, SVM, kNN

    ### Feature Importance → Biological Insights

    Po trenowaniu, feature importances pokazują:
    - **Które cechy leku** są kluczowe (np. hydrophobicity, aromaticity)
    - **Które cechy białka** są kluczowe (np. powierzchnia wiązania, motyw domenowy)
    - **Mechanizmy wiązania** mogą być wnioskowane!

    ### Inne Zastosowania RF w Bioinformatyce

    - **Klasyfikacja chorób** - Na podstawie ekspresji genów, biomarkerów
    - **Variant calling** - Identyfikacja mutacji z danych sekwencyjnych
    - **Protein function prediction** - Przewidywanie funkcji białka
    - **microRNA target prediction** - Przewidywanie celów miRNA

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
    - [RF in Drug Discovery](https://jcheminf.biomedcentral.com/)
    - [Understanding Random Forests](https://www.stat.berkeley.edu/~breiman/RandomForests/)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Breast Cancer Classification")

    st.markdown("""
    Ten demo pokazuje wykorzystanie Random Forest do klasyfikacji nowotworów piersi.
    Porównaj wyniki z pojedynczym drzewem decyzyjnym (poprzednia strona)!

    **Zbiór danych**: Breast Cancer Wisconsin Dataset (569 próbek, 30 cech)
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Ustawienia Demo")

    n_estimators = st.sidebar.slider(
        "Liczba drzew (n_estimators):",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    max_depth_option = st.sidebar.selectbox(
        "Maksymalna głębokość (max_depth):",
        options=['None', '5', '10', '20'],
    )
    max_depth = None if max_depth_option == 'None' else int(max_depth_option)

    max_features = st.sidebar.selectbox(
        "Max features na split:",
        options=['sqrt', 'log2', 'None'],
        index=0,
        help="sqrt: √p features, log2: log₂(p) features, None: wszystkie features"
    )
    max_features = None if max_features == 'None' else max_features

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
    - Zwiększ `n_estimators` → stabilniejszy model
    - `max_depth=None` → pełne drzewa (typowe dla RF)
    - `max_features='sqrt'` → dobra dekorelacja drzew
    """)

    # Load data
    try:
        X, y, feature_names, target_names = load_breast_cancer_data()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train Random Forest
        with st.spinner('Trenowanie Random Forest...'):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
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

        # OOB Score (if oob_score was enabled)
        # Note: We'll calculate it separately to demonstrate
        oob_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        oob_model.fit(X_train, y_train)
        oob_score = oob_model.oob_score_

        # Display info
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Parametry Modelu:**
            - Liczba drzew: **{n_estimators}**
            - Max głębokość: **{max_depth if max_depth else 'Unlimited'}**
            - Max features: **{max_features if max_features else 'All'}**
            - Min próbek do podziału: **{min_samples_split}**
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

        col1, col2, col3, col4, col5 = st.columns(5)

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
                "OOB Score",
                f"{oob_score:.4f}",
                help="Out-of-Bag accuracy (darmowa walidacja!)"
            )
        with col4:
            st.metric(
                "Test F1-Score",
                f"{test_f1:.4f}",
                help="F1-score na zbiorze testowym"
            )
        with col5:
            st.metric(
                "Overfitting Gap",
                f"{overfitting_gap:.4f}",
                delta=f"{-overfitting_gap:.4f}",
                delta_color="inverse",
                help="Różnica między Train i Test Accuracy"
            )

        # Performance interpretation
        if test_accuracy > 0.95:
            st.success(f"""
            ✅ **Doskonała wydajność!**

            Test Accuracy = {test_accuracy:.2%} - Model osiąga znakomitą accuracy na nowych danych.
            Random Forest skutecznie klasyfikuje nowotwory!

            **OOB Score** ({oob_score:.2%}) jest zbliżony do Test Accuracy - dobry znak!
            """)
        elif test_accuracy > 0.90:
            st.success(f"""
            ✅ **Bardzo dobra wydajność!**

            Test Accuracy = {test_accuracy:.2%} - Model działa bardzo dobrze.
            """)

        if overfitting_gap < 0.05:
            st.success("""
            ✅ **Model dobrze generalizuje!**

            Niewielka różnica między Train a Test accuracy. Random Forest skutecznie
            redukuje przeuczenie dzięki ensemble averaging!
            """)
        elif overfitting_gap > 0.15:
            st.warning(f"""
            ⚠️ **Wykryto przeuczenie**

            Gap = {overfitting_gap:.2%}. Możliwe rozwiązania:
            - Zwiększ `min_samples_split`
            - Zmniejsz `max_depth`
            - Zmniejsz `max_features`
            """)

        # Feature Importance
        st.subheader("🔍 Feature Importances (Ważność Cech)")

        st.markdown(f"""
        **Feature importance** uśredniona z **{n_estimators} drzew** - bardziej stabilna
        niż w pojedynczym drzewie!

        Te cechy są najważniejsze dla modelu Random Forest:
        """)

        # Use the helper function from plots.py
        fig_importance = plot_feature_importance(
            model.feature_importances_,
            feature_names,
            top_n=20
        )

        st.plotly_chart(fig_importance, use_container_width=True)

        # Most important features
        top_5_features = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)

        st.markdown("**Top 5 najważniejszych cech:**")
        for idx, row in top_5_features.iterrows():
            st.markdown(f"- **{row['feature']}**: {row['importance']:.4f}")

        st.info("""
        **Biological Interpretation:**

        W klasyfikacji raka piersi, cechy związane z:
        - **Worst concave points** - wklęsłość komórki (wysoka = złośliwa)
        - **Worst perimeter/area** - duże, nieregularne komórki
        - **Mean concave points** - przeciętna wklęsłość

        są najlepszymi **biomarkerami** do odróżnienia nowotworów łagodnych od złośliwych.
        """)

        # Confusion Matrix
        st.subheader("📊 Macierz Pomyłek (Confusion Matrix)")

        col_cm1, col_cm2 = st.columns([1, 1])

        with col_cm1:
            cm = confusion_matrix(y_test, y_test_pred)
            fig_cm = plot_confusion_matrix(cm, target_names)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm2:
            # Calculate detailed metrics
            tn, fp, fn, tp = cm.ravel()

            st.markdown(f"""
            **Interpretacja:**

            - **True Negatives (TN)**: {tn} - Prawidłowo jako Benign
            - **True Positives (TP)**: {tp} - Prawidłowo jako Malignant
            - **False Positives (FP)**: {fp} - Błędnie jako Malignant
            - **False Negatives (FN)**: {fn} - Błędnie jako Benign

            **Sensitivity (Recall)**: {tp/(tp+fn):.2%}
            - Procent złośliwych poprawnie zidentyfikowanych

            **Specificity**: {tn/(tn+fp):.2%}
            - Procent łagodnych poprawnie zidentyfikowanych

            **W diagnostyce medycznej:**
            FN (False Negative) jest **najgorszy** - przegapienie
            raka! Model ma tylko **{fn} FN** - bardzo dobry wynik.
            """)

        # Comparison with Single Tree
        with st.expander("🌳 vs 🌲 Porównanie: Single Tree vs Random Forest"):
            st.markdown("""
            ### Dlaczego Random Forest jest lepszy?

            Wróć do poprzedniej strony (Decision Tree) i porównaj wyniki dla tych samych danych:

            **Pojedyncze Drzewo Decyzyjne** (max_depth=5):
            - Test Accuracy: ~93-95%
            - **Wysokie przeuczenie** dla głębokich drzew
            - **Niestabilne** - różne wyniki dla różnych splitów
            - **Interpretowalny** - można zobaczyć drzewo

            **Random Forest** (100+ drzew):
            - Test Accuracy: ~95-97% ✅
            - **Niskie przeuczenie** - lepsze generalizowanie
            - **Stabilny** - consistent wyniki
            - **Mniej interpretowalny** - las setek drzew

            ### Kiedy użyć czego?

            **Użyj Decision Tree gdy**:
            - Potrzebujesz **interpretowalności** (wyjaśnić lekarzom/regulatorom)
            - Masz proste dane
            - Szybkość predykcji jest kluczowa

            **Użyj Random Forest gdy**:
            - Potrzebujesz **najwyższej accuracy**
            - Możesz poświęcić interpretowalność
            - Masz złożone, zaszumione dane
            - Chcesz **feature importance** (stabilniejsze niż w drzewie)
            """)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Liczba Drzew (n_estimators)**:
           - Zacznij od 10: Zobaczysz niestabilne wyniki
           - 100: Typowa wartość startowa
           - 500: Lepsze, ale wolniejsze
           - Obserwuj: Test accuracy stabilizuje się po ~100-200 drzew

        2. **Max Depth**:
           - None (unlimited): Typowe dla RF, działa dobrze!
           - 5: Bardzo prostsze drzewa
           - Porównaj: RF z głębokimi drzewami rzadko przeuczy się (dzięki bagging)

        3. **Max Features**:
           - 'sqrt': Domyślne, dobra dekorelacja
           - 'log2': Jeszcze większa dekorelacja (więcej różnorodności)
           - None (all): Drzewa mogą być bardziej podobne

        4. **OOB Score**:
           - Obserwuj jak OOB Score jest zbliżony do Test Accuracy
           - To "darmowa" walidacja - nie potrzeba validation set!

        ### 🧬 Zastosowanie w Drug Discovery:

        W przewidywaniu Drug-Target Interactions, Random Forest często osiąga:
        - **95%+ accuracy** na dobrych danych
        - **Feature importance** ujawnia kluczowe cechy leku i białka
        - **Szybka predykcja** dla miliardów par Drug-Target (screening)

        Ten model może przesiewać **miliony związków chemicznych** in silico,
        drastycznie redukując liczbę związków do testów eksperymentalnych!
        """)

        # Data preview
        with st.expander("📋 Podgląd Danych (pierwsze 10 wierszy)"):
            df_display = X.head(10).copy()
            df_display['target'] = y.head(10).map({0: target_names[0], 1: target_names[1]})
            st.dataframe(df_display)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Upewnij się, że funkcja load_breast_cancer_data() działa poprawnie.")
