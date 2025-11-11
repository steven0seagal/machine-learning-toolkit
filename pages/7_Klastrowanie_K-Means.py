"""
Klastrowanie K-Means - K-Means Clustering
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_breast_cancer_data
from src.plots import plot_elbow_curve, plot_silhouette_scores

st.set_page_config(page_title="K-Means Clustering", page_icon="🎯", layout="wide")

st.title("🎯 Klastrowanie K-Means (K-Means Clustering)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest K-Means Clustering?

    K-Means to **unsupervised learning algorithm** (uczenie nienadzorowane) używany do
    **klastrowania** - grupowania danych w homogeniczne klastry (grupy) na podstawie
    podobieństwa.

    ### Kluczowa różnica: Supervised vs Unsupervised

    - **Supervised** (Klasyfikacja, Regresja): Mamy etykiety (labels) → model się uczy przewidywać
    - **Unsupervised** (Klastrowanie): **Nie mamy etykiet** → model znajduje strukturę w danych

    ### Cel K-Means

    Znaleźć **k klastrów** w danych tak, aby:
    - Punkty **w tym samym klastrze** były jak najbardziej podobne (blisko siebie)
    - Punkty **w różnych klastrach** były jak najbardziej różne (daleko od siebie)

    ## 2. Jak działa algorytm K-Means?

    Algorytm K-Means iteracyjnie przypisuje punkty do klastrów i aktualizuje centroidy.

    ### Kroki Algorytmu
    """)

    st.latex(r"\text{Minimalizuj: } J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2")

    st.markdown("""
    Gdzie $C_i$ to klaster $i$, $\\mu_i$ to centroid (środek) klastra $i$.

    **Algorytm (Lloyd's Algorithm)**:

    1. **Inicjalizacja**: Losowo wybierz k punktów jako początkowe centroidy

    2. **Przydział (Assignment)**:
       - Dla każdego punktu $x$: przypisz do najbliższego centroidu
       - Używamy odległości euklidesowej: $d(x, \\mu_i) = ||x - \\mu_i||$

    3. **Aktualizacja (Update)**:
       - Dla każdego klastra: przelicz centroid jako średnią wszystkich punktów w klastrze
       - $\\mu_i = \\frac{1}{|C_i|} \\sum_{x \\in C_i} x$

    4. **Powtarzaj** kroki 2-3 aż:
       - Centroidy przestaną się zmieniać, LUB
       - Osiągnięto maksymalną liczbę iteracji

    ### Wizualizacja Procesu

    ```
    Iteracja 0: [Losowe centroidy]
    Iteracja 1: Przypisz punkty → Przelicz centroidy
    Iteracja 2: Przypisz punkty → Przelicz centroidy (centroidy się przesuwają)
    ...
    Iteracja N: Przypisz punkty → Centroidy nie zmieniają się → STOP
    ```

    Algorytm **zawsze zbiega** (converges), ale do lokalnego minimum (nie zawsze globalnego!).

    ## 3. Wybór Liczby Klastrów (k)

    **Największe wyzwanie w K-Means**: Ile klastrów (k) wybrać?

    Nie ma jednoznacznej odpowiedzi - używamy heurystyk:

    ### 3.1. Elbow Method (Metoda Łokcia)

    **Idea**: Trenuj K-Means dla różnych wartości k (np. 2-10) i rysuj **Inertia** (SSE).
    """)

    st.latex(r"\text{Inertia} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2")

    st.markdown("""
    - **Inertia** = suma kwadratów odległości punktów od ich centroidów (within-cluster sum of squares)
    - **Im niższa Inertia**, tym lepsze dopasowanie (punkty bliżej centroidów)

    **Wykres Inertia vs k**:
    - k=1: Bardzo wysoka Inertia (wszystkie punkty w 1 klastrze)
    - k→∞: Inertia→0 (każdy punkt to osobny klaster)

    **Metoda Łokcia**:
    - Szukaj "łokcia" (elbow) na wykresie
    - Punkt gdzie Inertia zaczyna spadać wolniej
    - To sugerowane **optymalne k**

    ### 3.2. Silhouette Score (Współczynnik Sylwetkowy)
    """)

    st.latex(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}")

    st.markdown("""
    Dla każdego punktu $i$:
    - $a(i)$ = średnia odległość do punktów w **tym samym** klastrze (compactness)
    - $b(i)$ = średnia odległość do punktów w **najbliższym innym** klastrze (separation)

    **Silhouette Score**: $s \\in [-1, 1]$
    - **s ≈ 1**: Punkt jest dobrze dopasowany do swojego klastra
    - **s ≈ 0**: Punkt jest na granicy klastrów
    - **s < 0**: Punkt prawdopodobnie w złym klastrze

    **Average Silhouette Score** (dla wszystkich punktów):
    - Im wyższy (~0.7-1.0), tym lepsze klastrowanie
    - Wybierz **k z najwyższym Silhouette Score**

    ### 3.3. Domain Knowledge (Wiedza dziedzinowa)

    Czasami liczba klastrów jest **znana z góry**:
    - Klastrowanie pacjentów → znamy 3 typy choroby → k=3
    - Klastrowanie genów → znamy 4 grupy funkcyjne → k=4

    ## 4. Wady i Zalety

    ### ✅ Zalety:

    - **Prostota** - Łatwy do zrozumienia i implementacji
    - **Szybkość** - O(nki) gdzie n=punkty, k=klastry, i=iteracje (zazwyczaj <100)
    - **Skalowalność** - Działa na dużych zbiorach danych
    - **Centroids interpretable** - Można interpretować centroidy jako "prototypowe" punkty

    ### ❌ Wady:

    - **Trzeba wybrać k** - Nie ma automatycznego k
    - **Wrażliwość na inicjalizację** - Różne losowe startowe centroidy → różne wyniki
      - Rozwiązanie: `n_init=10` (uruchom 10 razy, wybierz najlepszy)
    - **Zakłada sferyczne klastry** - Nie radzi sobie z nieregularnymi kształtami
    - **Wrażliwość na outliery** - Outliery mocno wpływają na centroidy
    - **Wymaga skalowania** - Cechy o dużych wartościach dominują (→ StandardScaler!)
    - **Tylko odległość Euklidesowa** - Nie radzi sobie z danymi kategorycznymi

    ## 5. Zastosowanie w Bioinformatyce: Gene Expression Clustering

    K-Means jest **bardzo popularny** w analizie ekspresji genów dla odkrywania
    **grup współregulowanych genów** (co-regulated genes).

    ### Problem

    Eksperyment RNA-seq/mikromacierz:
    - **Wiersze**: Geny (np. 20,000 genów)
    - **Kolumny**: Próbki/warunki (np. 10 próbek)
    - **Wartości**: Poziomy ekspresji (FPKM, TPM, log2FC)

    **Pytanie**: Które geny zachowują się podobnie w różnych warunkach?

    ### Zastosowanie K-Means

    1. **Transpozycja**: Geny jako punkty, warunki jako wymiary (features)
       - Każdy gen = punkt w p-wymiarowej przestrzeni (p = liczba warunków)

    2. **Normalizacja**:
       - Z-score normalizacja per gen (mean=0, std=1)
       - Lub log2 transformation

    3. **Klastrowanie**:
       - K-Means z k=5-20 (w zależności od danych)
       - Każdy klaster = grupa genów o podobnej ekspresji

    4. **Interpretacja Klastrów**:
       - **Klaster 1**: Geny upregulated w warunku A (np. stress response)
       - **Klaster 2**: Geny downregulated w warunku B (np. metabolizm)
       - **Klaster 3**: Geny konstytutywne (housekeeping)

    ### Biological Insights
    """)

    st.latex(r"\text{Klaster} \\rightarrow \text{Funkcja Biologiczna (Gene Ontology Enrichment)}")

    st.markdown("""
    **Gene Ontology (GO) Enrichment**:
    - Dla każdego klastra: sprawdź jakie funkcje biologiczne są wzbogacone
    - Przykład:
      - Klaster 1: Enriched for "DNA repair" → geny odpowiedzi na uszkodzenia DNA
      - Klaster 2: Enriched for "cell cycle" → geny kontroli cyklu komórkowego

    ### Przykład: Cancer Subtyping

    K-Means na danych ekspresji genów pacjentów z rakiem:
    - **Dane**: 100 pacjentów × 1000 genów
    - **Klastrowanie**: K-Means z k=3
    - **Wynik**: 3 podtypy raka (subtypes) z różnymi profilami ekspresji
    - **Zastosowanie**: Personalizowana terapia - różne subtypes → różne leki!

    ### Inne Zastosowania w Bioinformatyce

    - **Protein structure clustering** - Grupowanie struktur białek
    - **Patient stratification** - Segmentacja pacjentów na grupy
    - **Sequencing read clustering** - Grupowanie readów DNA/RNA
    - **Metabolomics** - Klastrowanie profili metabolicznych

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    - [Gene Expression Clustering](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184648/)
    - [K-Means in Bioinformatics](https://bmcbioinformatics.biomedcentral.com/)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Breast Cancer Clustering")

    st.markdown("""
    Ten demo pokazuje klastrowanie K-Means na danych Breast Cancer Wisconsin.

    **Uwaga**: To uczenie nienadzorowane - **ignorujemy etykiety** (malignant/benign)
    i patrzymy czy K-Means sam odkryje naturalne grupy w danych!

    Po klastrowaniu **porównamy** odkryte klastry z rzeczywistymi etykietami.
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Ustawienia Demo")

    k = st.sidebar.slider(
        "Liczba klastrów (k):",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    n_init = st.sidebar.selectbox(
        "Liczba inicjalizacji (n_init):",
        options=[1, 10, 20, 50],
        index=1,
        help="Algorytm uruchomi się n_init razy i wybierze najlepszy wynik"
    )

    st.sidebar.markdown("""
    ---
    **Wskazówki:**
    - k=2: Spróbuj odtworzyć 2 klasy (benign/malignant)
    - k=3-5: Zobacz substruktury w danych
    - Zwiększ n_init dla stabilniejszych wyników
    """)

    # Load data
    try:
        X, y, feature_names, target_names = load_breast_cancer_data()

        # Standardize features (CRITICAL for K-Means!)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # K-Means clustering
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, cluster_labels)

        # Compare with true labels (only for k=2)
        if k == 2:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(y, cluster_labels)
            nmi = normalized_mutual_info_score(y, cluster_labels)
        else:
            ari, nmi = None, None

        # Display info
        st.subheader("📊 Wyniki Klastrowania")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Inertia",
                f"{inertia:.2f}",
                help="Suma kwadratów odległości - im niższa, tym lepiej"
            )
        with col2:
            st.metric(
                "Silhouette Score",
                f"{silhouette:.4f}",
                help="Jakość klastrowania: -1 (źle) do 1 (idealnie)"
            )
        with col3:
            if ari is not None:
                st.metric(
                    "ARI vs True Labels",
                    f"{ari:.4f}",
                    help="Adjusted Rand Index: zgodność z prawdziwymi etykietami"
                )
            else:
                st.metric("ARI", "N/A", help="Dostępne tylko dla k=2")
        with col4:
            if nmi is not None:
                st.metric(
                    "NMI vs True Labels",
                    f"{nmi:.4f}",
                    help="Normalized Mutual Information"
                )
            else:
                st.metric("NMI", "N/A", help="Dostępne tylko dla k=2")

        # Interpretation
        if silhouette > 0.5:
            st.success(f"""
            ✅ **Dobre klastrowanie!**

            Silhouette Score = {silhouette:.4f} (>0.5) - Klastry są dobrze rozdzielone i zwarte.
            """)
        elif silhouette > 0.3:
            st.info(f"""
            **Umiarkowane klastrowanie**

            Silhouette Score = {silhouette:.4f} (0.3-0.5) - Klastry są widoczne, ale mogą się nakładać.
            """)
        else:
            st.warning(f"""
            ⚠️ **Słabe klastrowanie**

            Silhouette Score = {silhouette:.4f} (<0.3) - Klastry są słabo rozdzielone.
            Może k jest nieodpowiednie?
            """)

        if k == 2:
            if ari > 0.7:
                st.success(f"""
                ✅ **K-Means odkrył prawdziwe klasy!**

                ARI = {ari:.4f} (>0.7) - Klastry K-Means mocno korelują z prawdziwymi
                etykietami (benign/malignant). To pokazuje, że dane mają naturalną strukturę 2-klasową!
                """)
            elif ari > 0.4:
                st.info(f"""
                **K-Means częściowo odkrył klasy**

                ARI = {ari:.4f} (0.4-0.7) - Umiarkowana zgodność z prawdziwymi etykietami.
                """)
            else:
                st.warning(f"""
                **K-Means nie odkrył klas**

                ARI = {ari:.4f} (<0.4) - Słaba zgodność. Klastry K-Means nie odpowiadają
                prawdziwym etykietom.
                """)

        # Visualization: 2D PCA with clusters
        st.subheader("🎨 Wizualizacja Klastrów (PCA 2D)")

        st.markdown(f"""
        Dane są wysokowymiarowe (30 cech), więc używamy **PCA** do redukcji do 2D dla wizualizacji.

        **Explained Variance**: PC1 + PC2 = {sum(pca.explained_variance_ratio_[:2]):.1%}
        """)

        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': cluster_labels.astype(str),
            'True Label': y.map({0: target_names[0], 1: target_names[1]})
        })

        # Plot with cluster colors
        fig_cluster = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'K-Means Clustering (k={k}) - Cluster Labels',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
            hover_data=['True Label'],
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Add centroids
        centroids_pca = pca.transform(scaler.transform(
            scaler.inverse_transform(kmeans.cluster_centers_)
        ))
        fig_cluster.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers',
            marker=dict(size=20, symbol='x', color='black', line=dict(width=2)),
            name='Centroids',
            showlegend=True
        ))

        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Plot with true labels for comparison
        if k == 2:
            st.markdown("**Porównanie z prawdziwymi etykietami:**")

            fig_true = px.scatter(
                df_plot,
                x='PC1',
                y='PC2',
                color='True Label',
                title='Prawdziwe Etykiety (Benign/Malignant)',
                labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                color_discrete_map={target_names[0]: 'blue', target_names[1]: 'red'}
            )
            fig_true.update_layout(height=500)
            st.plotly_chart(fig_true, use_container_width=True)

            st.info("""
            **Porównaj oba wykresy**:
            - Czy kolory w "Cluster Labels" odpowiadają kolorom w "True Labels"?
            - Jeśli tak → K-Means skutecznie odkrył naturalne klasy!
            - Jeśli nie → Dane mogą mieć bardziej złożoną strukturę
            """)

        # Elbow Method
        st.subheader("📉 Elbow Method - Wybór Optymalnego k")

        st.markdown("""
        Trenujemy K-Means dla różnych wartości k i rysujemy **Inertia vs k**.
        Szukamy "łokcia" na wykresie.
        """)

        with st.spinner('Obliczanie Elbow Plot...'):
            k_range = range(2, 11)
            inertias = []

            for k_test in k_range:
                kmeans_test = KMeans(n_clusters=k_test, n_init=10, random_state=42)
                kmeans_test.fit(X_scaled)
                inertias.append(kmeans_test.inertia_)

            fig_elbow = plot_elbow_curve(inertias, k_range)

            # Highlight current k
            current_k_idx = k - 2 if k >= 2 and k <= 10 else None
            if current_k_idx is not None and current_k_idx < len(inertias):
                fig_elbow.add_trace(go.Scatter(
                    x=[k],
                    y=[inertias[current_k_idx]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Aktualny k={k}',
                    showlegend=True
                ))

            st.plotly_chart(fig_elbow, use_container_width=True)

            st.markdown("""
            **Interpretacja**:
            - Inertia zawsze maleje gdy k rośnie (więcej klastrów = mniejsze błędy)
            - Szukamy "łokcia" - punktu gdzie krzywa zaczyna wypłaszczać się
            - Dla tych danych: łokieć często przy **k=2** lub **k=3**
            """)

        # Silhouette Score Plot
        st.subheader("📊 Silhouette Score - Wybór Optymalnego k")

        st.markdown("""
        Obliczamy Silhouette Score dla różnych wartości k.
        **Wyższy score = lepsze klastrowanie**.
        """)

        with st.spinner('Obliczanie Silhouette Scores...'):
            silhouette_scores = []

            for k_test in k_range:
                kmeans_test = KMeans(n_clusters=k_test, n_init=10, random_state=42)
                labels_test = kmeans_test.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels_test)
                silhouette_scores.append(score)

            fig_silhouette = plot_silhouette_scores(silhouette_scores, k_range)

            # Highlight current k
            if current_k_idx is not None and current_k_idx < len(silhouette_scores):
                fig_silhouette.add_trace(go.Scatter(
                    x=[k],
                    y=[silhouette_scores[current_k_idx]],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='diamond'),
                    name=f'Aktualny k={k}',
                    showlegend=True
                ))

            st.plotly_chart(fig_silhouette, use_container_width=True)

            optimal_k = list(k_range)[np.argmax(silhouette_scores)]
            max_silhouette = max(silhouette_scores)

            st.info(f"""
            **Optymalne k według Silhouette Score**: **k={optimal_k}** (score={max_silhouette:.4f})

            Czerwona gwiazdka pokazuje k z najwyższym Silhouette Score.
            """)

        # Cluster sizes
        with st.expander("📊 Rozmiary Klastrów"):
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Liczba punktów w każdym klastrze:**")
                for cluster_id, size in cluster_sizes.items():
                    st.markdown(f"- **Klaster {cluster_id}**: {size} próbek ({size/len(cluster_labels)*100:.1f}%)")

            with col2:
                fig_sizes = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    labels={'x': 'Klaster', 'y': 'Liczba Próbek'},
                    title='Rozmiary Klastrów'
                )
                st.plotly_chart(fig_sizes, use_container_width=True)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Eksploruj różne k**:
           - k=2: Zobacz czy K-Means odkryje 2 klasy (benign/malignant)
           - k=3-5: Może istnieją subtypy w danych?
           - Użyj Elbow i Silhouette plots jako wskazówek

        2. **Porównaj z prawdziwymi etykietami**:
           - Dla k=2: Sprawdź ARI score
           - Czy klastry odpowiadają benign/malignant?

        3. **Obserwuj Silhouette Score**:
           - >0.7: Doskonałe klastrowanie
           - 0.5-0.7: Dobre
           - 0.3-0.5: Umiarkowane
           - <0.3: Słabe

        4. **Stabilność**:
           - Zmień `n_init` na 1 → zobaczysz różne wyniki przy różnych uruchomieniach
           - Zwiększ do 10-50 → stabilniejsze wyniki

        ### 🧬 Zastosowanie w Gene Expression:

        W analizie ekspresji genów, K-Means może odkryć:
        - **Subtypes nowotworów** - Pacjenci z podobnymi profilami ekspresji
        - **Co-regulated genes** - Geny które są razem upregulated/downregulated
        - **Treatment groups** - Pacjenci którzy odpowiadają podobnie na terapię

        **Przykład**: Breast cancer ma subtypes (Luminal A, Luminal B, HER2+, Triple-negative).
        K-Means na danych ekspresji genów może je odkryć **bez etykiet**!

        ### 🔍 Dlaczego potrzebujemy PCA?

        - Oryginalne dane: 30 wymiarów (cech)
        - Ludzkie oko: 2-3 wymiary
        - **PCA redukuje 30D → 2D** zachowując jak najwięcej informacji
        - To tylko wizualizacja! K-Means pracuje na oryginalnych 30 cechach
        """)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Upewnij się, że funkcja load_breast_cancer_data() działa poprawnie.")
