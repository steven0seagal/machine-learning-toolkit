"""
Analiza Głównych Składowych (PCA) - Principal Component Analysis
Educational page with theory and interactive demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders import load_gene_expression_cancer
from src.plots import plot_pca_scree
from src.navigation import render_sidebar_navigation

st.set_page_config(page_title="PCA", page_icon="📐", layout="wide")

# Render sidebar navigation
render_sidebar_navigation()

st.title("📐 Analiza Głównych Składowych (PCA)")

# Create tabs
tab_teoria, tab_demo = st.tabs(["📚 Teoria i Zastosowania", "🎮 Interaktywna Demonstracja"])

with tab_teoria:
    st.header("Teoria i Zastosowania w Bioinformatyce")

    st.markdown("""
    ## 1. Czym jest PCA?

    **Principal Component Analysis (PCA)** to technika **unsupervised learning** używana do
    **redukcji wymiarowości** (dimensionality reduction). PCA transformuje dane z przestrzeni
    wysokowymiarowej do przestrzeni niskwymiarowej, **zachowując jak najwięcej informacji**.

    ### Problem: Klątwa Wymiarowości (Curse of Dimensionality)

    W bioinformatyce często mamy:
    - **Wysokowymiarowe dane**: 20,000+ genów (cech), ale tylko 100-1000 próbek
    - **p >> n**: Liczba cech >> liczba obserwacji
    - **Trudności**:
      - Wizualizacja niemożliwa (nie widzimy 20,000 wymiarów!)
      - Przeuczenie modeli ML
      - Zwiększona złożoność obliczeniowa
      - Szum w danych

    **Rozwiązanie**: PCA redukuje p=20,000 → k=2-50, zachowując większość wariancji!

    ## 2. Jak działa PCA?

    PCA znajduje **nowe osie** (Principal Components) w przestrzeni danych, wzdłuż których
    dane mają **największą wariancję**.

    ### Kluczowa Idea
    """)

    st.latex(r"\text{PC}_1 = \text{kierunek z maksymalną wariancją}")
    st.latex(r"\text{PC}_2 = \text{kierunek z maksymalną wariancją, prostopadły do PC}_1")
    st.latex(r"\text{PC}_3 = \text{kierunek z maksymalną wariancją, prostopadły do PC}_1, \text{PC}_2")

    st.markdown("""
    I tak dalej...

    **Principal Components (PC)** to:
    - **Nowe osie współrzędnych** (liniowe kombinacje oryginalnych cech)
    - **Ortogonalne** (prostopadłe) do siebie
    - **Uporządkowane** według wyjaśnianej wariancji (PC1 > PC2 > PC3 > ...)

    ### Matematyka PCA (Uproszczona)

    Dane: Macierz $X$ (n próbek × p cech), wycentrowane (mean=0)

    1. **Oblicz macierz kowariancji**: $C = \\frac{1}{n-1} X^T X$

    2. **Znajdź wektory własne (eigenvectors) i wartości własne (eigenvalues)** macierzy C:
    """)

    st.latex(r"C \mathbf{v}_i = \lambda_i \mathbf{v}_i")

    st.markdown("""
    - $\\mathbf{v}_i$ = i-ty wektor własny = kierunek i-tego PC
    - $\\lambda_i$ = i-ta wartość własna = wariancja wzdłuż i-tego PC

    3. **Sortuj według wartości własnych**: $\\lambda_1 > \\lambda_2 > ... > \\lambda_p$

    4. **Wybierz k pierwszych PC** (np. k=2, 10, 50)

    5. **Transformuj dane**:
    """)

    st.latex(r"X_{PCA} = X \cdot V_k")

    st.markdown("""
    Gdzie $V_k$ to macierz k pierwszych wektorów własnych.

    **Wynik**: $X_{PCA}$ (n próbek × k PC) - dane w nowej, niskwymiarowej przestrzeni!

    ## 3. Explained Variance (Wyjaśniona Wariancja)

    Każdy PC wyjaśnia pewien **procent całkowitej wariancji** w danych.
    """)

    st.latex(r"\text{Explained Variance Ratio (PC}_i) = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}")

    st.markdown("""
    - **PC1** zazwyczaj wyjaśnia największy procent (np. 40-60%)
    - **PC2** wyjaśnia kolejny procent (np. 15-25%)
    - **PC3, PC4, ...** wyjaśniają coraz mniej
    - **Suma** wszystkich PC = 100% wariancji

    ### Cumulative Explained Variance (Wariancja Skumulowana)

    Suma wyjaśnianej wariancji przez pierwsze k PC:
    """)

    st.latex(r"\text{Cumulative Variance}(k) = \sum_{i=1}^{k} \text{EVR}(\text{PC}_i)")

    st.markdown("""
    **Pytanie**: Ile PC wybrać?

    **Typowa heurystyka**:
    - Wybierz k tak, aby **cumulative variance ≥ 90%** (lub 95%)
    - Przykład: Jeśli PC1-PC10 wyjaśniają 92% wariancji → używamy k=10

    **Scree Plot**: Wykres Explained Variance vs PC → szukamy "łokcia"

    ## 4. Wady i Zalety

    ### ✅ Zalety:

    - **Redukcja wymiarowości** - p=20,000 → k=50 (lub nawet k=2 dla wizualizacji!)
    - **Usuwanie szumu** - Niskie PC często zawierają szum → odrzucamy je
    - **Wizualizacja** - PC1 vs PC2 plot pozwala zobaczyć strukturę danych
    - **Przyspieszenie ML** - Mniej cech → szybsze trenowanie
    - **Redukcja przeuczenia** - Mniej cech → mniejsze ryzyko overfittingu
    - **Odkrywanie struktur** - PC mogą odpowiadać ukrytym czynnikom biologicznym

    ### ❌ Wady:

    - **Utrata interpretowalności** - PC to liniowe kombinacje cech (trudne do interpretacji)
      - Przykład: PC1 = 0.2×Gen1 + 0.15×Gen2 + ... (co to znaczy biologicznie?)
    - **Zakłada liniowość** - PCA działa najlepiej gdy zależności są liniowe
    - **Wrażliwość na skalę** - **Wymaga standaryzacji** (StandardScaler)!
    - **Utrata informacji** - Odrzucamy PC z niską wariancją (mogą zawierać coś ważnego)
    - **Nie supervised** - PCA nie wie o target (może wyrzucić ważne dla predykcji PC)

    ## 5. PCA w praktyce: Preprocessing

    ### KRYTYCZNE: Standaryzacja!

    PCA jest wrażliwe na skalę cech. Cechy o dużych wartościach dominują!

    **Przykład**:
    - Gen A: wartości 0-1000 (expression counts)
    - Gen B: wartości 0-1 (normalized)

    Bez standaryzacji: PC1 będzie prawie całkowicie zdominowany przez Gen A!

    **Rozwiązanie**: Zawsze używaj `StandardScaler` przed PCA:
    """)

    st.latex(r"x_{scaled} = \frac{x - \mu}{\sigma}")

    st.markdown("""
    (mean=0, std=1 dla każdej cechy)

    ## 6. Zastosowanie w Bioinformatyce: Genomics Visualization

    PCA jest **najczęściej używaną** techniką do wizualizacji danych genomicznych!

    ### Przypadek użycia: Gene Expression Cancer Data

    **Dane**:
    - 800 próbek pacjentów
    - 20,000 genów (ekspresja z RNA-seq)
    - 5 typów raka: BRCA (piersi), KIRC (nerka), COAD (okrężnica), LUAD (płuco), PRAD (prostata)

    **Problem**: Jak wizualizować 20,000-wymiarowe dane?

    **Rozwiązanie PCA**:

    1. **Standaryzacja**: StandardScaler na 20,000 genów

    2. **PCA**: Redukcja 20,000 → 50 PC (zachowujemy 90% wariancji)

    3. **Wizualizacja**: Scatter plot PC1 vs PC2, kolorowany typem raka

    ### Co możemy zobaczyć na PC1 vs PC2 plot?

    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Dobrze rozdzielone klastry**:
        - Różne typy raka tworzą oddzielne grupy
        - PC1 i PC2 wychwytują **główne różnice** między typami
        - Sugeruje silne sygnały genomiczne
        - ML modele będą działać dobrze!
        """)

    with col2:
        st.markdown("""
        **Nakładające się klastry**:
        - Typy raka są mieszane
        - PC1 i PC2 nie wystarczają (spróbuj PC3, PC4)
        - Może potrzeba więcej PC lub innych metod (t-SNE, UMAP)
        - Klasyfikacja będzie trudniejsza
        """)

    st.markdown("""
    ### Interpretacja PC w kontekście biologicznym

    Czasami PC odpowiadają **znanym czynnikom biologicznym**:

    - **PC1** może reprezentować "cell cycle phase" (faza cyklu komórkowego)
    - **PC2** może reprezentować "tissue type" (typ tkanki)
    - **PC3** może reprezentować "batch effect" (efekt serii eksperymentu)

    **Loadings** (wagi genów w PC) pokazują, które geny najbardziej przyczyniają się do PC:
    - Geny z wysokimi loadings w PC1 → kluczowe dla największej zmienności
    - Analiza tych genów (Gene Ontology) → biological insights!

    ### Inne zastosowania PCA w bioinformatyce:

    - **Population genetics**: Wizualizacja struktur populacyjnych z danych SNP
    - **Proteomics**: Redukcja wymiarowości profili białkowych
    - **Metabolomics**: Identyfikacja głównych wzorców metabolicznych
    - **Quality control**: Wykrywanie outliers, batch effects
    - **Feature extraction**: Pre-processing przed ML (np. przed Random Forest)

    ### PCA vs t-SNE vs UMAP

    | Metoda | Cel | Zachowuje | Interpretacja |
    |--------|-----|-----------|---------------|
    | **PCA** | Redukcja liniowa | Struktury globalne | PC interpretowalny (liniowa kombinacja) |
    | **t-SNE** | Wizualizacja nieliniowa | Struktury lokalne | Odległości nie mają znaczenia |
    | **UMAP** | Wizualizacja nieliniowa | Lokalne + globalne | Szybsze niż t-SNE |

    **Wybór**:
    - **Wizualizacja 2D**: t-SNE lub UMAP (lepsze rozdzielenie)
    - **Feature extraction dla ML**: PCA (zachowuje globalne struktury)
    - **Interpretacja**: PCA (PC mają znaczenie matematyczne)

    ---

    ## 📖 Dodatkowe Zasoby
    - [Scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
    - [PCA in Genomics](https://www.nature.com/articles/nbt0308-303)
    - [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
    """)

with tab_demo:
    st.header("Interaktywna Demonstracja: Gene Expression Cancer Data")

    st.markdown("""
    Ten demo pokazuje PCA na danych ekspresji genów z różnych typów raka.

    **Zbiór danych**: Gene Expression Cancer RNA-Seq
    - 801 próbek pacjentów
    - 100 genów (uproszczone z ~20,000 dla demonstracji)
    - 5 typów raka: BRCA, KIRC, COAD, LUAD, PRAD
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Ustawienia Demo")

    # Load data first to get n_components
    try:
        X, y = load_gene_expression_cancer()

        # Standardize features (CRITICAL!)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine max PCs (min of samples or features)
        max_pcs = min(X_scaled.shape[0], X_scaled.shape[1], 50)

        # Fit PCA with max components
        pca_full = PCA(n_components=max_pcs, random_state=42)
        X_pca_full = pca_full.fit_transform(X_scaled)

        # UI controls
        n_components = st.sidebar.slider(
            "Liczba PC do obliczenia:",
            min_value=2,
            max_value=max_pcs,
            value=min(10, max_pcs),
            step=1,
            help="Liczba głównych składowych do analizy"
        )

        pc_x = st.sidebar.selectbox(
            "Oś X (PC):",
            options=[f"PC{i+1}" for i in range(n_components)],
            index=0
        )

        pc_y = st.sidebar.selectbox(
            "Oś Y (PC):",
            options=[f"PC{i+1}" for i in range(n_components)],
            index=1 if n_components > 1 else 0
        )

        st.sidebar.markdown("""
        ---
        **Wskazówki:**
        - PC1 vs PC2: Największa wariancja
        - Spróbuj PC2 vs PC3, PC1 vs PC3
        - Szukaj separacji typów raka
        """)

        # Extract PC indices
        pc_x_idx = int(pc_x.replace('PC', '')) - 1
        pc_y_idx = int(pc_y.replace('PC', '')) - 1

        # Dataset info
        st.subheader("📊 Informacje o Danych")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Liczba Próbek", len(X))
        with col2:
            st.metric("Liczba Genów", X.shape[1])
        with col3:
            st.metric("Liczba Typów Raka", len(y.unique()))
        with col4:
            st.metric("Liczba PC", n_components)

        # Show cancer types distribution
        cancer_counts = y.value_counts()

        st.markdown("**Rozkład typów raka w zbiorze:**")
        col_dist1, col_dist2 = st.columns([2, 1])

        with col_dist1:
            fig_dist = px.bar(
                x=cancer_counts.index,
                y=cancer_counts.values,
                labels={'x': 'Typ Raka', 'y': 'Liczba Próbek'},
                title='Rozkład Typów Raka'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_dist2:
            st.markdown("")
            st.markdown("")
            for cancer_type, count in cancer_counts.items():
                st.markdown(f"- **{cancer_type}**: {count} próbek")

        # PCA Visualization
        st.subheader(f"🎨 Wizualizacja PCA: {pc_x} vs {pc_y}")

        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            pc_x: X_pca_full[:, pc_x_idx],
            pc_y: X_pca_full[:, pc_y_idx],
            'Cancer Type': y
        })

        # Calculate explained variance for selected PCs
        explained_var_x = pca_full.explained_variance_ratio_[pc_x_idx] * 100
        explained_var_y = pca_full.explained_variance_ratio_[pc_y_idx] * 100

        fig_pca = px.scatter(
            df_plot,
            x=pc_x,
            y=pc_y,
            color='Cancer Type',
            title=f'PCA: {pc_x} ({explained_var_x:.1f}% variance) vs {pc_y} ({explained_var_y:.1f}% variance)',
            labels={
                pc_x: f'{pc_x} ({explained_var_x:.1f}% var)',
                pc_y: f'{pc_y} ({explained_var_y:.1f}% var)'
            },
            hover_data={'Cancer Type': True},
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        fig_pca.update_layout(height=600)

        st.plotly_chart(fig_pca, use_container_width=True)

        # Interpretation
        st.markdown("""
        **Interpretacja:**
        - Każdy punkt = pacjent
        - Kolor = typ raka
        - **Dobrze rozdzielone klastry** → różne typy mają różne profile ekspresji genów
        - **Nakładające się klastry** → typy są genomicznie podobne
        """)

        # Explained Variance
        st.subheader("📈 Explained Variance (Wyjaśniona Wariancja)")

        st.markdown(f"""
        Każdy PC wyjaśnia pewien procent całkowitej wariancji w danych.
        Obliczono {n_components} PC.
        """)

        # Individual explained variance
        col_var1, col_var2 = st.columns(2)

        with col_var1:
            st.markdown(f"""
            **Wybrane PC:**
            - **{pc_x}**: {explained_var_x:.2f}% wariancji
            - **{pc_y}**: {explained_var_y:.2f}% wariancji
            - **Razem**: {explained_var_x + explained_var_y:.2f}% wariancji
            """)

        with col_var2:
            st.markdown(f"""
            **Top 3 PC:**
            - **PC1**: {pca_full.explained_variance_ratio_[0]*100:.2f}% wariancji
            - **PC2**: {pca_full.explained_variance_ratio_[1]*100:.2f}% wariancji
            - **PC3**: {pca_full.explained_variance_ratio_[2]*100:.2f}% wariancji
            """)

        # Bar chart of explained variance
        explained_var_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(n_components)],
            'Explained Variance (%)': pca_full.explained_variance_ratio_[:n_components] * 100
        })

        fig_var = px.bar(
            explained_var_df,
            x='PC',
            y='Explained Variance (%)',
            title=f'Explained Variance per Principal Component',
            labels={'Explained Variance (%)': 'Explained Variance (%)'}
        )
        fig_var.update_layout(height=400)

        st.plotly_chart(fig_var, use_container_width=True)

        # Cumulative explained variance (Scree Plot)
        st.subheader("📉 Scree Plot (Cumulative Explained Variance)")

        st.markdown("""
        **Scree Plot** pokazuje skumulowaną wyjaśnioną wariancję.
        Używamy go do decyzji: **ile PC wybrać?**

        Typowa heurystyka: Wybierz k PC, aby wyjaśnić ≥90% (lub 95%) wariancji.
        """)

        fig_scree = plot_pca_scree(
            pca_full.explained_variance_ratio_[:n_components],
            cumulative=True
        )

        st.plotly_chart(fig_scree, use_container_width=True)

        # Calculate how many PCs for 90% variance
        cumsum = np.cumsum(pca_full.explained_variance_ratio_[:n_components])
        n_pcs_90 = np.argmax(cumsum >= 0.90) + 1 if any(cumsum >= 0.90) else n_components
        variance_at_threshold = cumsum[n_pcs_90 - 1] * 100 if n_pcs_90 <= len(cumsum) else cumsum[-1] * 100

        st.info(f"""
        **Rekomendacja**: Aby wyjaśnić ≥90% wariancji, potrzebujesz **{n_pcs_90} PC**
        (wyjaśniają {variance_at_threshold:.1f}% wariancji).

        Wszystkie {n_components} PC razem wyjaśniają {cumsum[-1]*100:.1f}% wariancji.
        """)

        # PC Loadings (Top contributing genes)
        with st.expander("🔬 PC Loadings - Które geny najbardziej przyczyniają się do PC?"):
            st.markdown(f"""
            **Loadings** to wagi genów w liniowej kombinacji definiującej PC.

            Dla wybranego PC ({pc_x}), pokażemy geny z najwyższymi loadings (dodatnimi i ujemnymi).
            """)

            selected_pc_for_loadings = st.selectbox(
                "Wybierz PC do analizy loadings:",
                options=[f"PC{i+1}" for i in range(n_components)],
                index=0,
                key='loadings_pc'
            )

            pc_loadings_idx = int(selected_pc_for_loadings.replace('PC', '')) - 1

            # Get loadings for this PC
            loadings = pca_full.components_[pc_loadings_idx]
            gene_names = X.columns if isinstance(X, pd.DataFrame) else [f"Gene_{i}" for i in range(X.shape[1])]

            loadings_df = pd.DataFrame({
                'Gene': gene_names,
                'Loading': loadings,
                'Abs Loading': np.abs(loadings)
            }).sort_values('Abs Loading', ascending=False)

            # Top 10 positive and negative
            top_10 = loadings_df.head(10)

            col_load1, col_load2 = st.columns(2)

            with col_load1:
                st.markdown(f"**Top 10 genów dla {selected_pc_for_loadings}:**")
                st.dataframe(top_10[['Gene', 'Loading']], use_container_width=True, hide_index=True)

            with col_load2:
                fig_loadings = px.bar(
                    top_10,
                    x='Loading',
                    y='Gene',
                    orientation='h',
                    title=f'Top 10 Gene Loadings for {selected_pc_for_loadings}',
                    color='Loading',
                    color_continuous_scale='RdBu_r'
                )
                fig_loadings.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_loadings, use_container_width=True)

            st.markdown("""
            **Interpretacja**:
            - **Wysokie dodatnie loadings** → gen jest silnie skorelowany z PC
            - **Wysokie ujemne loadings** → gen jest silnie anty-skorelowany z PC
            - Te geny najbardziej definiują dany PC

            **Biological insight**: Analiza tych genów (np. Gene Ontology Enrichment)
            może ujawnić, jakie procesy biologiczne reprezentuje PC!
            """)

        # 3D Visualization (optional)
        if n_components >= 3:
            with st.expander("🎲 Wizualizacja 3D (PC1, PC2, PC3)"):
                st.markdown("""
                Wizualizacja 3D pierwszych trzech głównych składowych.
                Możesz obracać wykres myszką!
                """)

                df_3d = pd.DataFrame({
                    'PC1': X_pca_full[:, 0],
                    'PC2': X_pca_full[:, 1],
                    'PC3': X_pca_full[:, 2],
                    'Cancer Type': y
                })

                fig_3d = px.scatter_3d(
                    df_3d,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='Cancer Type',
                    title='3D PCA Visualization (PC1, PC2, PC3)',
                    labels={
                        'PC1': f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)',
                        'PC2': f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)',
                        'PC3': f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%)'
                    },
                    color_discrete_sequence=px.colors.qualitative.Set1
                )

                fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
                fig_3d.update_layout(height=700)

                st.plotly_chart(fig_3d, use_container_width=True)

        # Experimentation tips
        st.markdown("""
        ---
        ### 💡 Wskazówki do eksperymentowania:

        1. **Eksploruj różne pary PC**:
           - PC1 vs PC2: Zazwyczaj najlepsza separacja
           - PC2 vs PC3, PC1 vs PC3: Inne perspektywy na dane
           - Wyższe PC (PC5, PC6): Często zawierają szum, ale czasami ciekawe wzorce

        2. **Explained Variance**:
           - PC1 wyjaśnia najwięcej (często 30-60%)
           - Obserwuj jak szybko spada wariancja dla wyższych PC
           - Ile PC potrzebujesz do 90% wariancji?

        3. **Separacja typów raka**:
           - Czy typy tworzą oddzielne klastry?
           - Które typy są najbardziej podobne genomicznie?
           - Czy outliers (odstające punkty) istnieją?

        4. **Loadings**:
           - Które geny definiują PC1?
           - Możesz użyć tych genów jako **biomarkerów**!

        ### 🧬 Zastosowanie w rzeczywistości:

        **Cancer Subtyping**:
        - Dane: 20,000 genów × 1000 pacjentów
        - PCA: Redukcja do 50 PC (90% wariancji)
        - Wizualizacja: PC1 vs PC2 → odkrycie subtypów
        - ML: Użyj 50 PC jako features dla Random Forest → klasyfikacja

        **Quality Control**:
        - Outliers na PC plot → potencjalnie złe próbki (kontaminacja, błąd techniczny)
        - Batch effects → próbki grupują się według daty eksperymentu (nie biologii!)

        **Feature Selection**:
        - Geny z wysokimi loadings w PC1-PC5 → ważne geny
        - Użyj tylko tych genów dla uproszczonych modeli

        ### 🔬 Biologiczna interpretacja PC:

        W genomice, PC często odpowiadają znanym czynnikom:
        - **PC1**: Cell proliferation (proliferacja komórkowa)
        - **PC2**: Immune response (odpowiedź immunologiczna)
        - **PC3**: Tissue-specific signatures (sygnatura tkankowa)

        Analiza loadings (Gene Ontology Enrichment) może to potwierdzić!
        """)

        # Data preview
        with st.expander("📋 Podgląd Danych (pierwsze 5 próbek × 10 genów)"):
            df_display = X.head(5).iloc[:, :10].copy()
            df_display.insert(0, 'Cancer Type', y.head(5).values)
            st.dataframe(df_display)

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {str(e)}")
        st.info("Upewnij się, że funkcja load_gene_expression_cancer() działa poprawnie.")
