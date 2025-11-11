# 🧬 Platforma Edukacyjna Uczenia Maszynowego w Bioinformatyce

> Interaktywna platforma edukacyjna do nauki algorytmów ML z zastosowaniami w bioinformatyce, zbudowana w Streamlit

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Spis Treści

- [Opis Projektu](#opis-projektu)
- [Funkcjonalności](#funkcjonalności)
- [Architektura](#architektura)
- [Instalacja](#instalacja)
- [Uruchomienie](#uruchomienie)
- [Algorytmy](#algorytmy)
- [Zbiory Danych](#zbiory-danych)
- [Struktura Projektu](#struktura-projektu)
- [Technologie](#technologie)
- [Użytkowanie](#użytkowanie)
- [Rozwój](#rozwój)
- [Licencja](#licencja)

## 🎯 Opis Projektu

Platforma Edukacyjna Uczenia Maszynowego w Bioinformatyce to interaktywna aplikacja webowa stworzona w celu nauki i zrozumienia algorytmów ML w kontekście zastosowań bioinformatycznych.

### Trzy Filary Platformy

1. **📚 Baza Wiedzy** - Szczegółowe wyjaśnienia teoretyczne każdego algorytmu
2. **🎮 Interaktywne Demonstracje** - Wizualizacje i eksperymenty w czasie rzeczywistym
3. **🔬 Narzędzie BYOD** - "Bring Your Own Data" - analiza własnych zbiorów danych

## ✨ Funkcjonalności

### Dla Każdego Algorytmu:

- ✅ **Teoria i Matematyka** - Kompletne wyjaśnienia z formułami LaTeX
- ✅ **Zastosowania w Bioinformatyce** - Rzeczywiste przykłady użycia (QSAR, GWAS, klasyfikacja białek, itp.)
- ✅ **Interaktywne Parametry** - Suwaki i selectboxy do eksperymentowania
- ✅ **Wizualizacje Plotly** - Interaktywne wykresy (granice decyzyjne, PCA, feature importance)
- ✅ **Metryki Ewaluacji** - Dokładność, F1, R², Silhouette, itp.
- ✅ **Porady Edukacyjne** - Wskazówki do eksperymentowania

### Narzędzie BYOD (Przeanalizuj Własne Dane):

- 📤 **Upload CSV** - Wgrywanie własnych zbiorów danych
- 🎯 **Definicja Zmiennych** - Wybór target i features
- 🔧 **Preprocessing** - Imputacja brakujących danych, skalowanie
- 🤖 **Wybór Modelu** - Wszystkie algorytmy z dynamicznymi hiperparametrami
- 📊 **Wyniki** - Metryki, wizualizacje, confusion matrix
- 💾 **Eksport** - Pobieranie predykcji do CSV

## 🏗️ Architektura

### Struktura Aplikacji (MPA - Multi-Page App)

```
streamlit_app.py          # 🏠 Strona główna
├── pages/                # 📑 Strony algorytmów (auto-navigation)
│   ├── 1_Regresja_Liniowa.py
│   ├── 2_Regresja_Logistyczna.py
│   ├── 3_kNajblizszych_Sasiadow_kNN.py
│   ├── 4_Maszyny_Wektorow_Nosnych_SVM.py
│   ├── 5_Drzewa_Decyzyjne.py
│   ├── 6_Las_Losowy.py
│   ├── 7_Klastrowanie_K-Means.py
│   ├── 8_Analiza_Glownych_Skladowych_PCA.py
│   └── 9_Analizuj_Wlasne_Dane.py
├── src/                  # 🔧 Moduły pomocnicze
│   ├── data_loaders.py   # Ładowanie i cachowanie danych
│   ├── plots.py          # Funkcje wizualizacji Plotly
│   └── ml_models.py      # Wrappery modeli scikit-learn
└── data/                 # 📊 Zbiory danych (CSV)
    ├── qsar_fish_toxicity.csv
    └── gene_expression_cancer_rna_seq.csv
```

### Separacja Logiki i UI

- **UI**: Pliki `pages/*.py` - Streamlit widgets i layout
- **Logika**: Moduły `src/*.py` - Czysty Python, reużywalny kod
- **Dane**: `data/*.csv` - Statyczne zbiory danych z cachingiem

## 🚀 Instalacja

### Wymagania

- Python 3.8 lub nowszy
- pip

### Krok 1: Klonowanie Repozytorium

```bash
git clone <repository-url>
cd machine-learning-toolkit
```

### Krok 2: Instalacja Zależności

```bash
pip install -r requirements.txt
```

**Zależności:**
- `streamlit>=1.28.0` - Framework aplikacji webowej
- `pandas>=2.0.0` - Manipulacja danymi
- `numpy>=1.24.0` - Operacje numeryczne
- `scikit-learn>=1.3.0` - Algorytmy ML
- `plotly>=5.17.0` - Wizualizacje interaktywne
- `matplotlib>=3.7.0` - Wizualizacje (drzewa decyzyjne)

## ▶️ Uruchomienie

```bash
streamlit run streamlit_app.py
```

Aplikacja otworzy się automatycznie w przeglądarce pod adresem: `http://localhost:8501`

### Alternatywnie (z określonym portem):

```bash
streamlit run streamlit_app.py --server.port 8080
```

## 🧠 Algorytmy

### Uczenie Nadzorowane - Regresja

#### 1. **Regresja Liniowa**
- **Teoria**: OLS, RSS, założenia modelu (liniowość, homoskedastyczność)
- **Zastosowanie**: QSAR (Quantitative Structure-Activity Relationship)
- **Dataset**: QSAR Fish Toxicity (908 związków, 6 deskryptorów molekularnych)
- **Demo**: Wybór deskryptora, wizualizacja linii regresji, R²/MAE/MSE
- **Interaktywność**: Selectbox deskryptora → auto-update wykresu i metryk

### Uczenie Nadzorowane - Klasyfikacja

#### 2. **Regresja Logistyczna**
- **Teoria**: Funkcja sigmoid, log-odds, regularyzacja (parametr C)
- **Zastosowanie**: GWAS (Genome-Wide Association Studies), SNP analysis
- **Dataset**: Breast Cancer Wisconsin (569 próbek, 30 cech)
- **Demo**: Wizualizacja 2D granicy decyzyjnej z prawdopodobieństwami
- **Interaktywność**: Suwak C, wybór 2 cech → granica decyzyjna

#### 3. **k-Najbliższych Sąsiadów (k-NN)**
- **Teoria**: Lazy learning, metryki odległości, kompromis bias-wariancja
- **Zastosowanie**: Klasyfikacja ekspresji genów, podobieństwo próbek
- **Dataset**: Breast Cancer Wisconsin (2 cechy dla wizualizacji 2D)
- **Demo**: Mozaika Voronoi, wizualizacja k sąsiadów
- **Interaktywność**: Suwak k (1-51) → obserwacja przeuczenia vs niedouczenia

#### 4. **Maszyny Wektorów Nośnych (SVM)**
- **Teoria**: Kernel trick, hiperparametry C i gamma, maksymalizacja marginesu
- **Zastosowanie**: Klasyfikacja białek, przewidywanie funkcji i struktury
- **Dataset**: Syntetyczne dane (Moons/Circles) - nieliniowo separowalne
- **Demo**: Porównanie kerneli (linear/rbf/poly), support vectors
- **Interaktywność**: Selectbox kernel, suwaki C i gamma → nieliniowe granice

#### 5. **Drzewa Decyzyjne**
- **Teoria**: Gini vs Entropy, przeuczenie, pruning, white-box model
- **Zastosowanie**: Selekcja genów-biomarkerów, interpretowalne reguły
- **Dataset**: Breast Cancer Wisconsin (30 cech)
- **Demo**: Wizualizacja struktury drzewa (matplotlib), feature importance
- **Interaktywność**: Suwak max_depth → obserwacja ekspozji złożoności

#### 6. **Las Losowy**
- **Teoria**: Bagging, feature randomness, ensemble learning
- **Zastosowanie**: DTI (Drug-Target Interaction), ważność cech
- **Dataset**: Breast Cancer Wisconsin
- **Demo**: Wykres ważności cech (top 20), OOB score
- **Interaktywność**: Suwaki n_estimators i max_depth → stabilność ważności cech

### Uczenie Nienadzorowane

#### 7. **Klastrowanie K-Means**
- **Teoria**: Algorytm Lloyda, Elbow Method, Silhouette Score
- **Zastosowanie**: Klastrowanie ekspresji genów, odkrywanie podtypów nowotworów
- **Dataset**: Breast Cancer Wisconsin + PCA (2D)
- **Demo**: Wizualizacja klastrów w PCA, elbow plot, silhouette
- **Interaktywność**: Suwak k → optymalizacja liczby klastrów

#### 8. **Analiza Głównych Składowych (PCA)**
- **Teoria**: Redukcja wymiaru, wariancja wyjaśniona, eigenvectors
- **Zastosowanie**: Wizualizacja danych RNA-Seq, eksploracja wysokowymiarowa
- **Dataset**: Gene Expression Cancer RNA-Seq (801 próbek, 5 typów nowotworów)
- **Demo**: Scatter plot (PC1 vs PC2) kolorowany typem raka, scree plot
- **Interaktywność**: Wybór osi PC → separacja typów nowotworów

### Narzędzie Uniwersalne

#### 9. **Przeanalizuj Własne Dane (BYOD)**
- **5-stopniowy workflow**: Upload → Definicja zmiennych → Preprocessing → Model → Wyniki
- **Wspierane zadania**: Klasyfikacja, Regresja, Klastrowanie, PCA
- **Wspierane algorytmy**: Wszystkie powyższe (1-8)
- **Dynamiczny UI**: Hiperparametry dostosowane do wybranego algorytmu
- **Preprocessing**: Imputacja (mean/median/most_frequent), StandardScaler
- **Wyniki**: Metryki, confusion matrix, visualizations, CSV export

## 📊 Zbiory Danych

### Wbudowane Datasety

| Dataset | Źródło | Próbki | Cechy | Zadanie | Algorytmy |
|---------|--------|---------|-------|---------|-----------|
| **QSAR Fish Toxicity** | UCI ML Repository | 908 | 6 | Regresja | Linear Regression |
| **Breast Cancer Wisconsin** | scikit-learn | 569 | 30 | Klasyfikacja | Logistic Reg, k-NN, SVM, Trees, RF, K-Means |
| **Gene Expression Cancer** | Syntetyczne/Real | 801 | 100+ | Multi-class | PCA |

### Własne Dane (BYOD)

Platforma akceptuje pliki CSV z:
- Automatyczną detekcją separatora (`,` lub `;`)
- Kodowaniem UTF-8 lub ISO-8859-1
- Obsługą brakujących wartości
- Danymi numerycznymi i kategorycznymi (z ostrzeżeniami)

## 📁 Struktura Projektu

```
machine-learning-toolkit/
│
├── streamlit_app.py              # Strona główna
│
├── pages/                        # Strony algorytmów (9 plików)
│   ├── 1_Regresja_Liniowa.py
│   ├── 2_Regresja_Logistyczna.py
│   ├── 3_kNajblizszych_Sasiadow_kNN.py
│   ├── 4_Maszyny_Wektorow_Nosnych_SVM.py
│   ├── 5_Drzewa_Decyzyjne.py
│   ├── 6_Las_Losowy.py
│   ├── 7_Klastrowanie_K-Means.py
│   ├── 8_Analiza_Glownych_Skladowych_PCA.py
│   └── 9_Analizuj_Wlasne_Dane.py
│
├── src/                          # Moduły pomocnicze
│   ├── __init__.py
│   ├── data_loaders.py          # Funkcje ładowania danych (@st.cache_data)
│   ├── plots.py                 # Funkcje wizualizacji Plotly
│   └── ml_models.py             # Wrappery modeli, pipelines, metryki
│
├── data/                         # Zbiory danych
│   ├── qsar_fish_toxicity.csv
│   └── gene_expression_cancer_rna_seq.csv
│
├── requirements.txt              # Zależności
├── README_STREAMLIT.md          # Ten plik
└── LICENSE
```

## 🛠️ Technologie

### Frontend & Framework
- **Streamlit 1.28+** - Framework aplikacji webowej
  - Multi-Page App (MPA) z automatyczną nawigacją
  - Session state dla zarządzania stanem
  - Caching (`@st.cache_data`) dla wydajności
  - Responsive layout (kolumny, expandery, tabs)

### Wizualizacje
- **Plotly 5.17+** - Interaktywne wykresy (scatter, contour, bar, heatmap)
- **Matplotlib 3.7+** - Wizualizacja drzew decyzyjnych (`plot_tree`)

### Machine Learning
- **scikit-learn 1.3+** - Wszystkie algorytmy ML
  - Regresja: `LinearRegression`
  - Klasyfikacja: `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`
  - Klastrowanie: `KMeans`
  - Redukcja wymiaru: `PCA`
  - Preprocessing: `StandardScaler`, `SimpleImputer`
  - Metryki: `accuracy_score`, `f1_score`, `r2_score`, `silhouette_score`

### Data Processing
- **Pandas 2.0+** - Manipulacja danymi, DataFrames
- **NumPy 1.24+** - Operacje numeryczne, tablice

## 📖 Użytkowanie

### Dla Studentów i Uczących Się

1. **Rozpocznij od strony głównej** - Przeczytaj wprowadzenie
2. **Wybierz algorytm** z paska bocznego (sortowane wg złożoności)
3. **Przeczytaj teorię** w zakładce "Teoria i Zastosowania"
   - Matematyka i intuicja
   - Zastosowania w bioinformatyce
   - Wady i zalety
4. **Eksperymentuj z demo** w zakładce "Interaktywna Demonstracja"
   - Zmieniaj parametry suwakami
   - Obserwuj wpływ na wizualizacje i metryki
   - Czytaj porady edukacyjne
5. **Testuj na własnych danych** w narzędziu BYOD (strona 9)

### Dla Nauczycieli

- **Prezentacje na żywo** - Uruchom aplikację podczas wykładu
- **Zadania domowe** - Poproś studentów o eksperymenty z parametrami
- **Projekty** - Użyj narzędzia BYOD do analizy rzeczywistych danych
- **Customizacja** - Łatwo dodać nowe algorytmy lub datasety

### Dla Badaczy

- **Prototypowanie** - Szybkie testowanie algorytmów na danych pilotażowych
- **Eksploracja** - Wizualizacja wysokowymiarowych danych (PCA)
- **Edukacja zespołu** - Wprowadzenie współpracowników do ML
- **Analiza danych** - Narzędzie BYOD dla podstawowych analiz

## 🔧 Rozwój

### Dodawanie Nowego Algorytmu

1. **Utwórz nowy plik** `pages/X_Nazwa_Algorytmu.py`
2. **Użyj szablonu** z istniejących stron (1-8):
   ```python
   import streamlit as st
   import sys
   from pathlib import Path

   sys.path.append(str(Path(__file__).parent.parent))
   from src.data_loaders import ...
   from src.plots import ...

   st.set_page_config(page_title="...", page_icon="...", layout="wide")
   st.title("...")

   tab_teoria, tab_demo = st.tabs(["📚 Teoria", "🎮 Demo"])

   with tab_teoria:
       # Teoria

   with tab_demo:
       # Demo
   ```
3. **Dodaj do `src/ml_models.py`** jeśli potrzebne nowe wrappery
4. **Streamlit auto-detektuje** nowy plik w `pages/`

### Dodawanie Nowego Datasetu

1. **Umieść CSV** w `data/new_dataset.csv`
2. **Dodaj loader** do `src/data_loaders.py`:
   ```python
   @st.cache_data
   def load_new_dataset():
       df = pd.read_csv(Path(__file__).parent.parent / "data" / "new_dataset.csv")
       return df
   ```
3. **Użyj w stronie algorytmu**

### Best Practices

- ✅ Używaj `@st.cache_data` dla funkcji ładujących dane
- ✅ Separuj logikę (src/) od UI (pages/)
- ✅ Dodawaj try-except dla obsługi błędów
- ✅ Używaj `st.expander()` dla dodatkowych informacji
- ✅ Dodawaj porady edukacyjne (`st.info()`, `st.warning()`)
- ✅ Testuj na różnych rozmiarach ekranu (responsywność)

## 🤝 Wkład

Projekt jest otwarty na kontryb ucje! Mile widziane:

- 🐛 Zgłaszanie bugów
- 💡 Propozycje nowych funkcjonalności
- 📝 Poprawki dokumentacji
- 🧠 Dodawanie nowych algorytmów
- 📊 Dodawanie nowych datasetów bioinformatycznych

## 📄 Licencja

Ten projekt jest licencjonowany na zasadach licencji MIT - szczegóły w pliku [LICENSE](LICENSE).

## 🙏 Podziękowania

- **scikit-learn** - za doskonałe implementacje algorytmów ML
- **Streamlit** - za intuicyjny framework do tworzenia aplikacji ML
- **Plotly** - za piękne, interaktywne wizualizacje
- **Społeczność bioinformatyczna** - za inspirację i przykłady zastosowań

---

**Zbudowane z ❤️ dla edukacji w Machine Learning i Bioinformatyce**

🚀 **Rozpocznij naukę już teraz:** `streamlit run streamlit_app.py`
