"""
Platforma Edukacyjna Uczenia Maszynowego w Bioinformatyce
Educational Machine Learning Platform for Bioinformatics

Main landing page of the application.
"""

import streamlit as st
from src.navigation import render_sidebar_navigation

# Page configuration
st.set_page_config(
    page_title="ML w Bioinformatyce",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render sidebar navigation
render_sidebar_navigation()

# Main title
st.title("🧬 Platforma Edukacyjna Uczenia Maszynowego w Bioinformatyce")

# Introduction
st.markdown("""
## Witaj!

Ta platforma została stworzona w celu edukacji w zakresie algorytmów uczenia maszynowego
w kontekście zastosowań bioinformatycznych. Platforma składa się z trzech głównych filarów:

### 📚 1. Baza Wiedzy
Każdy algorytm zawiera szczegółowe wyjaśnienie:
- Teoretyczne podstawy działania
- Kluczowe założenia i hiperparametry
- Metryki ewaluacji
- Konkretne zastosowania w bioinformatyce
- Wady i zalety

### 🎮 2. Interaktywne Demonstracje
Każdy algorytm ma interaktywną demonstrację pozwalającą na:
- Eksperymentowanie z hiperparametrami w czasie rzeczywistym
- Wizualizację wyników za pomocą wykresów Plotly
- Zrozumienie wpływu parametrów na wyniki modelu
- Pracę na rzeczywistych zbiorach danych bioinformatycznych

### 🔬 3. Narzędzie "Przeanalizuj Własne Dane"
Uniwersalne narzędzie pozwalające na:
- Wgranie własnego zbioru danych (CSV)
- Wybór zmiennej docelowej i cech
- Przetwarzanie wstępne (imputacja, skalowanie)
- Wybór i trenowanie modelu
- Pobieranie wyników analizy

## 📖 Algorytmy

Platforma obejmuje następujące algorytmy uczenia maszynowego:

### Uczenie Nadzorowane - Regresja
1. **Regresja Liniowa** - Modelowanie zależności liniowych (QSAR)

### Uczenie Nadzorowane - Klasyfikacja
2. **Regresja Logistyczna** - Klasyfikacja binarna (GWAS, SNP)
3. **k-Najbliższych Sąsiadów (k-NN)** - Klasyfikacja oparta na podobieństwie
4. **Maszyny Wektorów Nośnych (SVM)** - Klasyfikacja z kernelami (białka)
5. **Drzewa Decyzyjne** - Modele interpretowalne (selekcja genów)
6. **Las Losowy** - Ensemble learning (ważność cech, DTI)

### Uczenie Nienadzorowane
7. **Klastrowanie K-Means** - Grupowanie próbek (ekspresja genów)
8. **Analiza Głównych Składowych (PCA)** - Redukcja wymiaru (wizualizacja RNA-Seq)

## 🚀 Jak Zacząć?

1. **Wybierz algorytm** z paska bocznego nawigacji
2. **Przeczytaj teorię** w zakładce "Teoria i Zastosowania"
3. **Eksperymentuj** z interaktywną demonstracją w zakładce "Demo"
4. **Przeanalizuj własne dane** używając narzędzia BYOD

## 💡 Wskazówki

- Wszystkie wizualizacje są interaktywne (możesz je powiększać, przesuwać)
- Zmieniaj parametry suwakami, aby zobaczyć ich wpływ w czasie rzeczywistym
- Zwróć uwagę na metryki ewaluacji przy różnych ustawieniach
- Porównuj wyniki różnych algorytmów na tych samych danych

## 📊 Zbiory Danych

Platforma wykorzystuje rzeczywiste zbiory danych bioinformatycznych:
- **QSAR Fish Toxicity** - deskryptory molekularne i toksyczność
- **Breast Cancer Wisconsin** - cechy komórek nowotworowych
- **Gene Expression Cancer RNA-Seq** - dane ekspresji genów z 5 typów nowotworów

---

**Rozpocznij naukę wybierając algorytm z paska bocznego! 👈**
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Platforma edukacyjna | Machine Learning w Bioinformatyce</p>
    <p><small>Zbudowana z wykorzystaniem Streamlit, scikit-learn i Plotly</small></p>
</div>
""", unsafe_allow_html=True)
