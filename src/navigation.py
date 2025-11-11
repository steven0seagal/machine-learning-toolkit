"""
Shared navigation component for the ML platform
"""
import streamlit as st


def render_sidebar_navigation():
    """Render the left sidebar navigation panel with subject categories"""

    st.sidebar.title("🧬 Nawigacja")
    st.sidebar.markdown("---")

    # Home section
    st.sidebar.page_link("streamlit_app.py", label="🏠 Strona Główna", icon="🏠")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Algorytmy")

    # Supervised Learning - Regression
    with st.sidebar.expander("📈 Uczenie Nadzorowane - Regresja", expanded=False):
        st.page_link("pages/1_Regresja_Liniowa.py", label="Regresja Liniowa", icon="📊")

    # Supervised Learning - Classification
    with st.sidebar.expander("🎯 Uczenie Nadzorowane - Klasyfikacja", expanded=False):
        st.page_link("pages/2_Regresja_Logistyczna.py", label="Regresja Logistyczna", icon="📉")
        st.page_link("pages/3_kNajblizszych_Sasiadow_kNN.py", label="k-NN", icon="👥")
        st.page_link("pages/4_Maszyny_Wektorow_Nosnych_SVM.py", label="SVM", icon="⚡")
        st.page_link("pages/5_Drzewa_Decyzyjne.py", label="Drzewa Decyzyjne", icon="🌳")
        st.page_link("pages/6_Las_Losowy.py", label="Las Losowy", icon="🌲")

    # Unsupervised Learning
    with st.sidebar.expander("🔍 Uczenie Nienadzorowane", expanded=False):
        st.page_link("pages/7_Klastrowanie_K-Means.py", label="K-Means", icon="🎨")
        st.page_link("pages/8_Analiza_Glownych_Skladowych_PCA.py", label="PCA", icon="📐")

    st.sidebar.markdown("---")

    # Tools section
    st.sidebar.subheader("🛠️ Narzędzia")
    st.page_link("pages/9_Analizuj_Wlasne_Dane.py", label="Analizuj Własne Dane", icon="🔬")

    st.sidebar.markdown("---")

    # Quick info
    with st.sidebar.expander("ℹ️ Informacje", expanded=False):
        st.markdown("""
        **Platforma ML w Bioinformatyce**

        Wybierz algorytm z menu powyżej, aby:
        - Poznać teorię
        - Zobaczyć demo
        - Eksperymentować z parametrami
        """)
