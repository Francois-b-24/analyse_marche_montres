import streamlit as st
import pandas as pd
from . import config
from .data import charger_donnees
from .tabs import overview, analyses, anomalies, segmentation, regression

def run() -> None:
    # Configuration de la page
    st.set_page_config(page_title=config.TITRE_PAGE, layout=config.LAYOUT)

    # Chargement des données
    with st.spinner("Chargement de la base..."):
        base: pd.DataFrame = charger_donnees(config.CHEMIN_BDD)
        echantillon = base.copy()

    # En-tête
    overview.render_header(base)

    # Onglets
    TAB_ANALYSE, TAB_ANALYSES, TAB_ANOMALIES, TAB_SEGMENTATION, TAB_REGRESSION = st.tabs([
        "Overview", "Visualisation", "Deals", "Clustering", "Régression du prix"
    ])

    with TAB_ANALYSE:
        overview.render_tab_overview(echantillon)
    with TAB_ANALYSES:
        analyses.render_tab_analyses(echantillon)
    with TAB_ANOMALIES:
        anomalies.render_tab_anomalies(echantillon)
    with TAB_SEGMENTATION:
        segmentation.render_tab_segmentation(echantillon)
    with TAB_REGRESSION:
        regression.render_tab_regression(echantillon)