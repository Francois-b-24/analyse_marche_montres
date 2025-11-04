import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def charger_donnees(db_path: str) -> pd.DataFrame:
    """Charge la base de données et nettoie les colonnes parasites."""
    try:
        df = pd.read_excel(db_path)
        df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed:')]
    except Exception as e:
        st.error(f"Impossible d'ouvrir la base de données: {e}")
        st.stop()

    # Retirer éventuelle première colonne id/index sans couper la dernière
    if len(df.columns) > 0 and str(df.columns[0]).lower() in ("id", "index"):
        df = df.iloc[:, 1:-1]
    return df
