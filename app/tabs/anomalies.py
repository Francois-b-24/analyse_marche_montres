import pandas as pd
import plotly.express as px
import streamlit as st
from app.utils import colonnes_numeriques_disponibles

def render_tab_anomalies(echantillon: pd.DataFrame) -> None:
    st.subheader("Détection d'anomalies sur les prix")
    st.caption("Isolation Forest repère les annonces qui s'écartent nettement du reste.")
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception:
        st.info("`scikit-learn` n'est pas installé. Installe-le avec: `pip install scikit-learn`.")
        return

    vars_cand = ['prix', 'diametre', 'annee_prod', 'reserve_de_marche']
    variables = colonnes_numeriques_disponibles(echantillon, vars_cand)
    if 'prix' not in variables:
        st.warning("Colonne 'prix' requise pour la détection d'anomalies.")
        return

    X = echantillon[variables].dropna().copy()
    if len(X) < 50:
        st.info("Échantillon trop petit pour un modèle robuste (min ~50 lignes après NA).")
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    contamination = st.slider("Proportion d'anomalies attendue", 0.01, 0.10, 0.04, 0.01)
    rng = st.number_input("Random state", value=42, step=1, key="rng_nom")
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=rng)
    model.fit(Xs)
    scores = model.decision_function(Xs)  # plus grand = plus normal
    labels = model.predict(Xs)            # -1 = anomalie

    resultats = echantillon.loc[X.index, ['marque','modele','pays'] + variables].copy()
    resultats['anomaly_score'] = -scores
    resultats['is_outlier'] = (labels == -1)

    q1 = resultats['prix'].quantile(0.25)
    resultats['bonne_affaire'] = resultats['is_outlier'] & (resultats['prix'] <= q1)

    col1, col2 = st.columns([2,1])
    with col1:
        deals = resultats[resultats['bonne_affaire']].sort_values('anomaly_score', ascending=False).head(30)
        if deals.empty:
            st.info("Aucune bonne affaire détectée selon les paramètres actuels.")
        else:
            st.dataframe(deals, use_container_width=True)
    with col2:
        fig = px.histogram(resultats, x='anomaly_score', nbins=40, title="Distribution des scores d'anomalie")
        st.plotly_chart(fig, use_container_width=True)