import pandas as pd
import plotly.express as px
import streamlit as st
from app.utils import colonnes_numeriques_disponibles

def render_tab_segmentation(echantillon: pd.DataFrame) -> None:
    st.subheader("Segmentation non supervisée (K-Means)")
    st.caption("Regroupe les annonces aux caractéristiques proches; PCA 2D pour visualiser.")
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except Exception:
        st.info("`scikit-learn` n'est pas installé. Installe-le avec: `pip install scikit-learn`.")
        return

    vars_cand = ['prix', 'diametre', 'annee_prod', 'reserve_de_marche']
    dispo = colonnes_numeriques_disponibles(echantillon, vars_cand)
    if not dispo:
        st.info("Aucune variable numérique disponible pour segmenter.")
        return

    defaults = [v for v in ['prix', 'diametre'] if v in dispo]
    variables = st.multiselect("Variables à utiliser pour la segmentation", options=dispo,
                               default=defaults if len(defaults) >= 2 else dispo[:2])
    if len(variables) < 2:
        st.info("Sélectionne au moins 2 variables.")
        return

    c1, c2 = st.columns(2)
    with c1:
        standardiser = st.checkbox("Standardiser (recommandé)", value=True)
    with c2:
        tronquer = st.checkbox("Limiter les extrêmes (1–99e pct.)", value=True)

    X = echantillon[variables].dropna().copy()
    if tronquer and len(X) > 0:
        for col in variables:
            q1, q99 = X[col].quantile([0.01, 0.99])
            X[col] = X[col].clip(q1, q99)

    if len(X) < 50:
        st.info("Échantillon trop petit pour une segmentation robuste (min ~50 lignes).")
        return

    if standardiser:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.values

    _, colk = st.columns([2,1])
    with colk:
        k = st.slider("Nombre de clusters (k)", 2, 10, 5, 1)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    try:
        sil = silhouette_score(Xs, labels)
        st.caption(f"Indice de silhouette pour k={k} : **{sil:.2f}**")
    except Exception:
        pass

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    vis = pd.DataFrame(coords, columns=['PC1','PC2'], index=X.index)
    vis['cluster'] = labels.astype(int)
    for meta in ['marque','modele','pays']:
        if meta in echantillon.columns:
            vis[meta] = echantillon.loc[vis.index, meta]
    for f in variables:
        vis[f] = echantillon.loc[vis.index, f]

    fig = px.scatter(vis, x='PC1', y='PC2', color='cluster',
                     hover_data=[c for c in ['marque','modele','pays'] + variables if c in vis.columns])
    st.plotly_chart(fig, use_container_width=True)

    prof = X.assign(cluster=labels).groupby('cluster')[variables].median()
    taille = X.assign(cluster=labels).groupby('cluster').size().rename('taille')
    st.subheader("Profil des clusters")
    st.dataframe(prof.join(taille).reset_index(), use_container_width=True)