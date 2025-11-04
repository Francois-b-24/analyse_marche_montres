import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from app.utils import top_n_comptes, fmt_eur

def render_header(base: pd.DataFrame) -> None:
    st.title("⌚Analyse interactive des montres (Chrono24)")
    st.caption("Plongée dans le marché horloger – *Données récupérées par scraping sur le site Chrono24 du 16/04/2024 au 23/08/2025.*")
    st.write(f"Observations totales : **{len(base):,}**".replace(",", " "))

def render_tab_overview(echantillon: pd.DataFrame) -> None:
    echantillon_viz = echantillon.iloc[:, :-6] if echantillon.shape[1] > 6 else echantillon
    st.caption("*Aperçu général des données*")
    st.dataframe(echantillon_viz.head(5))

    c1, c2, c3 = st.columns(3)
    with c2:
        if 'prix' in echantillon and echantillon['prix'].notna().any():
            med = np.nanmedian(echantillon['prix'])
            st.metric("Prix médian", fmt_eur(med))
            st.caption(f"*Cela signifie que 50% des données ont un prix inférieur à {fmt_eur(med)}*")
        else:
            st.metric("Prix médian (€)", "—")
    with c1:
        if 'marque' in echantillon and not echantillon['marque'].dropna().empty:
            st.metric("La marque la plus représentée est : ", echantillon['marque'].mode().iloc[0])
        else:
            st.metric("Marque la plus représentée", "—")
    with c3:
        if 'prix' in echantillon and echantillon['prix'].notna().any():
            st.metric("Prix moyen", fmt_eur(np.nanmean(echantillon['prix'])))
        else:
            st.metric("Prix moyen (€)", "—")

    st.subheader("Distribution des prix (€)")
    if 'prix' in echantillon and echantillon['prix'].notna().sum() > 0:
        fig = px.histogram(echantillon, x='prix', nbins=50)
        fig.update_yaxes(title="Nombre d'offres")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de prix renseignés pour l’échantillon filtré.")

    st.subheader("Boxplot des 10 marques les plus représentées")
    if all(c in echantillon.columns for c in ['prix', 'marque']) and echantillon['prix'].notna().sum() > 0 and echantillon['marque'].notna().sum() > 0:
        marques_top = echantillon['marque'].value_counts().head(10).index
        tmp = echantillon[echantillon['marque'].isin(marques_top)]
        fig2 = px.box(tmp, x='marque', y='prix', points='outliers')
        fig2.update_layout(xaxis_title='', yaxis_title='Prix (€)')
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("💡*La boîte à moustaches montre médiane, dispersion et valeurs extrêmes.*")

    st.subheader("Top 10 pays vendeurs")
    if 'pays' in echantillon and not echantillon['pays'].empty:
        comptes_pays = top_n_comptes(echantillon['pays'], n=10)
        if len(comptes_pays) > 0:
            fig3 = px.bar(comptes_pays, x='valeur', y='nb')
            fig3.update_layout(xaxis_title='', yaxis_title="Nombre d'offres")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Aucune donnée de pays à afficher.")
    else:
        st.info("Colonne 'pays' absente.")