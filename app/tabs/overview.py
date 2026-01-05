import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.utils import top_n_comptes, fmt_eur

def render_header(base: pd.DataFrame) -> None:
    st.title("⌚ Analyse interactive des montres (Chrono24)")
    st.caption("Plongée dans le marché horloger – *Données récupérées par scraping sur le site Chrono24 du 16/04/2024 au 23/08/2024.*")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Observations totales", f"{len(base):,}".replace(",", " "))
    with col2:
        if 'marque' in base.columns:
            nb_marques = base['marque'].nunique()
            st.metric("🏢 Marques différentes", f"{nb_marques:,}".replace(",", " "))
    with col3:
        if 'pays' in base.columns:
            nb_pays = base['pays'].nunique()
            st.metric("🌍 Pays vendeurs", f"{nb_pays}")
    with col4:
        if 'prix' in base.columns and base['prix'].notna().any():
            prix_total = base['prix'].sum()
            st.metric("💰 Valeur totale", f"{prix_total/1e6:.1f}M €")

def render_tab_overview(echantillon: pd.DataFrame) -> None:
    st.subheader("📋 Aperçu des données")

    # Affichage des premières lignes (sans les colonnes catégorielles redondantes)
    cols_to_show = [c for c in echantillon.columns if not c.endswith('_cat') and c != 'Unnamed: 0']
    st.dataframe(echantillon[cols_to_show].head(10), use_container_width=True)

    st.markdown("---")

    # KPIs principaux
    st.subheader("📈 Indicateurs clés")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if 'prix' in echantillon and echantillon['prix'].notna().any():
            med = np.nanmedian(echantillon['prix'])
            st.metric("💶 Prix médian", fmt_eur(med))
            st.caption(f"*50% des montres ≤ {fmt_eur(med)}*")

    with c2:
        if 'prix' in echantillon and echantillon['prix'].notna().any():
            moy = np.nanmean(echantillon['prix'])
            st.metric("📊 Prix moyen", fmt_eur(moy))

    with c3:
        if 'marque' in echantillon and not echantillon['marque'].dropna().empty:
            top_marque = echantillon['marque'].mode().iloc[0]
            count_marque = (echantillon['marque'] == top_marque).sum()
            st.metric("🏆 Marque #1", top_marque)
            st.caption(f"*{count_marque:,} offres*".replace(",", " "))

    with c4:
        if 'diametre' in echantillon and echantillon['diametre'].notna().any():
            diam_moy = np.nanmean(echantillon['diametre'])
            st.metric("⭕ Diamètre moyen", f"{diam_moy:.1f} mm")

    st.markdown("---")

    # Distribution des prix
    st.subheader("💰 Distribution des prix")

    if 'prix' in echantillon and echantillon['prix'].notna().sum() > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Histogramme interactif
            prix_data = echantillon['prix'].dropna()

            # Option de filtrage
            filter_outliers = st.checkbox("Filtrer les valeurs extrêmes (99e percentile)", value=True)
            if filter_outliers:
                p99 = prix_data.quantile(0.99)
                prix_data = prix_data[prix_data <= p99]
                st.caption(f"*Affichage jusqu'à {fmt_eur(p99)} (99e percentile)*")

            fig = px.histogram(prix_data, nbins=50,
                             title="Répartition des prix",
                             labels={'value': 'Prix (€)', 'count': 'Nombre de montres'})
            fig.update_traces(marker_color='#1f77b4', opacity=0.7)
            fig.update_layout(
                showlegend=False,
                xaxis_title="Prix (€)",
                yaxis_title="Nombre de montres",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Statistiques des prix**")
            stats = echantillon['prix'].describe()
            st.metric("Min", fmt_eur(stats['min']))
            st.metric("Q1 (25%)", fmt_eur(stats['25%']))
            st.metric("Médiane", fmt_eur(stats['50%']))
            st.metric("Q3 (75%)", fmt_eur(stats['75%']))
            st.metric("Max", fmt_eur(stats['max']))
    else:
        st.info("Pas de prix renseignés pour l'échantillon filtré.")

    st.markdown("---")

    # Boxplot des marques
    st.subheader("📦 Prix par marque (Top 10)")

    if all(c in echantillon.columns for c in ['prix', 'marque']) and echantillon['prix'].notna().sum() > 0:
        marques_top = echantillon['marque'].value_counts().head(10).index
        tmp = echantillon[echantillon['marque'].isin(marques_top) & echantillon['prix'].notna()].copy()

        # Filtrer les outliers pour une meilleure visualisation
        p99 = tmp['prix'].quantile(0.99)
        tmp_filtered = tmp[tmp['prix'] <= p99]

        fig2 = px.box(tmp_filtered, x='marque', y='prix',
                     title=f"Distribution des prix par marque (jusqu'au 99e percentile : {fmt_eur(p99)})",
                     color='marque')
        fig2.update_layout(
            xaxis_title='',
            yaxis_title='Prix (€)',
            showlegend=False,
            height=500,
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("💡 *La boîte à moustaches montre la médiane (ligne centrale), les quartiles (boîte) et les valeurs extrêmes (points).*")

    st.markdown("---")

    # Géographie
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Top 10 pays vendeurs")
        if 'pays' in echantillon and not echantillon['pays'].empty:
            comptes_pays = top_n_comptes(echantillon['pays'], n=10)
            if len(comptes_pays) > 0:
                fig3 = px.bar(comptes_pays, x='valeur', y='nb',
                            title="Nombre d'offres par pays",
                            color='nb',
                            color_continuous_scale='Blues')
                fig3.update_layout(
                    xaxis_title='',
                    yaxis_title="Nombre d'offres",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("⚙️ Types de mouvement")
        if 'mouvement' in echantillon and not echantillon['mouvement'].empty:
            mouv_counts = echantillon['mouvement'].value_counts().head(10)
            fig4 = px.pie(
                values=mouv_counts.values,
                names=mouv_counts.index,
                title="Répartition des mouvements",
                hole=0.4
            )
            fig4.update_traces(textposition='inside', textinfo='percent+label')
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # État et sexe
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 État des montres")
        if 'etat' in echantillon:
            etat_counts = echantillon['etat'].value_counts()
            fig5 = px.bar(etat_counts,
                         title="Distribution par état",
                         color=etat_counts.values,
                         color_continuous_scale='Greens')
            fig5.update_layout(
                xaxis_title='',
                yaxis_title="Nombre de montres",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("👤 Segmentation par sexe")
        if 'sexe' in echantillon:
            sexe_counts = echantillon['sexe'].value_counts()
            fig6 = px.pie(
                values=sexe_counts.values,
                names=sexe_counts.index,
                title="Répartition Homme/Femme",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            fig6.update_traces(textposition='inside', textinfo='percent+label')
            fig6.update_layout(height=350)
            st.plotly_chart(fig6, use_container_width=True)
