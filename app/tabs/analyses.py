import pandas as pd
import plotly.express as px
import streamlit as st
from app.utils import quantile_labels

def render_tab_analyses(echantillon: pd.DataFrame) -> None:
    st.subheader("Analyse descriptive")
    st.caption("📊 Aperçu rapide des répartitions et croisements.")

    st.markdown("### Mouvement")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if 'mouvement' in echantillon.columns and not echantillon['mouvement'].dropna().empty:
            comptes = (
                echantillon['mouvement'].fillna('(NA)').value_counts().head(10)
                .rename_axis('mouvement').reset_index(name='nb')
            )
            fig = px.bar(comptes, x='mouvement', y='nb', text='nb', title="Répartition des types de mouvement")
            fig.update_traces(textposition='outside', cliponaxis=False)
            fig.update_layout(xaxis_title='', yaxis_title='Nombre', margin=dict(t=30, r=10, b=10, l=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de mouvement à afficher.")
    with mcol2:
        if {'prix','mouvement'}.issubset(echantillon.columns) and echantillon['prix'].notna().any():
            afficher_pct = st.checkbox("Afficher en pourcentage (par ligne)", value=True, key="ad_mv_pct")
            tmp = echantillon.loc[echantillon['prix'].notna(), ['mouvement','prix']].copy()
            if tmp.empty:
                st.info("Pas assez de prix non manquants pour construire les tranches.")
            else:
                bins = pd.qcut(tmp['prix'], q=4, duplicates='drop')
                labels = quantile_labels(bins.cat)
                tmp['tranche_prix'] = bins.cat.rename_categories(labels)
                tmp['mouvement'] = tmp['mouvement'].fillna('(NA)')
                tab = pd.crosstab(tmp['mouvement'], tmp['tranche_prix'], normalize='index' if afficher_pct else False)
                if afficher_pct: tab = (tab * 100).round(1)
                tab = tab.sort_index(axis=1)
                st.caption("Mouvement × Tranche de prix")
                st.dataframe(tab, use_container_width=True)
        else:
            st.info("Colonnes 'prix' et/ou 'mouvement' manquantes.")

    st.markdown('---')

    st.markdown("### Matière du boîtier")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if 'matiere_boitier' in echantillon.columns and not echantillon['matiere_boitier'].dropna().empty:
            counts = (
                echantillon['matiere_boitier'].fillna('(NA)').value_counts().head(10)
                .rename_axis('matiere_boitier').reset_index(name='nb')
            )
            fig2 = px.bar(counts, x='matiere_boitier', y='nb', text='nb', title="Répartition des types de boîtiers")
            fig2.update_traces(textposition='outside', cliponaxis=False)
            fig2.update_layout(xaxis_title='', yaxis_title='Nombre', margin=dict(t=30, r=10, b=10, l=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée de matière de boîtier à afficher.")
    with bcol2:
        if {'prix','matiere_boitier'}.issubset(echantillon.columns) and echantillon['prix'].notna().any():
            afficher_pct2 = st.checkbox("Afficher en pourcentage (par ligne)", value=True, key="ad_mb_pct")
            tmp2 = echantillon.loc[echantillon['prix'].notna(), ['matiere_boitier','prix']].copy()
            if tmp2.empty:
                st.info("Pas assez de prix non manquants pour construire les tranches.")
            else:
                bins2 = pd.qcut(tmp2['prix'], q=4, duplicates='drop')
                labels2 = quantile_labels(bins2.cat)
                tmp2['tranche_prix'] = bins2.cat.rename_categories(labels2)
                tmp2['matiere_boitier'] = tmp2['matiere_boitier'].fillna('(NA)')
                tab2 = pd.crosstab(tmp2['matiere_boitier'], tmp2['tranche_prix'], normalize='index' if afficher_pct2 else False)
                if afficher_pct2: tab2 = (tab2 * 100).round(1)
                tab2 = tab2.sort_index(axis=1)
                st.caption("Matière du boîtier × Tranche de prix")
                st.dataframe(tab2, use_container_width=True)
        else:
            st.info("Colonnes 'prix' et/ou 'matiere_boitier' manquantes.")