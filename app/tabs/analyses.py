import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.utils import quantile_labels, fmt_eur

def render_tab_analyses(echantillon: pd.DataFrame) -> None:
    st.subheader("📊 Analyses descriptives approfondies")
    st.caption("Explorez les relations entre les différentes caractéristiques des montres.")

    # Onglets secondaires pour organiser les analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Mouvement",
        "🔩 Matériaux",
        "📏 Dimensions & Caractéristiques",
        "🔄 Analyses croisées"
    ])

    # ===== TAB 1: MOUVEMENT =====
    with tab1:
        st.markdown("### Analyse des mouvements")

        col1, col2 = st.columns(2)

        with col1:
            if 'mouvement' in echantillon.columns and not echantillon['mouvement'].dropna().empty:
                comptes = (
                    echantillon['mouvement'].fillna('(NA)').value_counts().head(10)
                    .rename_axis('mouvement').reset_index(name='nb')
                )
                fig = px.bar(comptes, x='mouvement', y='nb', text='nb',
                           title="Répartition des types de mouvement",
                           color='nb', color_continuous_scale='Viridis')
                fig.update_traces(textposition='outside', cliponaxis=False)
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Nombre',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if {'prix', 'mouvement'}.issubset(echantillon.columns) and echantillon['prix'].notna().any():
                tmp = echantillon.loc[echantillon['prix'].notna(), ['mouvement', 'prix']].copy()
                tmp['mouvement'] = tmp['mouvement'].fillna('(NA)')

                prix_moy = tmp.groupby('mouvement')['prix'].agg(['mean', 'median', 'count']).reset_index()
                prix_moy = prix_moy[prix_moy['count'] >= 10].sort_values('median', ascending=False)

                fig = px.bar(prix_moy, x='mouvement', y='median',
                           title="Prix médian par type de mouvement",
                           text='median', color='median',
                           color_continuous_scale='Blues')
                fig.update_traces(texttemplate='%{text:,.0f} €', textposition='outside')
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Prix médian (€)',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        if {'prix', 'mouvement'}.issubset(echantillon.columns) and echantillon['prix'].notna().any():
            st.markdown("#### 🔢 Mouvement × Tranche de prix")

            col1, col2 = st.columns([3, 1])
            with col2:
                afficher_pct = st.checkbox("Afficher en %", value=True, key="ad_mv_pct")

            with col1:
                tmp = echantillon.loc[echantillon['prix'].notna(), ['mouvement', 'prix']].copy()
                if not tmp.empty and len(tmp) > 10:
                    bins = pd.qcut(tmp['prix'], q=4, duplicates='drop')
                    labels = quantile_labels(bins.cat)
                    tmp['tranche_prix'] = bins.cat.rename_categories(labels)
                    tmp['mouvement'] = tmp['mouvement'].fillna('(NA)')
                    tab = pd.crosstab(tmp['mouvement'], tmp['tranche_prix'],
                                    normalize='index' if afficher_pct else False)
                    if afficher_pct:
                        tab = (tab * 100).round(1)
                    tab = tab.sort_index(axis=1)
                    st.dataframe(tab, use_container_width=True)
                else:
                    st.info("Pas assez de données pour construire les tranches.")

    # ===== TAB 2: MATÉRIAUX =====
    with tab2:
        st.markdown("### Analyse des matériaux")

        st.markdown("#### 🔲 Matière du boîtier")
        col1, col2 = st.columns(2)

        with col1:
            if 'matiere_boitier' in echantillon.columns and not echantillon['matiere_boitier'].dropna().empty:
                counts = (
                    echantillon['matiere_boitier'].fillna('(NA)').value_counts().head(10)
                    .rename_axis('matiere_boitier').reset_index(name='nb')
                )
                fig = px.bar(counts, x='matiere_boitier', y='nb', text='nb',
                           title="Répartition par matière de boîtier",
                           color='nb', color_continuous_scale='Oranges')
                fig.update_traces(textposition='outside', cliponaxis=False)
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Nombre',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if {'prix', 'matiere_boitier'}.issubset(echantillon.columns) and echantillon['prix'].notna().any():
                tmp = echantillon.loc[echantillon['prix'].notna(), ['matiere_boitier', 'prix']].copy()
                tmp['matiere_boitier'] = tmp['matiere_boitier'].fillna('(NA)')

                prix_mat = tmp.groupby('matiere_boitier')['prix'].agg(['median', 'count']).reset_index()
                prix_mat = prix_mat[prix_mat['count'] >= 10].sort_values('median', ascending=False).head(10)

                fig = px.bar(prix_mat, x='matiere_boitier', y='median',
                           title="Prix médian par matière",
                           text='median', color='median',
                           color_continuous_scale='Reds')
                fig.update_traces(texttemplate='%{text:,.0f} €', textposition='outside')
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Prix médian (€)',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ⌚ Matière du bracelet")
        if 'matiere_bracelet' in echantillon.columns:
            bracelet_counts = echantillon['matiere_bracelet'].value_counts().head(10)
            fig = px.pie(
                values=bracelet_counts.values,
                names=bracelet_counts.index,
                title="Distribution des matières de bracelet",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 3: DIMENSIONS & CARACTÉRISTIQUES =====
    with tab3:
        st.markdown("### Dimensions et caractéristiques techniques")

        col1, col2 = st.columns(2)

        with col1:
            if 'diametre' in echantillon.columns and echantillon['diametre'].notna().sum() > 0:
                st.markdown("#### 📏 Distribution du diamètre")
                diam_data = echantillon['diametre'].dropna()

                q1, q99 = diam_data.quantile([0.01, 0.99])
                diam_filtered = diam_data[(diam_data >= q1) & (diam_data <= q99)]

                fig = px.histogram(diam_filtered, nbins=30,
                                 title=f"Diamètre (mm) - Distribution",
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=diam_filtered.median(), line_dash="dash",
                            line_color="red",
                            annotation_text=f"Médiane: {diam_filtered.median():.1f}mm")
                fig.update_layout(
                    xaxis_title='Diamètre (mm)',
                    yaxis_title='Nombre de montres',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'etencheite' in echantillon.columns and echantillon['etencheite'].notna().sum() > 0:
                st.markdown("#### 💧 Étanchéité")
                etanch_data = echantillon['etencheite'].dropna()

                etanch_counts = etanch_data.value_counts().sort_index().head(10)

                fig = px.bar(etanch_counts,
                           title="Distribution de l'étanchéité (ATM)",
                           color=etanch_counts.values,
                           color_continuous_scale='Blues')
                fig.update_layout(
                    xaxis_title='Étanchéité (ATM)',
                    yaxis_title='Nombre de montres',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        if 'reserve_de_marche' in echantillon.columns and echantillon['reserve_de_marche'].notna().sum() > 0:
            st.markdown("#### ⏱️ Réserve de marche")

            reserve_data = echantillon['reserve_de_marche'].dropna()
            q99 = reserve_data.quantile(0.99)
            reserve_filtered = reserve_data[reserve_data <= q99]

            fig = px.box(reserve_filtered,
                       title=f"Distribution de la réserve de marche (heures) - jusqu'au 99e percentile",
                       color_discrete_sequence=['#e74c3c'])
            fig.update_layout(
                yaxis_title='Heures',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Médiane", f"{reserve_filtered.median():.0f}h")
            with col2:
                st.metric("Moyenne", f"{reserve_filtered.mean():.0f}h")
            with col3:
                st.metric("Max (99%)", f"{q99:.0f}h")

    # ===== TAB 4: ANALYSES CROISÉES =====
    with tab4:
        st.markdown("### 🔄 Analyses croisées")

        if {'prix', 'diametre'}.issubset(echantillon.columns):
            st.markdown("#### 💰 Prix × Diamètre")

            tmp = echantillon[['prix', 'diametre', 'marque']].dropna()

            p99_prix = tmp['prix'].quantile(0.99)
            q1_diam, q99_diam = tmp['diametre'].quantile([0.01, 0.99])
            tmp_filtered = tmp[
                (tmp['prix'] <= p99_prix) &
                (tmp['diametre'] >= q1_diam) &
                (tmp['diametre'] <= q99_diam)
            ]

            if len(tmp_filtered) > 5000:
                tmp_filtered = tmp_filtered.sample(5000, random_state=42)

            fig = px.scatter(tmp_filtered, x='diametre', y='prix',
                           hover_data=['marque'],
                           title="Relation Prix-Diamètre",
                           opacity=0.5,
                           color='prix',
                           color_continuous_scale='Viridis')

            fig.update_layout(
                xaxis_title='Diamètre (mm)',
                yaxis_title='Prix (€)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        if {'prix', 'etat', 'sexe'}.issubset(echantillon.columns):
            st.markdown("#### 🔧 Prix × État × Sexe")

            tmp = echantillon[['prix', 'etat', 'sexe']].dropna()
            tmp = tmp[tmp['prix'] <= tmp['prix'].quantile(0.99)]

            fig = px.violin(tmp, x='etat', y='prix', color='sexe',
                          title="Distribution des prix par état et sexe",
                          box=True, points=False)
            fig.update_layout(
                xaxis_title='État',
                yaxis_title='Prix (€)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
