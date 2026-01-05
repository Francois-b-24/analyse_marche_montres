import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.utils import colonnes_numeriques_disponibles, fmt_eur

def render_tab_anomalies(echantillon: pd.DataFrame) -> None:
    st.subheader("🔍 Détection d'anomalies et bonnes affaires")
    st.caption("L'algorithme Isolation Forest repère les montres aux caractéristiques inhabituelles.")

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        st.error("❌ `scikit-learn` n'est pas installé. Installez-le avec: `pip install scikit-learn`")
        return

    # Paramètres du modèle
    with st.expander("⚙️ Configuration du modèle", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            contamination = st.slider(
                "Proportion d'anomalies attendue",
                min_value=0.01,
                max_value=0.15,
                value=0.05,
                step=0.01,
                help="% de montres considérées comme anomalies"
            )

        with col2:
            n_estimators = st.slider(
                "Nombre d'arbres",
                min_value=100,
                max_value=500,
                value=300,
                step=50,
                help="Plus d'arbres = plus précis mais plus lent"
            )

        with col3:
            rng = st.number_input(
                "Random state",
                value=42,
                step=1,
                help="Pour des résultats reproductibles"
            )

    # Variables à utiliser
    vars_cand = ['prix', 'diametre', 'reserve_de_marche', 'etencheite']
    variables = colonnes_numeriques_disponibles(echantillon, vars_cand)

    if 'prix' not in variables:
        st.warning("⚠️ La colonne 'prix' est requise pour la détection d'anomalies.")
        return

    st.info(f"📊 Variables utilisées: {', '.join(variables)}")

    # Préparation des données
    X = echantillon[variables].dropna().copy()

    if len(X) < 100:
        st.warning(f"⚠️ Échantillon trop petit ({len(X)} lignes). Minimum recommandé: 100 lignes.")
        return

    # Standardisation
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Entraînement du modèle
    with st.spinner("🔄 Entraînement du modèle Isolation Forest..."):
        model = IsolationForest(
            n_estimators=int(n_estimators),
            contamination=contamination,
            random_state=int(rng),
            n_jobs=-1
        )
        model.fit(Xs)
        scores = model.decision_function(Xs)
        labels = model.predict(Xs)

    # Résultats
    cols_meta = ['marque', 'modele', 'pays', 'mouvement', 'matiere_boitier', 'etat']
    cols_available = [c for c in cols_meta if c in echantillon.columns]

    resultats = echantillon.loc[X.index, cols_available + variables].copy()
    resultats['anomaly_score'] = -scores
    resultats['is_outlier'] = (labels == -1)

    # Définir les bonnes affaires
    q1 = resultats['prix'].quantile(0.25)
    median_prix = resultats['prix'].median()
    resultats['bonne_affaire'] = resultats['is_outlier'] & (resultats['prix'] <= q1)

    # Métriques
    st.markdown("---")
    st.subheader("📊 Résumé de la détection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nb_anomalies = resultats['is_outlier'].sum()
        pct_anomalies = (nb_anomalies / len(resultats)) * 100
        st.metric("🔴 Anomalies détectées", f"{nb_anomalies}", f"{pct_anomalies:.1f}%")

    with col2:
        nb_deals = resultats['bonne_affaire'].sum()
        st.metric("💎 Bonnes affaires", f"{nb_deals}")

    with col3:
        if nb_deals > 0:
            prix_moy_deals = resultats[resultats['bonne_affaire']]['prix'].mean()
            st.metric("💰 Prix moyen deals", fmt_eur(prix_moy_deals))
        else:
            st.metric("💰 Prix moyen deals", "—")

    with col4:
        st.metric("📏 Échantillon analysé", f"{len(resultats):,}".replace(",", " "))

    st.markdown("---")

    # Visualisations
    tab1, tab2, tab3 = st.tabs(["💎 Bonnes affaires", "📈 Distribution des scores", "🔬 Analyse détaillée"])

    with tab1:
        st.markdown("### 💎 Top 50 bonnes affaires détectées")
        deals = resultats[resultats['bonne_affaire']].sort_values('anomaly_score', ascending=False).head(50)

        if deals.empty:
            st.info("Aucune bonne affaire détectée avec les paramètres actuels. Essayez d'augmenter la contamination.")
        else:
            # Formater le prix pour l'affichage
            deals_display = deals.copy()
            deals_display['prix'] = deals_display['prix'].apply(lambda x: fmt_eur(x))

            # Réorganiser les colonnes
            cols_order = [c for c in ['anomaly_score', 'prix', 'marque', 'modele', 'pays', 'mouvement', 'matiere_boitier', 'etat'] if c in deals_display.columns]
            other_cols = [c for c in deals_display.columns if c not in cols_order and c != 'is_outlier' and c != 'bonne_affaire']
            cols_order.extend(other_cols)

            st.dataframe(
                deals_display[cols_order],
                use_container_width=True,
                height=600
            )

            # Bouton de téléchargement
            csv = deals.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les bonnes affaires (CSV)",
                data=csv,
                file_name="bonnes_affaires_montres.csv",
                mime="text/csv"
            )

    with tab2:
        st.markdown("### 📈 Distribution des scores d'anomalie")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.histogram(
                resultats,
                x='anomaly_score',
                nbins=50,
                title="Distribution des scores d'anomalie",
                color='is_outlier',
                color_discrete_map={True: '#e74c3c', False: '#3498db'},
                labels={'is_outlier': 'Anomalie'}
            )
            fig.update_layout(
                xaxis_title="Score d'anomalie (plus élevé = plus anormal)",
                yaxis_title="Nombre de montres",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Statistiques des scores**")
            st.metric("Min", f"{resultats['anomaly_score'].min():.3f}")
            st.metric("Médiane", f"{resultats['anomaly_score'].median():.3f}")
            st.metric("Max", f"{resultats['anomaly_score'].max():.3f}")
            st.metric("Std", f"{resultats['anomaly_score'].std():.3f}")

        # Scatter plot prix vs score
        st.markdown("#### 💰 Prix vs Score d'anomalie")
        sample_size = min(5000, len(resultats))
        sample = resultats.sample(sample_size, random_state=42) if len(resultats) > sample_size else resultats

        fig2 = px.scatter(
            sample,
            x='anomaly_score',
            y='prix',
            color='is_outlier',
            hover_data=[c for c in ['marque', 'modele'] if c in sample.columns],
            title="Relation entre le score d'anomalie et le prix",
            color_discrete_map={True: '#e74c3c', False: '#3498db'},
            labels={'is_outlier': 'Anomalie'},
            opacity=0.6
        )
        fig2.add_hline(y=q1, line_dash="dash", line_color="green",
                      annotation_text=f"Q1 (25%): {fmt_eur(q1)}")
        fig2.update_layout(
            xaxis_title="Score d'anomalie",
            yaxis_title="Prix (€)",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 🔬 Analyse détaillée par marque")

        if 'marque' in resultats.columns:
            # Top marques avec le plus de bonnes affaires
            deals_by_marque = resultats[resultats['bonne_affaire']].groupby('marque').size().sort_values(ascending=False).head(10)

            if not deals_by_marque.empty:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        deals_by_marque,
                        title="Top 10 marques avec le plus de bonnes affaires",
                        color=deals_by_marque.values,
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(
                        xaxis_title='',
                        yaxis_title="Nombre de bonnes affaires",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Prix moyen des deals par marque
                    prix_moy_deals = (
                        resultats[resultats['bonne_affaire']]
                        .groupby('marque')['prix']
                        .agg(['mean', 'count'])
                        .sort_values('count', ascending=False)
                        .head(10)
                    )

                    fig2 = px.bar(
                        prix_moy_deals.reset_index(),
                        x='marque',
                        y='mean',
                        title="Prix moyen des bonnes affaires par marque",
                        text='mean',
                        color='mean',
                        color_continuous_scale='Blues'
                    )
                    fig2.update_traces(texttemplate='%{text:,.0f} €', textposition='outside')
                    fig2.update_layout(
                        xaxis_title='',
                        yaxis_title='Prix moyen (€)',
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("#### 📋 Détail des anomalies par marque")
                anom_by_marque = (
                    resultats.groupby('marque')
                    .agg({
                        'is_outlier': 'sum',
                        'bonne_affaire': 'sum',
                        'prix': ['count', 'mean', 'median']
                    })
                    .round(0)
                )
                anom_by_marque.columns = ['Anomalies', 'Bonnes affaires', 'Total', 'Prix moyen', 'Prix médian']
                anom_by_marque = anom_by_marque.sort_values('Bonnes affaires', ascending=False).head(20)
                st.dataframe(anom_by_marque, use_container_width=True)
            else:
                st.info("Aucune bonne affaire détectée pour l'analyse par marque.")
