import sys
import os
sys.path.append(os.path.abspath("/Users/f.b/Desktop/Data_Science/Data_Science/Projets/analyse_marche_montres/"))

import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import Nettoyage

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DB_PATH = '/Users/f.b/Desktop/Data_Science/preowned-watch-predictor/data/raw/montre.db'
TABLE = 'montre'
PAGE_TITLE = "Montres – Statistiques descriptives (Chrono24)"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------

def safe_unique_sorted(df: pd.DataFrame, col: str) -> list:
    """Return sorted unique values for a column if it exists, else []."""
    if col in df.columns:
        vals = df[col].dropna().unique()
        return sorted([v for v in vals if str(v).strip() != ""])
    return []


def numeric_range(series: pd.Series, default_min: float, default_max: float) -> tuple[float, float]:
    if series.notna().any():
        return float(np.nanmin(series)), float(np.nanmax(series))
    return default_min, default_max


def top_n_counts(s: pd.Series, n: int = 10) -> pd.DataFrame:
    return s.value_counts(dropna=False).head(n).rename_axis("valeur").reset_index(name="nb")


def available_numeric_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------
# DATA LOADING & CLEANING (cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data(db_path: str, table: str) -> pd.DataFrame:
    """Load SQLite table into a DataFrame, robust to index/id columns."""
    try:
        con = sqlite3.connect(db_path)
    except Exception as e:
        st.error(f"Impossible d'ouvrir la base SQLite: {e}")
        st.stop()
    try:
        df = pd.read_sql_query(f'SELECT * FROM {table}', con)
    except Exception as e:
        con.close()
        st.error(f"Échec de lecture de la table '{table}': {e}")
        st.stop()
    finally:
        try:
            con.close()
        except Exception:
            pass

    # Retirer une éventuelle première colonne 'id/index'; ne pas couper la dernière colonne
    if len(df.columns) > 0 and df.columns[0].lower() in ("id", "index"):
        df = df.iloc[:, 1:-1]
    return df


@st.cache_data(show_spinner=False)
def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply Nettoyage pipeline safely and harmonize dtypes."""
    try:
        nettoyage = Nettoyage(df_raw.copy())
        df = nettoyage.nettoyage_colonnes()

        colonnes_a_renseigner = [
            'matiere_boitier', 'matiere_bracelet', 'sexe', 'diametre', 'etencheite',
            'matiere_lunette', 'matiere_verre', 'boucle', 'matiere_boucle', 'rouage',
            'reserve_de_marche', 'mouvement'
        ]
        for col in colonnes_a_renseigner:
            if col in df.columns:
                df = nettoyage.remplissage(col)

        if 'mouvement' in df.columns:
            df = nettoyage.remplissage_mouvement()
        if 'reserve_de_marche' in df.columns:
            df = nettoyage.remplissage_reserve_marche()
        if 'fonctions' in df.columns:
            df = nettoyage.compteur_complications('fonctions')

        # Suite du pipeline (méthodes idempotentes/robustes)
        for step in [
            nettoyage.suppression_colonnes,
            nettoyage.mise_en_forme,
            nettoyage.nettoyer_matiere_boitier,
            nettoyage.matiere,
            nettoyage.mapping_matiere,
            nettoyage.regroupement_etat_montres,
            nettoyage.extraction_elements_avant_euro,
        ]:
            try:
                df = step()
            except Exception:
                # Si une étape échoue, on continue avec le df actuel
                pass

        # Nettoyage prix / extraction numeriques si colonnes présentes
        if 'prix' in df.columns:
            try:
                df = nettoyage.nettoyage_prix('prix')
            except Exception:
                pass
        try:
            df = nettoyage.extraction_integer()
        except Exception:
            pass

        # Harmonisation de types
        if 'Date_recup' in df.columns:
            df['Date_recup'] = pd.to_datetime(df['Date_recup'], errors='coerce')
        for col_num in ['prix', 'annee_prod', 'diametre', 'reserve_de_marche']:
            if col_num in df.columns:
                df[col_num] = pd.to_numeric(df[col_num], errors='coerce')

        # Trim des textes
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

        return df
    except Exception as e:
        st.error(f"Erreur pendant le nettoyage des données: {e}")
        return df_raw.copy()


# ------------------------------------------------------------------
# FILTERS / SIDEBAR
# ------------------------------------------------------------------

def build_filters(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filtres")

    marques = safe_unique_sorted(df, 'marque')
    marque_sel = st.sidebar.multiselect("Marque", marques)

    # Modèles filtrés par marques sélectionnées
    if marque_sel and 'modele' in df.columns and 'marque' in df.columns:
        models_pool = df[df['marque'].isin(marque_sel)]['modele'].dropna().unique()
    else:
        models_pool = df['modele'].dropna().unique() if 'modele' in df.columns else []
    modeles = sorted(models_pool)
    modele_sel = st.sidebar.multiselect("Modèle", modeles)

    mouv_sel = st.sidebar.multiselect("Mouvement", safe_unique_sorted(df, 'mouvement'))
    etat_sel = st.sidebar.multiselect("État", safe_unique_sorted(df, 'etat'))
    sexe_sel = st.sidebar.multiselect("Sexe", safe_unique_sorted(df, 'sexe'))
    pays_sel = st.sidebar.multiselect("Pays", safe_unique_sorted(df, 'pays'))

    # Sliders numériques
    if 'prix' in df.columns:
        min_prix, max_prix = numeric_range(df['prix'], 0.0, 100000.0)
        prix_range = st.sidebar.slider("Prix (€)", min_value=0.0, max_value=max(1000.0, max_prix), value=(min_prix, max_prix), step=100.0)
    else:
        prix_range = None

    if 'diametre' in df.columns:
        dmin, dmax = numeric_range(df['diametre'], 0.0, 60.0)
        diam_range = st.sidebar.slider("Diamètre (mm)", min_value=0.0, max_value=max(10.0, dmax), value=(dmin, dmax), step=0.5)
    else:
        diam_range = None

    if 'annee_prod' in df.columns:
        amin, amax = numeric_range(df['annee_prod'], 1950.0, 2025.0)
        amin, amax = int(amin), int(amax)
        annee_range = st.sidebar.slider("Année de production", min_value=1900, max_value=max(amax, 1900), value=(amin, amax), step=1)
    else:
        annee_range = None

    return {
        'marque_sel': marque_sel,
        'modele_sel': modele_sel,
        'mouv_sel': mouv_sel,
        'etat_sel': etat_sel,
        'sexe_sel': sexe_sel,
        'pays_sel': pays_sel,
        'prix_range': prix_range,
        'diam_range': diam_range,
        'annee_range': annee_range,
    }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f['marque_sel'] and 'marque' in df.columns:
        mask &= df['marque'].isin(f['marque_sel'])
    if f['modele_sel'] and 'modele' in df.columns:
        mask &= df['modele'].isin(f['modele_sel'])
    if f['mouv_sel'] and 'mouvement' in df.columns:
        mask &= df['mouvement'].isin(f['mouv_sel'])
    if f['etat_sel'] and 'etat' in df.columns:
        mask &= df['etat'].isin(f['etat_sel'])
    if f['sexe_sel'] and 'sexe' in df.columns:
        mask &= df['sexe'].isin(f['sexe_sel'])
    if f['pays_sel'] and 'pays' in df.columns:
        mask &= df['pays'].isin(f['pays_sel'])
    if f['prix_range'] and 'prix' in df.columns:
        a, b = f['prix_range']
        mask &= df['prix'].between(a, b) | df['prix'].isna()
    if f['diam_range'] and 'diametre' in df.columns:
        a, b = f['diam_range']
        mask &= df['diametre'].between(a, b) | df['diametre'].isna()
    if f['annee_range'] and 'annee_prod' in df.columns:
        a, b = f['annee_range']
        mask &= df['annee_prod'].between(a, b) | df['annee_prod'].isna()
    return df[mask].copy()


# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------
with st.spinner("Chargement de la base..."):
    df_raw = load_raw_data(DB_PATH, TABLE)
    df = clean_dataframe(df_raw)

st.title("⌚ Plongée dans le marché horloger – Analyse interactive des montres (Chrono24)")
st.caption("Données récupérées par scraping sur le site Chrono24 du 16/04/2024 au 23/08/2025.")
st.write(f"Observations totales : **{len(df):,}**".replace(",", " "))

# Filters
filters = build_filters(df)
dff = apply_filters(df, filters)

# Tabs (simulate multi-page without creating extra files)
TAB_ANALYSE, TAB_ANOMALIES, TAB_SEGMENTATION, TAB_EXPORT = st.tabs([
    "Analyse du marché", "Anomalies / Deals", "Segmentation du marché", "Export"
])

# ------------------------------------------------------------------
# TAB 1 – ANALYSE DU MARCHÉ
# ------------------------------------------------------------------
with TAB_ANALYSE:
    # KPIs
    c1, c2 = st.columns(2)
    with c1:
        if 'prix' in dff.columns and dff['prix'].notna().any():
            st.metric("Prix médian (€)", f"{int(np.nanmedian(dff['prix'])):,}".replace(",", " "))
        else:
            st.metric("Prix médian (€)", "—")
    with c2:
        if 'marque' in dff.columns and not dff['marque'].dropna().empty:
            st.metric("Marque la plus représentée", dff['marque'].mode().iloc[0])
        else:
            st.metric("Marque la plus représentée", "—")

    # DISTRIBUTIONS
    st.subheader("Répartition des prix")
    if 'prix' in dff.columns and dff['prix'].notna().sum() > 0:
        fig_hist = px.histogram(dff, x='prix', nbins=50, title='Distribution des prix (€)', marginal='box')
        fig_hist.update_yaxes(title="Nombre d'offres")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Pas de prix renseignés pour l’échantillon filtré.")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Prix par marque (boîte à moustaches)")
        if all(c in dff.columns for c in ['prix', 'marque']) and dff['prix'].notna().sum() > 0 and dff['marque'].notna().sum() > 0:
            top_marques = dff['marque'].value_counts().head(15).index
            tmp = dff[dff['marque'].isin(top_marques)]
            fig_box = px.box(tmp, x='marque', y='prix', points='outliers')
            fig_box.update_layout(xaxis_title='', yaxis_title='Prix (€)')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Données insuffisantes pour le boxplot.")

    with colB:
        st.subheader("Diamètre vs. prix")
        if all(c in dff.columns for c in ['diametre', 'prix']) and dff[['diametre','prix']].notna().sum().min() > 0:
            fig_scatter = px.scatter(
                dff.dropna(subset=['diametre', 'prix']),
                x='diametre', y='prix', color='marque' if 'marque' in dff.columns else None,
                hover_data=[c for c in ['modele', 'mouvement', 'etat', 'pays'] if c in dff.columns]
            )
            fig_scatter.update_layout(xaxis_title='Diamètre (mm)', yaxis_title='Prix (€)', legend_title='Marque')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Ajoute des valeurs de diamètre/prix pour voir ce graphique.")

    # COMPOSITIONS / PARTS
    colC, colD = st.columns(2)
    with colC:
        st.subheader("Part des mouvements")
        if 'mouvement' in dff.columns and not dff['mouvement'].empty:
            mv = top_n_counts(dff['mouvement'], n=10)
            if len(mv) > 0:
                fig_mv = px.pie(mv, names='valeur', values='nb', hole=0.5)
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.info("Aucune donnée de mouvement à afficher.")
        else:
            st.info("Colonne 'mouvement' absente.")

    with colD:
        st.subheader("Top pays vendeurs")
        if 'pays' in dff.columns and not dff['pays'].empty:
            ct = top_n_counts(dff['pays'], n=12)
            if len(ct) > 0:
                fig_ct = px.bar(ct, x='valeur', y='nb')
                fig_ct.update_layout(xaxis_title='', yaxis_title="Nombre d'offres")
                st.plotly_chart(fig_ct, use_container_width=True)
            else:
                st.info("Aucune donnée de pays à afficher.")
        else:
            st.info("Colonne 'pays' absente.")

    # ÉVOLUTION DES PRIX PAR MARQUE / MODÈLE
    st.header("📉 Évolution des prix par modèle")
    st.caption("Choisis une marque et, idéalement, un modèle. Agrégation médiane sur la période choisie.")

    marques_all = ["—"] + safe_unique_sorted(df, 'marque')
    marque_for_ts = st.selectbox("Marque (série temporelle)", marques_all, index=0)

    models_for_ts = []
    if marque_for_ts != "—" and 'modele' in df.columns and 'marque' in df.columns:
        models_for_ts = sorted(df.loc[df['marque'] == marque_for_ts, 'modele'].dropna().unique())
    modele_for_ts = st.selectbox("Modèle (optionnel)", ["—"] + models_for_ts, index=0)

    agg_choice = st.radio("Fréquence", ["Mensuelle", "Hebdomadaire", "Journalière"], horizontal=True)
    freq_map = {"Mensuelle": "M", "Hebdomadaire": "W", "Journalière": "D"}
    freq = freq_map[agg_choice]

    if marque_for_ts != "—" and 'Date_recup' in df.columns and 'prix' in df.columns:
        dfts = df[df['marque'] == marque_for_ts].copy()
        if modele_for_ts != "—" and 'modele' in dfts.columns:
            dfts = dfts[dfts['modele'] == modele_for_ts]

        if dfts['Date_recup'].isna().all():
            st.warning("Pas de dates disponibles pour tracer l’évolution.")
        elif dfts['prix'].notna().sum() < 3:
            st.warning("Trop peu de points de prix pour tracer une tendance.")
        else:
            dfts = dfts.dropna(subset=['Date_recup', 'prix'])
            ser = (
                dfts.set_index('Date_recup')
                    .sort_index()
                    .resample(freq)['prix']
                    .median()
                    .to_frame('prix_median')
                    .reset_index()
            )
            if len(ser) == 0:
                st.info("Aucune donnée après agrégation.")
            else:
                titre = f"Évolution des prix – {marque_for_ts}" + (f" / {modele_for_ts}" if modele_for_ts != "—" else " (tous modèles)")
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=ser['Date_recup'], y=ser['prix_median'], mode='lines+markers', name='Prix médian'))
                fig_ts.update_layout(title=titre, xaxis_title='Date', yaxis_title='Prix (€)')
                st.plotly_chart(fig_ts, use_container_width=True)

                # KPIs récentes
                recent = ser.tail(3)['prix_median']
                if len(recent) >= 2 and recent.iloc[-2] not in (0, np.nan):
                    try:
                        pct = (recent.iloc[-1] - recent.iloc[-2]) / recent.iloc[-2] * 100
                        st.metric("Var. dernière période", f"{pct:+.1f}%")
                    except Exception:
                        st.metric("Var. dernière période", "—")
                else:
                    st.metric("Var. dernière période", "—")
                st.metric("Dernière médiane (€)", f"{int(ser['prix_median'].iloc[-1]):,}".replace(",", " "))
                st.metric("Nb périodes", len(ser))
    else:
        st.info("Sélectionne une marque (et éventuellement un modèle) pour afficher la série.")

# ------------------------------------------------------------------
# TAB 2 – ANOMALIES / DEALS
# ------------------------------------------------------------------
with TAB_ANOMALIES:
    st.subheader("Détection d'anomalies sur les prix")
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception:
        st.info("`scikit-learn` n'est pas installé. Installe-le avec: `pip install scikit-learn`.")
    else:
        feature_candidates = ['prix', 'diametre', 'annee_prod', 'reserve_de_marche']
        feats = available_numeric_cols(dff, feature_candidates)
        if 'prix' not in feats:
            st.warning("Colonne 'prix' requise pour la détection d'anomalies.")
        else:
            X = dff[feats].dropna().copy()
            if len(X) < 50:
                st.info("Échantillon trop petit pour un modèle robuste (min ~50 lignes après NA).")
            else:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                contamination = st.slider("Proportion d'anomalies attendue", 0.01, 0.10, 0.04, 0.01)
                rng = st.number_input("Random state", value=42, step=1)
                model = IsolationForest(n_estimators=300, contamination=contamination, random_state=rng)
                model.fit(Xs)
                scores = model.decision_function(Xs)  # plus grand = plus normal
                labels = model.predict(Xs)  # -1 = anomalie

                res = dff.loc[X.index, ['marque','modele','pays'] + feats].copy()
                res['anomaly_score'] = -scores  # plus grand = plus anormal
                res['is_outlier'] = (labels == -1)

                # Définir un "deal" comme une anomalie à prix bas (prix < Q1 et outlier)
                q1 = res['prix'].quantile(0.25)
                res['good_deal'] = res['is_outlier'] & (res['prix'] <= q1)

                st.caption("Les bonnes affaires sont les anomalies à prix bas vs. le reste des caractéristiques.")

                col1, col2 = st.columns([2,1])
                with col1:
                    top_deals = res[res['good_deal']].sort_values('anomaly_score', ascending=False).head(30)
                    if top_deals.empty:
                        st.info("Aucune bonne affaire détectée selon les paramètres actuels.")
                    else:
                        st.dataframe(top_deals, use_container_width=True)
                with col2:
                    fig_an = px.histogram(res, x='anomaly_score', nbins=40, title="Distribution des scores d'anomalie")
                    st.plotly_chart(fig_an, use_container_width=True)

# ------------------------------------------------------------------
# TAB 3 – SEGMENTATION / CLUSTERING
# ------------------------------------------------------------------
with TAB_SEGMENTATION:
    st.subheader("Segmentation non supervisée (K-Means)")
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except Exception:
        st.info("`scikit-learn` n'est pas installé. Installe-le avec: `pip install scikit-learn`.")
    else:
        feature_candidates = ['prix', 'diametre', 'annee_prod', 'reserve_de_marche']
        feats = available_numeric_cols(dff, feature_candidates)
        if len(feats) < 2:
            st.info("Sélectionne/complète au moins 2 variables numériques parmi : prix, diamètre, année, réserve de marche.")
        else:
            X = dff[feats].dropna().copy()
            if len(X) < 50:
                st.info("Échantillon trop petit pour une segmentation robuste (min ~50 lignes après NA).")
            else:
                k = st.slider("Nombre de clusters (k)", 2, 10, 5, 1)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labs = km.fit_predict(Xs)

                # Réduction en 2D pour visualisation
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(Xs)
                vis = pd.DataFrame(coords, columns=['PC1','PC2'], index=X.index)
                vis['cluster'] = labs.astype(int)
                for meta in ['marque','modele','pays']:
                    if meta in dff.columns:
                        vis[meta] = dff.loc[vis.index, meta]
                for f in feats:
                    vis[f] = dff.loc[vis.index, f]

                st.caption("Projection PCA des clusters (2D).")
                fig_clu = px.scatter(
                    vis, x='PC1', y='PC2', color='cluster',
                    hover_data=[c for c in ['marque','modele','pays'] + feats if c in vis.columns]
                )
                st.plotly_chart(fig_clu, use_container_width=True)

                # Profil des clusters
                st.subheader("Profil des clusters (médianes par variable)")
                profile = vis.groupby('cluster')[feats].median().reset_index()
                st.dataframe(profile, use_container_width=True)

# ------------------------------------------------------------------
# TAB 4 – EXPORT
# ------------------------------------------------------------------
with TAB_EXPORT:
    st.subheader("Aperçu des données filtrées & exports")
    show_cols = ['marque', 'modele', 'mouvement', 'matiere_boitier', 'matiere_bracelet',
                 'annee_prod', 'etat', 'sexe', 'prix', 'reserve_de_marche', 'diametre',
                 'etencheite', 'matiere_lunette', 'matiere_verre', 'boucle',
                 'matiere_boucle', 'rouage', 'fonctions', 'Date_recup',
                 'comptage_fonctions', 'pays']
    cols_presentes = [c for c in show_cols if c in dff.columns]
    st.dataframe(dff[cols_presentes].head(500), use_container_width=True)

    colE, colF = st.columns(2)
    with colE:
        try:
            csv = dff[cols_presentes].to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV", csv, file_name="montres_filtrees.csv", mime="text/csv")
        except Exception as e:
            st.caption(f"Export CSV indisponible: {e}")
    with colF:
        try:
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                dff[cols_presentes].to_excel(writer, index=False, sheet_name="Données")
            st.download_button("Télécharger Excel", excel_buffer.getvalue(), file_name="montres_filtrees.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.caption(f"Export Excel indisponible: {e}")
