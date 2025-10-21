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

st.set_page_config(page_title="Montres – Statistiques descriptives (Chrono24)", layout="wide")

# ------------------------------------------------------------------
# CHARGEMENT & NETTOYAGE (avec cache)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data(db_path: str, table: str) -> pd.DataFrame:
    """Charge la BDD SQLite -> DataFrame brut (sans nettoyage)."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'SELECT * FROM {table}', con)
    con.close()

    # Certains exports précédents enlevaient la première et la dernière colonne.
    # On garde un comportement robuste : si première colonne est id/index on la retire.
    if df.columns[0].lower() in ("id", "index"):
        df = df.iloc[:, 1:-1]
    return df

@st.cache_data(show_spinner=False)
def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Applique le pipeline de nettoyage existant (classe Nettoyage) et renvoie une base propre."""
    nettoyage = Nettoyage(df_raw.copy())

    # Même pipeline que dans ton script original
    df = nettoyage.nettoyage_colonnes()

    colonnes_a_renseigner = ['matiere_boitier',
                             'matiere_bracelet',
                             'sexe',
                             'diametre',
                             'etencheite',
                             'matiere_lunette',
                             'matiere_verre',
                             'boucle',
                             'matiere_boucle',
                             'rouage',
                             'reserve_de_marche',
                             'mouvement']

    for col in colonnes_a_renseigner:
        df = nettoyage.remplissage(col)

    df = nettoyage.remplissage_mouvement()
    df = nettoyage.remplissage_reserve_marche()
    df = nettoyage.compteur_complications('fonctions')
    df = nettoyage.suppression_colonnes()
    df = nettoyage.mise_en_forme()
    df = nettoyage.nettoyer_matiere_boitier()
    df = nettoyage.matiere()
    df = nettoyage.mapping_matiere()
    df = nettoyage.regroupement_etat_montres()
    df = nettoyage.extraction_elements_avant_euro()
    df = nettoyage.nettoyage_prix('prix')
    df = nettoyage.extraction_integer()

    # Harmonisation de types
    if 'Date_recup' in df.columns:
        df['Date_recup'] = pd.to_datetime(df['Date_recup'], errors='coerce')
    if 'prix' in df.columns:
        df['prix'] = pd.to_numeric(df['prix'], errors='coerce')
    if 'annee_prod' in df.columns:
        df['annee_prod'] = pd.to_numeric(df['annee_prod'], errors='coerce')
    if 'diametre' in df.columns:
        df['diametre'] = pd.to_numeric(df['diametre'], errors='coerce')

    # Trim des textes
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    return df

# ------------------------------------------------------------------
# HELPERS UI
# ------------------------------------------------------------------
def kpi_card(label: str, value, suffix: str = ""):
    st.metric(label, f"{value}{suffix}")


def top_n_counts(s: pd.Series, n: int = 10) -> pd.DataFrame:
    return s.value_counts(dropna=False).head(n).rename_axis("valeur").reset_index(name="nb")


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
with st.spinner("Chargement de la base..."):
    df_raw = load_raw_data(DB_PATH, TABLE)
    df = clean_dataframe(df_raw)

st.title("⌚ Plongée dans le marché horloger – Analyse interactive des montres (Chrono24)")
st.caption(f"Données récupérées par scraping sur le site Chrono24 du 16/04/2024 au 23/08/2025.")
st.write(f"Observations totales : **{len(df):,}**".replace(",", " "))



# ------------------------------------------------------------------
# SIDEBAR – FILTRES
# ------------------------------------------------------------------
st.sidebar.header("Filtres")

marques = sorted([m for m in df['marque'].dropna().unique()]) if 'marque' in df.columns else []
marque_sel = st.sidebar.multiselect("Marque", marques)

# Modèles filtrés par marques sélectionnées
if marque_sel and 'modele' in df.columns:
    models_pool = df[df['marque'].isin(marque_sel)]['modele'].dropna().unique()
else:
    models_pool = df['modele'].dropna().unique() if 'modele' in df.columns else []
modeles = sorted(models_pool)
modele_sel = st.sidebar.multiselect("Modèle", modeles)

mouv_sel = st.sidebar.multiselect("Mouvement", sorted(df['mouvement'].dropna().unique()) if 'mouvement' in df.columns else [])
etat_sel = st.sidebar.multiselect("État", sorted(df['etat'].dropna().unique()) if 'etat' in df.columns else [])
sexe_sel = st.sidebar.multiselect("Sexe", sorted(df['sexe'].dropna().unique()) if 'sexe' in df.columns else [])
pays_sel = st.sidebar.multiselect("Pays", sorted(df['pays'].dropna().unique()) if 'pays' in df.columns else [])

min_prix = float(np.nanmin(df['prix'])) if 'prix' in df.columns and df['prix'].notna().any() else 0.0
max_prix = float(np.nanmax(df['prix'])) if 'prix' in df.columns and df['prix'].notna().any() else 100000.0
prix_range = st.sidebar.slider("Prix (€)", min_value=0.0, max_value=max(1000.0, max_prix), value=(min_prix, max_prix), step=100.0)

if 'diametre' in df.columns:
    dmin = float(np.nanmin(df['diametre'])) if df['diametre'].notna().any() else 0.0
    dmax = float(np.nanmax(df['diametre'])) if df['diametre'].notna().any() else 60.0
    diam_range = st.sidebar.slider("Diamètre (mm)", min_value=0.0, max_value=max(10.0, dmax), value=(dmin, dmax), step=0.5)
else:
    diam_range = None

if 'annee_prod' in df.columns:
    amin = int(np.nanmin(df['annee_prod'])) if df['annee_prod'].notna().any() else 1950
    amax = int(np.nanmax(df['annee_prod'])) if df['annee_prod'].notna().any() else 2025
    annee_range = st.sidebar.slider("Année de production", min_value=1900, max_value=amax, value=(amin, amax), step=1)
else:
    annee_range = None

# Application des filtres
mask = pd.Series(True, index=df.index)
if marque_sel: mask &= df['marque'].isin(marque_sel)
if modele_sel: mask &= df['modele'].isin(modele_sel)
if mouv_sel: mask &= df['mouvement'].isin(mouv_sel)
if etat_sel: mask &= df['etat'].isin(etat_sel)
if sexe_sel: mask &= df['sexe'].isin(sexe_sel)
if pays_sel: mask &= df['pays'].isin(pays_sel)
if 'prix' in df.columns:
    mask &= df['prix'].between(prix_range[0], prix_range[1]) | df['prix'].isna()
if diam_range is not None and 'diametre' in df.columns:
    mask &= df['diametre'].between(diam_range[0], diam_range[1]) | df['diametre'].isna()
if annee_range is not None and 'annee_prod' in df.columns:
    mask &= df['annee_prod'].between(annee_range[0], annee_range[1]) | df['annee_prod'].isna()

dff = df[mask].copy()

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
c1, c2 = st.columns(2)
if 'prix' in dff.columns and dff['prix'].notna().any():
    kpi_card("Prix médian (€)", f"{int(np.nanmedian(dff['prix'])):,}".replace(",", " "))
else:
    kpi_card("Prix médian (€)", "—")
if 'marque' in dff.columns and not dff['marque'].dropna().empty:
    kpi_card("Marque la plus représentée", dff['marque'].mode().iloc[0])
else:
    kpi_card("Marque la plus représentée", "—")

# ------------------------------------------------------------------
# DISTRIBUTIONS
# ------------------------------------------------------------------
st.subheader("Répartition des prix")
if 'prix' in dff.columns and dff['prix'].notna().sum() > 0:
    fig_hist = px.histogram(dff, x='prix', nbins=50, title='Distribution des prix (€)', marginal='box')
    fig_hist.update_yaxes(title='Nombre d\'offres')
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
    if all(c in dff.columns for c in ['diametre', 'prix', 'marque']) and dff['diametre'].notna().sum() > 0 and dff['prix'].notna().sum() > 0:
        fig_scatter = px.scatter(
            dff.dropna(subset=['diametre', 'prix']),
            x='diametre', y='prix', color='marque',
            hover_data=['modele', 'mouvement', 'etat', 'pays'] if 'modele' in dff.columns else None,
            trendline='ols'
        )
        fig_scatter.update_layout(xaxis_title='Diamètre (mm)', yaxis_title='Prix (€)', legend_title='Marque')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Ajoute des valeurs de diamètre/prix pour voir ce graphique.")

# ------------------------------------------------------------------
# COMPOSITIONS / PARTS
# ------------------------------------------------------------------
colC, colD = st.columns(2)
with colC:
    st.subheader("Part des mouvements")
    if 'mouvement' in dff.columns:
        mv = top_n_counts(dff['mouvement'], n=10)
        fig_mv = px.pie(mv, names='valeur', values='nb', hole=0.5)
        st.plotly_chart(fig_mv, use_container_width=True)
    else:
        st.info("Colonne 'mouvement' absente.")

with colD:
    st.subheader("Top pays vendeurs")
    if 'pays' in dff.columns:
        ct = top_n_counts(dff['pays'], n=12)
        fig_ct = px.bar(ct, x='valeur', y='nb')
        fig_ct.update_layout(xaxis_title='', yaxis_title='Nombre d\'offres')
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.info("Colonne 'pays' absente.")

# ------------------------------------------------------------------
# ÉVOLUTION DES PRIX PAR MARQUE / MODÈLE
# ------------------------------------------------------------------
st.header("📉 Évolution des prix par modèle")
st.caption("Choisis une marque et, idéalement, un modèle. Agrégation médiane sur la période choisie.")

marque_for_ts = st.selectbox("Marque (série temporelle)", ["—"] + marques if marques else ["—"], index=0)
models_for_ts = []
if marque_for_ts != "—" and 'modele' in df.columns:
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
            if len(recent) >= 2 and recent.iloc[-2] != 0:
                pct = (recent.iloc[-1] - recent.iloc[-2]) / recent.iloc[-2] * 100
                kpi_card("Var. dernière période", f"{pct:+.1f}%")
            else:
                kpi_card("Var. dernière période", "—")
            kpi_card("Dernière médiane (€)", f"{int(ser['prix_median'].iloc[-1]):,}".replace(",", " "))
            kpi_card("Nb périodes", len(ser))
else:
    st.info("Sélectionne une marque (et éventuellement un modèle) pour afficher la série.")

# ------------------------------------------------------------------
# APERÇU & EXPORT
# ------------------------------------------------------------------
st.subheader("Aperçu des données filtrées")
show_cols = ['marque', 'modele', 'mouvement', 'matiere_boitier', 'matiere_bracelet',
             'annee_prod', 'etat', 'sexe', 'prix', 'reserve_de_marche', 'diametre',
             'etencheite', 'matiere_lunette', 'matiere_verre', 'boucle',
             'matiere_boucle', 'rouage', 'fonctions', 'Date_recup',
             'comptage_fonctions', 'pays']
cols_presentes = [c for c in show_cols if c in dff.columns]
st.dataframe(dff[cols_presentes].head(500), use_container_width=True)

colE, colF = st.columns(2)
with colE:
    csv = dff[cols_presentes].to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger CSV", csv, file_name="montres_filtrees.csv", mime="text/csv")
with colF:
    try:
        import io
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            dff[cols_presentes].to_excel(writer, index=False, sheet_name="Données")
        st.download_button("Télécharger Excel", excel_buffer.getvalue(), file_name="montres_filtrees.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.caption("")

