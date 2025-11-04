# app/tabs/regression.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.utils import colonnes_numeriques_disponibles, fmt_eur

def render_tab_regression(echantillon: pd.DataFrame) -> None:
    st.subheader("Modélisation explicative du prix (€)")
    st.caption("Estimator + prétraitements; métriques sur échantillon de test (+ CV optionnelle).")
    try:
        from sklearn.model_selection import train_test_split, KFold, cross_val_score
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    except Exception:
        st.info("`scikit-learn` n'est pas installé. Installe-le avec: `pip install scikit-learn`.")
        return

    vars_num_cand = ['diametre', 'reserve_de_marche', 'annee_prod', 'etencheite']
    dispo_num = colonnes_numeriques_disponibles(echantillon, vars_num_cand)
    vars_cat_cand = ['marque','modele','mouvement','etat','matiere_boitier','matiere_bracelet',
                     'matiere_lunette','matiere_verre','boucle','matiere_boucle','rouage','sexe','pays']
    dispo_cat = [c for c in vars_cat_cand if c in echantillon.columns]

    if 'prix' not in echantillon.columns:
        st.warning("Colonne 'prix' absente : régression impossible.")
        return
    if len(dispo_num) == 0 and len(dispo_cat) == 0:
        st.info("Aucune variable disponible pour modéliser le prix.")
        return

    c1, c2 = st.columns(2)
    with c1:
        defaut_num = [v for v in ['diametre','reserve_de_marche'] if v in dispo_num]
        vars_num = st.multiselect("Variables numériques", options=dispo_num,
                                  default=defaut_num if len(defaut_num) >= 1 else dispo_num[:1], key="reg_num_vars")
    with c2:
        defaut_cat = [v for v in ['marque','modele','mouvement'] if v in dispo_cat]
        vars_cat = st.multiselect("Variables catégorielles", options=dispo_cat, default=defaut_cat, key="reg_cat_vars")

    c3, c4, c5 = st.columns(3)
    with c3:
        standardiser = st.checkbox("Standardiser numériques", value=True, key="reg_std")
    with c4:
        log_cible = st.checkbox("Modéliser log(prix)", value=True, key="reg_log")
    with c5:
        tronquer = st.checkbox("Limiter extrêmes (1–99e pct.)", value=True, key="reg_winsor")

    if len(vars_num) == 0 and len(vars_cat) == 0:
        st.info("Sélectionne au moins 1 variable explicative.")
        return

    cols = vars_num + vars_cat + ['prix']
    data = echantillon[cols].dropna().copy()
    if data.shape[0] < 200:
        st.info("Échantillon restreint (<200 lignes) : métriques potentiellement instables.")

    if tronquer and len(vars_num) > 0:
        for col in vars_num + ['prix']:
            q1, q99 = data[col].quantile([0.01, 0.99])
            data[col] = data[col].clip(q1, q99)

    y = data['prix'].values
    y_model = np.log1p(y) if log_cible else y

    taille_test = st.slider("Part d'échantillon de test", 0.1, 0.4, 0.2, 0.05, key="reg_test_size")
    rng = st.number_input("Random state", value=42, step=1, key="reg_rng")

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse=False)

    transfos = []
    if len(vars_num) > 0:
        transfos.append(('num', StandardScaler() if standardiser else 'passthrough', vars_num))
    if len(vars_cat) > 0:
        transfos.append(('cat', ohe, vars_cat))
    prep = ColumnTransformer(transfos)

    algo = st.radio("Modèle",
                    ["Régression linéaire", "Ridge (L2)", "Random Forest", "Gradient Boosting (Histogramme)", "Régression robuste (Huber)"],
                    index=3, horizontal=True, key="reg_algo")

    if algo == "Régression linéaire":
        est = LinearRegression()
    elif algo == "Ridge (L2)":
        alpha = st.slider("Alpha (Ridge)", 0.01, 10.0, 1.0, 0.01, key="reg_ridge_alpha")
        est = Ridge(alpha=alpha)
    elif algo == "Random Forest":
        n_estim = st.slider("n_estimators (RF)", 100, 1000, 400, 50, key="reg_rf_n_estim")
        max_depth = st.slider("max_depth (RF)", 2, 20, 10, 1, key="reg_rf_depth")
        est = RandomForestRegressor(n_estimators=int(n_estim), max_depth=int(max_depth), random_state=int(rng), n_jobs=-1)
    elif algo == "Régression robuste (Huber)":
        est = HuberRegressor()
    else:
        lrate = st.slider("learning_rate (HGBR)", 0.01, 0.3, 0.1, 0.01, key="reg_hgbr_lr")
        max_leaf = st.slider("max_leaf_nodes (HGBR)", 15, 63, 31, 2, key="reg_hgbr_leaf")
        est = HistGradientBoostingRegressor(learning_rate=float(lrate), max_leaf_nodes=int(max_leaf), random_state=int(rng))

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('prep', prep), ('model', est)])
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=['prix'])
    X_train, X_test, y_train, y_test = train_test_split(X, y_model, test_size=taille_test, random_state=int(rng))

    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)

    if log_cible:
        y_test_eur = np.expm1(y_test)
        y_pred_eur = np.expm1(y_pred_test)
    else:
        y_test_eur = y_test
        y_pred_eur = y_pred_test

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test_eur, y_pred_eur)
    rmse = float(np.sqrt(mean_squared_error(y_test_eur, y_pred_eur)))

    k1, k2, k3 = st.columns(3)
    k1.metric("R² (test)", f"{r2:.3f}")
    k2.metric("MAE (€/test)", fmt_eur(mae))
    k3.metric("RMSE (€/test)", fmt_eur(rmse))

    do_cv = st.checkbox("Cross-validation (5 folds)", value=True, key="reg_cv")
    if do_cv:
        from sklearn.model_selection import KFold, cross_val_score
        kf = KFold(n_splits=5, shuffle=True, random_state=int(rng))
        cv_mae = -cross_val_score(pipe, X, y_model, cv=kf, scoring='neg_mean_absolute_error')
        cv_rmse = np.sqrt(-cross_val_score(pipe, X, y_model, cv=kf, scoring='neg_mean_squared_error'))
        st.caption(f"CV (5-fold) — MAE: **{cv_mae.mean():,.0f} ± {cv_mae.std():,.0f} €**, RMSE: **{cv_rmse.mean():,.0f} ± {cv_rmse.std():,.0f} €**".replace(",", " "))

    st.caption("Comparaison sur l'échantillon de test")
    comp = pd.DataFrame({'observé (€)': y_test_eur, 'prédit (€)': y_pred_eur})
    fig = px.scatter(comp, x='observé (€)', y='prédit (€)')
    fig.add_trace(go.Scatter(x=[comp['observé (€)'].min(), comp['observé (€)'].max()],
                             y=[comp['observé (€)'].min(), comp['observé (€)'].max()],
                             mode='lines', name='Parité (y=x)'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Table des prédictions vs observé")
    comp['erreur (€)'] = comp['prédit (€)'] - comp['observé (€)']
    comp['erreur (%)'] = np.where(comp['observé (€)'] > 0, 100 * comp['erreur (€)'] / comp['observé (€)'], np.nan)
    comp = comp.round({'observé (€)': 0, 'prédit (€)': 0, 'erreur (€)': 0, 'erreur (%)': 1})
    st.dataframe(comp.head(2000), use_container_width=True)