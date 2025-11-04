# ⌚ Analyse interactive du marché des montres (Chrono24)

Ce projet est une application **Streamlit** permettant d’explorer et d’analyser le marché des montres d’occasion à partir de données issues de Chrono24 (scraping réalisé entre avril 2024 et août 2025).
L’application combine analyses descriptives, détection d’anomalies (“bonnes affaires”), segmentation du marché (clustering) et modélisation du prix.


## 🚀 Fonctionnalités principale

### 🔎 1. Analyse du marché
- KPIs : prix médian, marque la plus représentée.
- Visualisations : histogrammes, boxplots, top 10 pays vendeurs.
- Répartitions : mouvements (automatique, quartz, etc.), matières de boîtier, etc.

### ⚠️ 2. Détection d’anomalies / Bonnes affaires
- Identification des montres à prix anormalement bas via Isolation Forest.
- Visualisation de la distribution des scores d’anomalie.

### 🧩 3. Segmentation du marché
- Clustering K-Means sur les caractéristiques clés : prix, diamètre, année, réserve de marche.
- Visualisation 2D des clusters (PCA) et profils médians par cluster.

## 🤖 4. Modélisation du prix
- Régression linéaire, Ridge, Random Forest, Gradient Boosting, Huber.
- Mesures de performance (R², MAE, RMSE) et cross-validation.


## 🛠️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-profil/analyse_marche_montres.git
cd analyse_marche_montres
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   
.venv\Scripts\activate      
```
### 3. Installer les dépendances
   
```bash
pip install -r requirements.txt
```

### 4. Lancer l’application

```bash
streamlit run main.py
```

## 📂 Structure du projet

```bash
├── main.py                   lanceur
├── app/
│   ├── __init__.py
│   ├── app.py                # Gestion page et des onglets
│   ├── config.py             # Configuration (chemins, titres, etc.)
│   ├── data.py               # Chargement Excel 
│   ├── utils.py              # Fonctions utilitaires
│   └── tabs/                 # Modules d’analyse par onglet
│       ├── overview.py
│       ├── analyses.py
│       ├── anomalies.py
│       ├── segmentation.py
│       └── regression.py
├── data/
│   └── propre.db / propre.xlsx   # Données source (scraping Chrono24)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 📊 Données

Source : Chrono24
Période de collecte : avril 2024 → août 2025

Variables principales :
- marque, modèle, prix, diamètre, année_production
- mouvement, matière_boitier, matière_bracelet, matière_lunette
- état, sexe, pays, réserve_de_marche, étanchéité

  

Projet réalisé par BOUSSENGUI François, passionné de Data Science et d’horlogerie. ⌚
