# ⌚ Analyse interactive du marché des montres (Chrono24)

https://watchanalytics.streamlit.app/

Ce projet est une application **Streamlit** permettant d'explorer et d'analyser le marché des montres d'occasion à partir de données issues de Chrono24 (scraping réalisé entre avril 2024 et août 2024).
L'application combine analyses descriptives, détection d'anomalies ("bonnes affaires"), segmentation du marché (clustering) et modélisation du prix.


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

### Option 1 : Installation locale

#### 1. Cloner le dépôt

```bash
git clone https://github.com/Francois-b-24/analyse_marche_montres.git
cd analyse_marche_montres
```

#### 2. Utiliser le Makefile (recommandé)

```bash
make install  # Crée l'environnement virtuel et installe les dépendances
make run      # Lance l'application
```

#### Ou manuellement :

```bash
python -m venv .venv
source .venv/bin/activate      # Sur Linux/Mac
# .venv\Scripts\activate       # Sur Windows

pip install -r requirements.txt
streamlit run main.py
```

### Option 2 : Utilisation avec Docker

#### Méthode rapide avec Makefile :

```bash
make docker-build  # Construire l'image
make docker-up     # Démarrer les containers
make docker-down   # Arrêter les containers
```

#### Ou avec docker-compose :

```bash
docker-compose up -d          # Démarrer
docker-compose down           # Arrêter
docker-compose logs -f app    # Voir les logs
```

L'application sera accessible sur http://localhost:8701

## 📂 Structure du projet

```bash
├── main.py                   # Point d'entrée de l'application
├── app/
│   ├── __init__.py
│   ├── app.py                # Gestion de la page et des onglets
│   ├── config.py             # Configuration (chemins, titres, etc.)
│   ├── data.py               # Chargement des données Excel
│   ├── utils.py              # Fonctions utilitaires
│   └── tabs/                 # Modules d'analyse par onglet
│       ├── __init__.py
│       ├── overview.py       # Vue d'ensemble et KPIs
│       ├── analyses.py       # Analyses descriptives
│       ├── anomalies.py      # Détection d'anomalies (Isolation Forest)
│       ├── segmentation.py   # Clustering K-Means
│       └── regression.py     # Modélisation du prix
├── data/
│   └── propre.xlsx           # Données source (scraping Chrono24)
├── _nginx_/
│   └── conf.d/
│       └── default.conf      # Configuration nginx
├── requirements.txt          # Dépendances Python
├── Dockerfile                # Configuration Docker
├── docker-compose.yml        # Orchestration des services
├── .gitignore                # Fichiers à ignorer par Git
├── .dockerignore             # Fichiers à ignorer par Docker
├── Makefile                  # Commandes utiles
└── README.md                 # Documentation
```

## 📊 Données

Source : Chrono24
Période de collecte : avril 2024 → août 2024

Variables principales :
- marque, modèle, prix, diamètre, année_production
- mouvement, matière_boitier, matière_bracelet, matière_lunette
- état, sexe, pays, réserve_de_marche, étanchéité

## 🔧 Maintenance

### Commandes utiles avec Makefile :

```bash
make help          # Afficher toutes les commandes
make clean         # Nettoyer les fichiers temporaires
```

### Mise à jour des dépendances :

```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📝 Licence

Ce projet est sous licence MIT.

---

Projet réalisé par BOUSSENGUI François, passionné de Data Science et d'horlogerie. ⌚
