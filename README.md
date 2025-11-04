# ⌚ Analyse interactive du marché des montres (Chrono24)

Ce projet est une application Streamlit multipage permettant d’explorer et d’analyser le marché des montres d’occasion à partir de données récupérées par scraping sur le site Chrono24.
L’application combine analyses descriptives, détection d’anomalies (bonnes affaires), segmentation du marché (clustering) et exports interactifs.

⸻

## 🚀 Fonctionnalités principale

### 🔎1. Analyse du marché
- KPIs : prix médian, marque la plus représentée.
- Distribution des prix et des diamètres.
- Comparaisons par marque (boxplots, scatter plots).
- Répartition des mouvements (automatique, quartz, etc.).
- Top pays vendeurs.
- Évolution temporelle des prix par marque et modèle.

### ⚠️ 2. Détection d’anomalies / Bonnes affaires
- Détection des annonces atypiques via Isolation Forest.
- Mise en évidence des montres à prix inhabituellement bas (potentielles “bonnes affaires”).
- Histogramme des scores d’anomalie pour visualiser la distribution.

### 🧩 3. Segmentation du marché
- Clustering K-Means sur les caractéristiques (prix, diamètre, année, réserve de marche).
- Visualisation 2D des clusters avec réduction de dimension (PCA).
- Profil médian de chaque cluster.

### 📤 4. Export
- tableau interactif des données filtrées.
- Téléchargement des résultats filtrés en CSV ou Excel.

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
├── main.py                 # Application Streamlit principale 
├── utils/                  # Fonctions de nettoyage & utilitaires
├── data/
│   └── raw/montre.db       # Base SQLite contenant les données scrappées
├── requirements.txt        # Liste des dépendances
└── README.md               # Documentation du projet
```

## 📊 Données
Source : Chrono24 (scraping réalisé entre avril 2024 et août 2025).

Variables principales :
- marque, modèle
- prix
- diamètre
- année de production
- mouvement
- matière du boîtier/bracelet/lunette
- état, sexe, pays
- date de récupération

  

Projet réalisé par BOUSSENGUI François, passionné de Data Science et d’horlogerie.
