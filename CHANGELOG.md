# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

## [2026-01-05] - Améliorations majeures

### Ajouté
- `.gitignore` : Ignore les fichiers temporaires et système
- `.dockerignore` : Optimise la construction Docker
- `Makefile` : Commandes simplifiées pour l'installation et le déploiement
- `start.sh` : Script de démarrage rapide pour Unix/Linux/Mac
- `LICENSE` : Licence MIT
- `CONTRIBUTING.md` : Guide de contribution
- `CHANGELOG.md` : Historique des modifications
- Configuration nginx dans `_nginx_/conf.d/default.conf`

### Modifié
- **Dockerfile** : Correction des variables d'environnement (CHEMIN_BDD au lieu de DB_PATH)
- **Dockerfile** : Correction de l'adresse serveur (0.0.0.0 au lieu de 2.2.2.2)
- **docker-compose.yml** : Normalisation de la configuration nginx
- **app/data.py** : Correction du bug qui supprimait la dernière colonne (ligne 16)
- **requirements.txt** : Mise à jour des versions des dépendances
- **README.md** : Documentation complète avec instructions Docker et Makefile
- **README.md** : Correction de la période de collecte des données

### Supprimé
- Fichiers `.DS_Store` et `__pycache__` (maintenant ignorés par Git)

## [Précédent] - Version initiale

### Ajouté
- Application Streamlit pour l'analyse du marché des montres
- 5 onglets d'analyse : Overview, Visualisation, Deals, Clustering, Régression
- Détection d'anomalies avec Isolation Forest
- Segmentation K-Means
- Modélisation du prix avec plusieurs algorithmes
- Support Docker et docker-compose
