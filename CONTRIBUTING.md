# Guide de Contribution

Merci de votre intérêt pour contribuer à ce projet !

## Comment contribuer

### 1. Fork et Clone

```bash
git clone https://github.com/votre-username/analyse_marche_montres.git
cd analyse_marche_montres
```

### 2. Créer une branche

```bash
git checkout -b feature/ma-nouvelle-fonctionnalite
```

### 3. Installer l'environnement de développement

```bash
make install
# ou
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Faire vos modifications

- Suivez le style de code existant
- Testez vos modifications localement
- Assurez-vous que l'application fonctionne correctement

### 5. Commiter vos changements

```bash
git add .
git commit -m "feat: description de votre fonctionnalité"
```

Utilisez les préfixes de commit conventionnels :
- `feat:` pour une nouvelle fonctionnalité
- `fix:` pour une correction de bug
- `docs:` pour la documentation
- `style:` pour le formatage du code
- `refactor:` pour la refactorisation
- `test:` pour les tests
- `chore:` pour les tâches de maintenance

### 6. Pousser et créer une Pull Request

```bash
git push origin feature/ma-nouvelle-fonctionnalite
```

Puis créez une Pull Request sur GitHub.

## Standards de code

- Code Python : suivez PEP 8
- Nommage : variables en snake_case, fonctions explicites
- Commentaires : en français pour correspondre au reste du projet
- Types hints : recommandés pour les nouvelles fonctions

## Tests

Avant de soumettre une PR, vérifiez que :
- L'application démarre sans erreur
- Tous les onglets s'affichent correctement
- Les visualisations se chargent
- Il n'y a pas d'erreurs dans la console

## Questions ?

N'hésitez pas à ouvrir une issue si vous avez des questions !
