#!/bin/bash
# Script de démarrage rapide pour l'application

echo "🚀 Démarrage de l'application Analyse du marché des montres..."

# Vérifier si l'environnement virtuel existe
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv .venv
    echo "✅ Environnement virtuel créé"
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source .venv/bin/activate

# Installer/mettre à jour les dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Lancer l'application
echo "✨ Lancement de l'application..."
echo "📊 L'application sera accessible sur http://localhost:8501"
streamlit run main.py
