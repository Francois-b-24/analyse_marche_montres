.PHONY: help install run docker-build docker-up docker-down clean

help:
	@echo "Commandes disponibles :"
	@echo "  make install       - Installer les dépendances"
	@echo "  make run           - Lancer l'application en local"
	@echo "  make docker-build  - Construire l'image Docker"
	@echo "  make docker-up     - Démarrer les containers"
	@echo "  make docker-down   - Arrêter les containers"
	@echo "  make clean         - Nettoyer les fichiers temporaires"

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/streamlit run main.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
