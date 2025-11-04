import os
from pathlib import Path

CHEMIN_BDD = os.getenv(
    "CHEMIN_BDD",
    str(Path(__file__).resolve().parents[1] / "data" / "propre.xlsx")
)

# Paramètres Streamlit
TITRE_PAGE = "Montres – Statistiques descriptives (Chrono24)"
LAYOUT = "wide"