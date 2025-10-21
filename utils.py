#---------------------------------------------------------------------
# Libraire pour le nettoyage de la base de données
#---------------------------------------------------------------------
 
import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import defaultdict
from typing import Union

 

#---------------------------------------------------------------------
# Libraire pour la partie graphique
#---------------------------------------------------------------------

from rich.table import Table
from rich.console import Console
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


    
#---------------------------------------------------------------------
#Nettoyage et normalisation de la base
#---------------------------------------------------------------------

class Nettoyage:
    def __init__(self, df):
        self.df = df

    # Nettoyage préliminaire 
    
    def nettoyage_colonnes(self, remplacer_nan=np.nan) -> pd.DataFrame:
        
        # Suppression de la colonne fonctions :
        if 'fonctions' in self.df.columns:
            self.df.drop(columns=['fonctions'], inplace=True)
       
        for col in self.df.columns:
            # Supprimer les crochets et apostrophes dans les chaînes de caractères
            self.df[col] = self.df[col].apply(lambda x: re.sub(r"[\[\]'\"()]", '', str(x)) if isinstance(x, str) else x)
            
            # Supprimer les espaces multiples
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)
            
            # Convertit les colonnes en majuscules : 
            self.df[col] = self.df[col].str.upper()
            
            # Remplacer les valeurs NaN par la valeur spécifiée par défaut
            self.df[col] = self.df[col].fillna(remplacer_nan)
               

            
        # Supprimer les espaces en début et fin de chaîne si c'est une colonne de type chaîne
        if self.df[col].dtype == 'object':
            self.df[col] = self.df[col].str.strip()
        
        # Supprimer les doublons
        self.df.drop_duplicates(inplace = True)
                        

        # Remplacer les chaînes vides et 'None' par NaN
        self.df.replace('', np.nan, inplace=True)
        self.df.replace('None', np.nan, inplace=True)
        
        # Suppression des lignes pour lesquelles on ne connait pas la marque et le modèle
        self.df = self.df.dropna(subset=['marque','modele'])
        
        return self.df
    
    
    
    
    def remplissage(self, variable):
        
        """
        Renseigne l'information manquante pour une colonne donnée, si une ligne possède la marque, le modèle
        et un mouvement similaire. 

        Returns:
            colonne (str): Colonne pour laquelle on doit renseigner les valeurs manquantes. 
        """
        # Grouper par 'marque', 'modele', 'mouvement' pour trouver les valeurs similaires
        groupes_similaires = self.df.groupby(['marque', 'modele', 'mouvement'])[variable]

        # Remplir les valeurs manquantes avec la première valeur non manquante des groupes similaires
        self.df[variable] = self.df[variable].fillna(groupes_similaires.transform('first'))

        return self.df
    
    
    
    
    # Un peu de preprocessing sur les variables
    
    def remplissage_mouvement(self):
        """
        Renseigne la colonne mouvement. 

        Returns:
            pd.DataFrame
        """
        # Remplacer les listes vides par des NaN (si applicable, sinon cette étape peut être ignorée)
        self.df['mouvement'] = self.df['mouvement'].apply(lambda x: np.nan if isinstance(x, list) and not x else x)

        # Grouper par 'marque' et 'modele' pour trouver les valeurs similaires
        groupes_similaires = self.df.groupby(['marque', 'modele'])['mouvement']

        # Remplir les valeurs manquantes avec la première valeur non manquante des groupes similaires
        self.df['mouvement'] = self.df['mouvement'].fillna(groupes_similaires.transform('first'))

        return self.df
    
    
    
    
    def remplissage_reserve_marche(self):
        """
        Renseigne la colonne 'reserve de marche'.

        Returns:
            pd.DataFrame
        """
        
        # Remplir 'Pas_de_reserve' pour les lignes où 'rouage' commence par 'Quar' ou 'ETA' et où 'reserve de marche' est NaN ou vide
        masque_quartz_eta = (
            (self.df['reserve_de_marche'].isna() | (self.df['reserve_de_marche'] == '')) &  # Si 'variable' est manquant ou vide
            self.df['rouage'].apply(lambda x: isinstance(x, str) and (x.startswith('Quar') or x.startswith('ETA')))  # Vérifier 'rouage'
        )
        self.df.loc[masque_quartz_eta, 'reserve_de_marche'] = 'Pas_de_reserve'

        # Remplir 'Pas_de_reserve' pour les lignes où 'mouvement' est 'Quartz' et où 'variable' est NaN
        masque_quartz_mouvement = (self.df['reserve_de_marche'].isna() | (self.df['reserve_de_marche'] == '')) & (self.df['mouvement'] == 'Quartz')
        self.df.loc[masque_quartz_mouvement, 'reserve_de_marche'] = 'Pas_de_reserve'

        return self.df
    
    

    def comptage_fonctions(self, fonction_string):
        """
        Compte le nombre de complications/fonctions dans une chaîne donnée.

        Args:
            fonction_string (str): La chaîne à analyser.

        Returns:
            int ou str: Le nombre de fonctions trouvées, ou 'Non_renseignée' si aucune information.
        """
        if not isinstance(fonction_string, str) or pd.isna(fonction_string):
            return 0
    
        # Liste des mots-clés à rechercher pour les fonctions
        mots_clefs = ['FONCTIONS', 'AUTRES']
        
        for mots in mots_clefs:
            if mots in fonction_string:
                fonctions_part = fonction_string.split(mots)[-1]
                fonctions_list = [func.strip() for func in fonctions_part.split(',') if func.strip()]
                return len(fonctions_list)
        
        return 'Non_renseignée'


    def compteur_complications(self, column_name):
        """
        Ajoute une colonne au DataFrame contenant le nombre de fonctions pour chaque entrée.

        Args:
            column_name (str): Le nom de la colonne à traiter dans le DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame mis à jour avec une nouvelle colonne de comptage.
        """
        self.df[f'comptage_{column_name}'] = self.df[column_name].apply(self.comptage_fonctions)
        return self.df
        
    

    def suppression_colonnes(self):
        """
        Fonctions pour supprimer les colonnes inutiles
        
        Args:
            df (pd.DataFarme): DataFrame contenant les colonnes à traiter. 
            liste_colonnes (list): Liste des colonnes à traiter.
        
        Returns: 
            pd.DataFrame : DataFrame modifié.  
        """
        colonnes_a_supp = ['rouage', 'annee_prod']
        self.df = self.df.drop(columns=colonnes_a_supp)
        
        return self.df
    
    
    # Transformation des colonnes dans un format adéquat :
    
    
    def mise_en_forme(self):
        # Marque 
        marque = [i.replace(', ', '-').replace('.','') for i in self.df['marque']]
        self.df['marque'] = marque
        
        # Modèle
        modele = [i.replace(',','').replace(' ','-') for i in self.df['modele']]
        self.df['modele'] = modele

        # Mouvement 
        
        mapping = {"FOND, TRANSPARENT,, INDICATION, DE, LA, RÉSERVE, DE, MARCHE,, ÉTAT, DORIGINE/PIÈCES, ORIGINALES,, COUCHE, PVD/DLC" : "AUTOMATIQUE",
           
        "28000, A/H": "AUTOMATIQUE",
            "REMONTAGE, AUTOMATIQUE": "AUTOMATIQUE",
            "REMONTAGE, MANUEL" : "AUTOMATIQUE",
            "21600, A/H" : "AUTOMATIQUE",
            "REMONTAGE AUTOMATIQUE" : "AUTOMATIQUE",
            "MONTRE, CONNECTÉE" : "BATTERIE",
            "SQUELETTE" : "AUTOMATIQUE",
            "OSCILLATOIRE, 28800, A/H" : "AUTOMATIQUE", 
            "OSCILLATOIRE, 4, HZ" : "AUTOMATIQUE",
            "OSCILLATOIRE, 28800, HZ": "AUTOMATIQUE",
            "OSCILLATOIRE, 28, A/H" : "AUTOMATIQUE",
            "REMONTAGE MANUEL": "MANUEL"
            } 
           
           
        self.df['mouvement'] = self.df['mouvement'].replace(mapping)
        
        # Sexe
        
        mapping_sexe = {"HOMME/UNISEXE":"HOMME",
           "MONTRE HOMME/UNISEXE":"HOMME",
           "MONTRE, FEMME":"FEMME",
           "MONTRE, HOMME/UNISEXE":"HOMME"  
           }

        self.df['sexe'] = self.df['sexe'].replace(mapping_sexe)
        
        # Matière verre et boucle
        mapping_verre = {
        'VERRE SAPHIR' : 'SAPHIR',
        'VERRE, MINÉRAL' : 'MINÉRAL',
        'MATIÈRE, PLASTIQUE':'PLASTIQUE',
        'VERRE, SAPHIR':'SAPHIR'
            }
        
        mapping_boucle = {
            "PLIS,_COUVERT" : "PLIS"
        }
    
        self.df['matiere_verre'] = self.df['matiere_verre'].replace(mapping_verre)
        self.df['boucle'] = self.df['boucle'].str.replace(', ','_')
        self.df['boucle'] = self.df['boucle'].replace(mapping_boucle)
        
        # ville
        
        pays = [i.split(',')[0].strip() if isinstance(i, str) else 'INCONNU' for i in self.df['ville']]
        self.df = self.df.drop(columns=['ville'])
        self.df['pays'] = pays
        
        mapping = {
            "GRANDE-BRETAGNE": 'ROYAUME-UNI',
            'AFRIQUE': 'AFRIQUE_DU_SUD',
            'RÉPUBLIQUE' : 'RÉPUBLIQUE_TCHEQUE',
            'HONG' : 'HONG_KONG',
            'VIÊT' : 'VIETNAM',
            'PORTO' : 'PORTUGAL',
            'E.A.U.' : 'EMIRAT_ARABE_UNIS',
            'SRI': 'SRI_LANKA',
            'ARABIE' : 'ARABIE_SAOUDITE' 
        }

        self.df['pays'] = self.df['pays'].replace(mapping)
        
        return self.df
        
     
    def extraire_matiere(self, chaine):
        matières = [ "oracier", "acier", "cuir", "textile", "titane", "caoutchouc", "bronze",
            "silicone", "vache", "dautruche", "bronze","plastique", "platine", "céramique","or",
            "aluminium", "argent", "requin", "caoutchouc", "plastique", "silicone", 
            "céramique", "satin", "blanc", "requin", "agenté", "rose", "jaune",
            "rouge", "tungstène", "palladium","lisse","carbone", "plaquée"]

        if isinstance(chaine, str): # Vérifier si la chaîne est une chaîne de caractères
            chaine = chaine.replace('/', "").lower()
            for matiere in matières:
                if matiere in chaine:
                    return matiere.upper()
        return np.nan
    
    
    def matiere(self):
        self.df['matiere_bracelet'] = self.df['matiere_bracelet'].apply(self.extraire_matiere)
        self.df['matiere_lunette'] = self.df['matiere_lunette'].apply(self.extraire_matiere)
        self.df['matiere_boucle'] = self.df['matiere_boucle'].apply(self.extraire_matiere)
        return self.df
    
    def mapping_matiere(self):
        mapping_matiere = {"ORACIER": "OR_ACIER",
           "DAUTRUCHE": "CUIR_AUTRUCHE",
           "BLANC":"OR_BLANC",
           "ROSE":"OR_ROSE",
           "VACHE":"CUIRE_DE_VACHE",
           "JAUNE":"OR_JAUNE",
           "ROUGE":"OR_ROUGE"    
        }       
        self.df['matiere_bracelet'] = self.df['matiere_bracelet'].replace(mapping_matiere)
        self.df['matiere_lunette'] = self.df['matiere_lunette'].replace(mapping_matiere)
        self.df['matiere_boucle'] = self.df['matiere_boucle'].replace(mapping_matiere)
        return self.df
     
    def nettoyage_matiere_boitier(self, chaine):
        if isinstance(chaine, str):  # Vérifier si la variable est une chaîne
            # Remplacer les barres obliques par des virgules avec espaces
            chaine = chaine.replace("/", ", ")
            # Nettoyer les virgules en trop
            chaine = re.sub(r'\s*,\s*', ', ', chaine)
            # Supprimer les espaces au début et à la fin
            chaine = chaine.strip()
            # Remplacer les virgules et espaces par des underscores
            chaine = chaine.replace(', ', '_')
            return chaine
        else:
            return np.nan  # Retourner 'INCONNUE' si ce n'est pas une chaîne
    
    
    def nettoyer_matiere_boitier(self):
        self.df['matiere_boitier'] = self.df['matiere_boitier'].apply(self.nettoyage_matiere_boitier)
        return self.df
    
        
    def regrouper_état(self, chaine):
        catégories_état = {
            "neuf": "Neuf", "jamais porté": "Neuf", "usure nulle": "Neuf", "aucune trace d'usure": "Neuf",
            "bon": "Bon", "légères traces d'usure": "Bon",
            "traces d'usure visibles": "Satisfaisant", "modéré": "Satisfaisant", "satisfaisant": "Satisfaisant",
            "fortement usagé": "Usé", "traces d'usure importantes": "Usé",
            "défectueux": "Défectueux", "incomplet": "Défectueux", "pas fonctionnelle": "Défectueux"
        }

        if not isinstance(chaine, str):
            return "État non spécifié"
        
        chaine = chaine.lower()

        # Cherche la première correspondance
        return next((catégorie for mot_clé, catégorie in catégories_état.items() if mot_clé in chaine), "État non spécifié")  # Retourner "État non spécifié" si ce n'est pas une chaîne de caractères
    
    
    
    def regroupement_etat_montres(self):
        self.df['etat'] = self.df['etat'].apply(self.regrouper_état) 
        return self.df

   
    # Fonction pour extraire les éléments avant le symbole '€'
    def extraire_elements_avant_euro(self, chaine):
        """
        Extrait les deux éléments avant le symbole '€' dans une chaîne.

        Args:
            chaine (str): La chaîne de texte contenant les informations de prix.

        Returns:
            list: Liste contenant jusqu'à deux éléments avant le symbole '€', ou une liste vide si non trouvé.
        """
        if isinstance(chaine, str):
            # Séparer par virgules et espaces, enlever les espaces inutiles
            elements = re.split(r'[,\s]+', chaine.strip())
            
            # Chercher l'index de '€' et extraire les deux éléments précédents
            index_fin = next((i for i, el in enumerate(elements) if '€' in el), None)
            if index_fin:
                return elements[max(0, index_fin - 2):index_fin]  # Retourne jusqu'à deux éléments avant '€'
        return []  # Retourne une liste vide si chaîne non valide ou pas de symbole '€'
    
    # Applique l'extraction des éléments avant '€' sur la colonne 'prix'
    def extraction_elements_avant_euro(self):
        """
        Applique l'extraction des éléments avant '€' sur la colonne 'prix' et met à jour le DataFrame.

        Returns:
            pd.DataFrame: DataFrame mis à jour avec les éléments extraits dans la colonne 'prix'.
        """
        self.df['prix'] = self.df['prix'].apply(self.extraire_elements_avant_euro)
        return self.df
    
    # Fonction pour nettoyer et convertir les valeurs de la colonne spécifiée en nombres
    def nettoyage_prix(self, colonne):
        """
        Nettoie et convertit les éléments de la colonne spécifiée en nombres.

        Args:
            colonne (str): Le nom de la colonne contenant les valeurs à nettoyer.

        Returns:
            pd.DataFrame: DataFrame avec la colonne spécifiée convertie en nombres.
        """
        def convertir(val):
            if isinstance(val, list):
                # Joindre les parties numériques en une seule chaîne
                nombre_str = ''.join(val)
                
                # Nettoyer et convertir en nombre si possible
                if nombre_str.strip():
                    try:
                        # Conversion en int si possible, sinon float
                        return int(nombre_str) if '.' not in nombre_str else float(nombre_str)
                    except ValueError:
                        return np.nan  # En cas d'erreur de conversion
                else:
                    return np.nan  # Si la chaîne est vide
            return np.nan  # Gérer les valeurs non liste
        
        # Appliquer la conversion sur chaque élément de la colonne
        self.df[colonne] = self.df[colonne].apply(convertir)
        return self.df
    
   
            
    
    def extraction_int(self, chaine):
        """
        Extrait le premier chiffre entier trouvé dans une chaîne de caractères.

        Args:
            chaine (str): La chaîne à analyser.

        Returns:
            int ou NaN: Le premier nombre entier trouvé ou NaN si aucun nombre n'est trouvé.
        """
        if pd.isna(chaine):
            return np.nan
        match = re.search(r'\d+', chaine)
        return int(match.group()) if match else np.nan

    def extraction_integer(self):
        """
        Extrait le premier nombre entier de la colonne et l'ajoute sous forme d'entier.
        
        Returns:
            pd.DataFrame: DataFrame mis à jour avec la colonne modifiée.
        """
        # Utiliser directement apply avec extraire_chiffre pour plus de clarté
        self.df['reserve_de_marche'] = self.df['reserve_de_marche'].apply(self.extraction_int)
        self.df['etencheite'] = self.df['etencheite'].apply(self.extraction_int)
        self.df['diametre'] = self.df['diametre'].apply(self.extraction_int)
        
        return self.df
#---------------------------------------------------------------------
# Feature enginering
#---------------------------------------------------------------------

    def matiere_boitier_cat(self):
        # Mapping vers familles
        mapping_materiau = {
            # Métaux courants
            "ACIER": "commun",
            "TITANE": "commun",
            "ALUMINIUM": "commun",
            "BRONZE": "commun",
            "ARGENT": "commun",

            # Alliages or/acier
            "OR_ACIER": "mixte",

            # Or & dérivés
            "OR_JAUNE": "précieux",
            "OR JAUNE": "précieux",
            "OR_ROSE": "précieux",
            "OR ROSE": "précieux",
            "OR_BLANC": "précieux",
            "OR BLANC": "précieux",
            "OR_ROUGE": "précieux",
            "PLAQUÉE_OR": "précieux",

            # Autres métaux précieux
            "PLATINE": "précieux",
            "PALLADIUM": "précieux",

            # Matériaux high-tech
            "CÉRAMIQUE": "moderne",
            "CARBONE": "moderne",
            "VERRE_SAPHIR": "moderne",
            "PLASTIQUE": "moderne",
            "TANTALE": "moderne",
        }

        # Application du mapping
        self.df["matiere_boitier_cat"] = self.df["matiere_boitier"].map(mapping_materiau)
        return self.df

    def matiere_bracelet_cat(self):
        # Mapping vers familles
        mapping_bracelet = {
            # Métaux courants
            "ACIER": "commun",
            "TITANE": "commun",
            "ALUMINIUM": "commun",
            "BRONZE": "commun",
            "ARGENT": "commun",

            # Alliages or/acier
            "OR_ACIER": "mixte",

            # Or & métaux précieux
            "OR_JAUNE": "précieux",
            "OR_ROSE": "précieux",
            "OR_BLANC": "précieux",
            "OR_ROUGE": "précieux",
            "OR": "précieux",
            "PLATINE": "précieux",

            # Matériaux modernes
            "CÉRAMIQUE": "moderne",
            "CAOUTCHOUC": "moderne",
            "SILICONE": "moderne",
            "PLASTIQUE": "moderne",

            # Textiles
            "TEXTILE": "textile",
            "SATIN": "textile",

            # Cuirs / peaux
            "CUIR": "cuir",
            "CUIRE_DE_VACHE": "cuir",
            "CUIR_AUTRUCHE": "cuir_exotique",
            "REQUIN": "cuir_exotique",
        }

        # Application du mapping
        self.df["matiere_bracelet_cat"] = self.df["matiere_bracelet"].map(mapping_bracelet)
        return self.df

    def matiere_lunette_cat(self):
        # Mapping vers familles
        mapping_lunette = {
            # Métaux communs
            "ACIER": "commun",
            "TITANE": "commun",
            "ALUMINIUM": "commun",
            "BRONZE": "commun",
            "ARGENT": "commun",

            # Alliages or/acier
            "OR_ACIER": "mixte",

            # Or & métaux précieux
            "OR_JAUNE": "précieux",
            "OR_ROSE": "précieux",
            "OR_BLANC": "précieux",
            "OR_ROUGE": "précieux",
            "OR": "précieux",
            "PLATINE": "précieux",
            "PALLADIUM": "précieux",

            # Matériaux modernes / techniques
            "CÉRAMIQUE": "moderne",
            "CARBONE": "moderne",
            "PLASTIQUE": "moderne",
            "CAOUTCHOUC": "moderne",
            "TUNGSTÈNE": "moderne",
        }

        # Application du mapping
        self.df["matiere_lunette_cat"] = self.df["matiere_lunette"].map(mapping_lunette)
        return self.df

    def matiere_verre_cat(self):
        # Mapping vers familles
        mapping_verre = {
            # Verres haut de gamme
            "SAPHIR": "saphir",

            # Verres intermédiaires
            "MINÉRAL": "minéral",

            # Verres basiques / plastiques
            "PLASTIQUE": "plastique",
            "PLEXIGLAS": "plastique",
            "VERRE": "plastique",  # terme générique, souvent utilisé pour désigner verre simple
        }

        # Application du mapping
        self.df["matiere_verre_cat"] = self.df["matiere_verre"].map(mapping_verre)
        return self.df

    def boucle_cat(self):
        # Mapping vers familles
        mapping_fermoir = {
            # Types de fermoirs / boucles
            "PLIS": "deployante",
            "DOUBLE_PLIS": "deployante",
            "DOUBLE PLIS": "deployante",
            "PLIS_COUVERT": "deployante",
            "BOUCLE_ARDILLON": "ardillon",
            "BOUCLE ARDILLON": "ardillon",
            "FERMOIR_BIJOUX": "ardillon",
            "PAS_DE_BOUCLE": "ardillon",
        }
        # Application du mapping
        self.df["boucle_cat"] = self.df["boucle"].map(mapping_fermoir)
        return self.df

    def type_boucle_cat(self):
        # Mapping vers familles
        mapping_type_boucle = {
            # Métaux communs
            "ACIER": "commun",
            "TITANE": "commun",
            "ALUMINIUM": "commun",
            "BRONZE": "commun",
            "ARGENT": "commun",

            # Alliages or/acier
            "OR_ACIER": "mixte",

            # Or & métaux précieux
            "OR": "précieux",
            "PLATINE": "précieux",
            "PALLADIUM": "précieux",

            # Matériaux modernes
            "CÉRAMIQUE": "moderne",
            "CARBONE": "moderne",
            "PLASTIQUE": "moderne",
            "CAOUTCHOUC": "moderne",
            "TUNGSTÈNE": "moderne",
        }
        # Application du mapping
        self.df["matiere_boucle_cat"] = self.df["matiere_boucle"].map(mapping_type_boucle)
        return self.df
#---------------------------------------------------------------------
# Graphiques
#---------------------------------------------------------------------
    
class graphique:
    def __init__(self, df):
        self.df = df
     
    
    def effectif_pays(self):
        effectif = self.df.pays.value_counts().reset_index(name="effectif")
        effectif.columns = ['pays', 'nbre_montre']
        effectif['pourcentage'] = (effectif['nbre_montre'] / effectif['nbre_montre'].sum())*100
        effectif['pourcentage'] = effectif['pourcentage'].round(2)
        return effectif
        
    def fig_pays(self, effectif):

        fig = px.choropleth(effectif, 
                            locations="pays",  # Nom de la colonne contenant les pays
                            locationmode="country names",  # Mode pour faire correspondre les noms des pays
                            color="nbre_montre",  # Colonne des effectifs
                            color_continuous_scale="Plasma",  # Palette de couleurs
                            hover_name="pays",
                            title="Provenance des offres")

        return fig.show()
    
    def tab_pays(self, effectif):
        # Créer une console pour afficher le tableau
        console = Console()

        # Initialiser la table
        table = Table(title="Provenance des montres")

        # Ajouter les colonnes
        table.add_column("Pays", justify="center", style="cyan", no_wrap=True)
        table.add_column("Nombre de montres", justify="right", style="magenta")
        table.add_column("%", justify="right", style="green")

        # Ajouter les lignes du tableau à partir des données
        for _, row in effectif.iterrows():
            table.add_row(
                str(row["pays"]),
                str(row["nbre_montre"]),
                f"{row['pourcentage']:.2f} %"
            )

        # Afficher la table
        console.print(table)
    
    def stat_pays(self, effectif):
        stats_localisation = self.df.groupby('pays')['prix'].agg(['mean', 'min', 'max']).reset_index()
        stats_localisation.columns = ['pays', 'prix_moyen', 'prix_min', 'prix_max']
        stats_localisation_merge = pd.merge(stats_localisation, effectif, how='left')
        stats_localisation_merge['prix_moyen'] = stats_localisation_merge['prix_moyen'].round(2)
        stats_localisation_sorted = stats_localisation_merge.sort_values(by="prix_moyen", ascending=False).reset_index(drop=True)
        return stats_localisation_sorted
    
    def tab_pays_2(self, stats_localisation_sorted):
        # Créer une console pour afficher le tableau
        console = Console()

        # Initialiser la table
        table = Table(title="Statistiques par pays")

        # Ajouter les colonnes
        table.add_column("Pays", justify="center", style="cyan", no_wrap=True)
        table.add_column("Nombre de montres", justify="right", style="magenta")
        table.add_column("Prix moyen", justify="right", style="green")
        table.add_column("Prix min", justify="right", style="yellow")
        table.add_column("Prix max", justify="right", style="red")

        # Ajouter les lignes du tableau à partir des données
        for _, row in stats_localisation_sorted.iterrows():
            table.add_row(
                str(row["pays"]),
                str(row["nbre_montre"]),
                f"{row['prix_moyen']:.2f}",
                f"{row['prix_min']:.2f}",
                f"{row['prix_max']:.2f}"
            )

        # Afficher la table
        console.print(table)
    
    def tableau(self, colonne):
        # Calcul des statistiques
        stat = self.df.groupby(colonne)['prix'].agg(['mean', 'min', 'max']).reset_index()
        stat.columns = [colonne, 'prix_moyen', 'prix_min', 'prix_max']
        stat['prix_moyen'] = stat['prix_moyen'].round(2)
        stat['prix_min'] = stat['prix_min'].round(2)
        stat['prix_max'] = stat['prix_min'].round(2)
        stat = stat.sort_values(by="prix_moyen", ascending=False).reset_index(drop=True)

        # Créer une console pour afficher le tableau
        console = Console()

        # Initialiser la table
        table = Table(title=f"Statistiques par {colonne.capitalize()}")

        # Ajouter les colonnes
        table.add_column(colonne.capitalize(), justify="center", style="cyan", no_wrap=True)
        table.add_column("Prix moyen", justify="right", style="green")
        table.add_column("Prix min", justify="right", style="yellow")
        table.add_column("Prix max", justify="right", style="red")

        # Ajouter les lignes du tableau à partir des données
        for _, row in stat.iterrows():
            table.add_row(
                str(row[colonne]),
                f"{row['prix_moyen']:.2f}",
                f"{row['prix_min']:.2f}",
                f"{row['prix_max']:.2f}"
            )

        # Afficher la table
        console.print(table)
    
    
    

    def boxplot(self, colonne):
        """
        Crée un boxplot interactif avec Matplotlib et Seaborn pour visualiser la distribution des prix selon une colonne donnée.
        
        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            colonne (str): Le nom de la colonne pour la segmentation (par ex. 'Marque').
        
        Returns:
            None: Affiche directement le boxplot.
        """
        # Créer une figure et des axes
        plt.figure(figsize=(10, 8))
        
        # Créer le boxplot avec Seaborn
        sns.boxplot(
            data=self.df,
            x='prix_log',
            y=colonne,
            palette="Set2"  # Palette de couleurs
        )
        
        # Ajouter des titres et labels
        plt.title(f'Distribution des prix selon {colonne}', fontsize=14, fontweight='bold')
        plt.xlabel('Prix des montres (en log)', fontsize=12)
        plt.ylabel(colonne, fontsize=12)
        
        # Ajuster l'espacement
        plt.tight_layout()
        
        # Afficher le graphique
        plt.show()
        
 

    def barres(self, colonne):
        """
        Crée un barplot avec Matplotlib et Seaborn pour visualiser le prix moyen par catégorie.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            colonne (str): Le nom de la colonne pour la segmentation (par ex. 'marque').

        Returns:
            None: Affiche directement le barplot.
        """
        # Calculer le prix moyen par catégorie
        df_grouped = self.df.groupby(colonne)['prix'].mean().reset_index()
        df_grouped = df_grouped.sort_values(by='prix', ascending=True)

        # Créer une figure et des axes
        plt.figure(figsize=(10, 8))

        # Créer le barplot avec Seaborn
        sns.barplot(
            data=df_grouped,
            x='prix',
            y=colonne,
            palette="viridis"  # Palette de couleurs
        )

        # Ajouter des titres et labels
        plt.title(f'Prix moyen par {colonne}', fontsize=14, fontweight='bold')
        plt.xlabel('Prix moyen (en euros)', fontsize=12)
        plt.ylabel(f'Types de {colonne}', fontsize=12)

        # Ajuster l'espacement
        plt.tight_layout()

        # Afficher le graphique
        plt.show()


#---------------------------------------------------------------------
# Calculs
#---------------------------------------------------------------------
    

def is_dummy(x: Union[pd.Series, np.ndarray]) -> bool:
    """Description. Checks if a numpy array is a dummy variable."""

    if isinstance(x, pd.Series):
        x = x.values

    x = x.astype(float)
    x = x[~np.isnan(x)]

    return np.all(np.isin(x, [0., 1.]))

def is_numeric(x: pd.Series) -> bool:
    """Description. Checks if a numpy array is numeric."""

    if x.dtype == "int64" and not is_dummy(x):  
        return True
    
    return False

#---------------------------------------------------------------------
# Fonction pour la partie ml
#---------------------------------------------------------------------

def categoriser_reserve_marche(df, colonne='reserve_de_marche'):
    """
    Découpe la variable 'réserve de marche' en catégories selon des tranches définies,
    après avoir filtré les valeurs entre 24h et 168h.

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne 'reserve_marche' avec les valeurs de réserve de marche.
        colonne (str): Le nom de la colonne contenant les valeurs de réserve de marche.

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne 'reserve_marche_categorisée' contenant les catégories de réserve de marche.
    """
    
    # Filtrer les valeurs entre 24h et 168h
    df_filtre = df[(df[colonne] >= 24) & (df[colonne] <= 168)]
    
    # Définition des catégories selon les tranches de valeurs
    def categoriser(valeur):
        if pd.isna(valeur):
            return 'Non spécifié'
        elif valeur < 40:
            return 'Réserve de marche standard (24h - 40h)'
        elif 40 <= valeur <= 72:
            return 'Réserve de marche haute (40h - 72h)'
        elif 72 < valeur <= 120:
            return 'Réserve de marche très haute (72h - 120h)'
        elif 120 < valeur <= 168:
            return 'Réserve de marche ultra haute (120h - 168h)'
        else:
            return 'Non spécifié'  # Dans le cas où la valeur est en dehors de la plage définie
    
    # Appliquer la fonction de catégorisation à la colonne
    df_filtre['reserve_marche_cat'] = df_filtre[colonne].apply(categoriser)
    df_filtre = df_filtre.drop(columns='reserve_de_marche')
    return df_filtre

def categoriser_diametre(df, colonne='diametre'):
    """
    Découpe la variable 'diametre' en catégories selon des tranches de valeurs définies.

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne 'diametre' avec les valeurs de diamètre des montres.
        colonne (str): Le nom de la colonne contenant les valeurs de diamètre.

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne 'diametre_categorisé' contenant les catégories de diamètre.
    """
    
    # Filtrer les valeurs dans la plage acceptable de diamètres (28 mm à 50 mm)
    df_filtre = df[(df[colonne] >= 28) & (df[colonne] <= 50)]
    
    # Définition des catégories selon les tranches de valeurs
    def categoriser(valeur):
        if pd.isna(valeur):
            return 'Non spécifié'
        elif valeur < 34:
            return 'Petite montre (28mm - 34mm)'
        elif 34 <= valeur <= 40:
            return 'Montre de taille moyenne (34mm - 40mm)'
        elif 40 < valeur <= 44:
            return 'Grande montre (40mm - 44mm)'
        elif 44 < valeur <= 50:
            return 'Très grande montre (44mm - 50mm)'
        else:
            return 'Non spécifié'  # Dans le cas où la valeur est en dehors de la plage définie
    
    # Appliquer la fonction de catégorisation à la colonne
    df_filtre['diametre_cat'] = df_filtre[colonne].apply(categoriser)
    df_filtre = df_filtre.drop(columns='diametre')
    return df_filtre

def categoriser_etancheite(df, colonne='etencheite'):
    """
    Découpe la variable 'etancheite' en catégories selon des tranches de valeurs définies.

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne 'etancheite' avec les valeurs d'étanchéité des montres.
        colonne (str): Le nom de la colonne contenant les valeurs d'étanchéité.

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne 'etancheite_categorisé' contenant les catégories d'étanchéité.
    """
    
    # Filtrer les valeurs dans la plage acceptable d'étanchéité (1 à 100 ATM)
    df_filtre = df[(df[colonne] >= 1) & (df[colonne] <= 100)]
    
    # Définition des catégories selon les tranches de valeurs
    def categoriser(valeur):
        if pd.isna(valeur):
            return 'Non spécifié'
        elif valeur < 5:
            return 'Faible étanchéité (1-3 ATM)'
        elif 5 <= valeur < 10:
            return 'Résistance à l\'eau modérée (5-10 ATM)'
        elif 10 <= valeur < 20:
            return 'Plongée légère (10-20 ATM)'
        elif 20 <= valeur <= 50:
            return 'Plongée professionnelle (20-50 ATM)'
        else:
            return 'Plongée extrême (>50 ATM)'
    
    # Appliquer la fonction de catégorisation à la colonne
    df_filtre['etancheite_cat'] = df_filtre[colonne].apply(categoriser)
    df_filtre = df_filtre.drop(columns=['etencheite', 'comptage_fonctions'])
    
    return df_filtre