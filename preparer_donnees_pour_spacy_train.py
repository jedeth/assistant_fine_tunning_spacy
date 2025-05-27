# preparer_donnees_pour_spacy_train.py
import json
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split # Pour diviser les données
import random # Non utilisé directement ici, mais bon pour de futures divisions plus complexes

# Configuration des chemins
CHEMIN_DONNEES_JSON_ENTREE = "donnees_entrainement_longues.json"
CHEMIN_TRAIN_SPACY = "./train.spacy"  # Sortie pour les données d'entraînement
CHEMIN_DEV_SPACY = "./dev.spacy"      # Sortie pour les données de développement
PROPORTION_DEV_SET = 0.20             # 20% des données iront dans le dev set
MODELE_SPACY_POUR_TOKENISATION = "fr_core_news_md" # Modèle utilisé pour sa tokenisation et son vocabulaire

def creer_fichiers_spacy_binaire(chemin_json, chemin_train, chemin_dev, test_size=0.2):
    """
    Charge les données depuis un JSON, les convertit en objets Doc de SpaCy,
    les divise en ensembles d'entraînement et de développement,
    et les sauvegarde au format binaire .spacy.
    """
    try:
        with open(chemin_json, 'r', encoding='utf-8') as f:
            donnees_json_chargees = json.load(f)
        print(f"{len(donnees_json_chargees)} exemples chargés depuis '{chemin_json}'.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON d'entrée '{chemin_json}' est introuvable.")
        return False
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier JSON '{chemin_json}' n'est pas un JSON valide.")
        return False
    except Exception as e:
        print(f"Une erreur inattendue s'est produite lors de la lecture du JSON : {e}")
        return False

    if not donnees_json_chargees:
        print("Aucune donnée à traiter.")
        return False

    # Charger le pipeline nlp (même s'il est vierge ou un modèle de base,
    # c'est pour utiliser son tokeniseur et son vocabulaire)
    try:
        nlp = spacy.load(MODELE_SPACY_POUR_TOKENISATION)
        print(f"Modèle '{MODELE_SPACY_POUR_TOKENISATION}' chargé pour la tokenisation et la création des DocBin.")
    except OSError:
        nlp = spacy.blank("fr") # Solution de repli
        print(f"Avertissement: Modèle '{MODELE_SPACY_POUR_TOKENISATION}' non trouvé. Utilisation d'un pipeline français vierge ('fr').")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SpaCy '{MODELE_SPACY_POUR_TOKENISATION}': {e}")
        return False
        
    # Diviser les données en ensembles d'entraînement et de développement
    if len(donnees_json_chargees) < 2 : # Besoin d'au moins 2 échantillons pour diviser
        print("Pas assez de données pour créer des ensembles d'entraînement et de développement distincts.")
        # Dans ce cas, on pourrait tout mettre en entraînement et ne pas avoir de dev set,
        # ou dupliquer pour avoir un dev set (non idéal).
        # Pour l'instant, on arrête si pas assez de données.
        return False

    try:
        donnees_train, donnees_dev = train_test_split(donnees_json_chargees, test_size=test_size, random_state=42)
    except ValueError as e:
        print(f"Erreur lors de la division des données (test_size={test_size}, nombre d'échantillons={len(donnees_json_chargees)}): {e}")
        print("Assurez-vous d'avoir suffisamment de données pour la division.")
        return False

    print(f"{len(donnees_train)} exemples pour l'entraînement, {len(donnees_dev)} pour le développement.")

    # Créer et sauvegarder le fichier .spacy pour l'entraînement
    db_train = DocBin()
    for texte, annotations in donnees_train:
        doc = nlp.make_doc(texte)
        entites_valides = []
        for debut, fin, label in annotations.get("entities", []):
            span = doc.char_span(debut, fin, label=label, alignment_mode="contract")
            if span is None:
                print(f"Attention (train): Span non créé pour '{texte[debut:fin]}' (indices {debut}-{fin}) dans le texte : '{texte}'. Vérifiez les alignements avec les tokens.")
            else:
                entites_valides.append(span)
        try:
            doc.ents = entites_valides
        except ValueError as e: # Peut arriver si les spans se chevauchent
            print(f"Erreur lors de l'assignation des entités (train) pour le texte '{texte[:50]}...': {e}. Entités candidates: {[(e.text, e.label_) for e in entites_valides]}")
            continue # Passer à l'exemple suivant
        db_train.add(doc)
    db_train.to_disk(chemin_train)
    print(f"Données d'entraînement sauvegardées dans '{chemin_train}'")

    # Créer et sauvegarder le fichier .spacy pour le développement
    db_dev = DocBin()
    for texte, annotations in donnees_dev:
        doc = nlp.make_doc(texte)
        entites_valides = []
        for debut, fin, label in annotations.get("entities", []):
            span = doc.char_span(debut, fin, label=label, alignment_mode="contract")
            if span is None:
                print(f"Attention (dev): Span non créé pour '{texte[debut:fin]}' (indices {debut}-{fin}) dans le texte : '{texte}'. Vérifiez les alignements avec les tokens.")
            else:
                entites_valides.append(span)
        try:
            doc.ents = entites_valides
        except ValueError as e:
            print(f"Erreur lors de l'assignation des entités (dev) pour le texte '{texte[:50]}...': {e}. Entités candidates: {[(e.text, e.label_) for e in entites_valides]}")
            continue # Passer à l'exemple suivant
        db_dev.add(doc)
    db_dev.to_disk(chemin_dev)
    print(f"Données de développement sauvegardées dans '{chemin_dev}'")
    return True

if __name__ == "__main__":
    # Assurez-vous que scikit-learn est installé : pip install scikit-learn
    if creer_fichiers_spacy_binaire(CHEMIN_DONNEES_JSON_ENTREE, CHEMIN_TRAIN_SPACY, CHEMIN_DEV_SPACY, PROPORTION_DEV_SET):
        print("Conversion des données au format .spacy terminée avec succès.")
    else:
        print("La conversion des données a échoué. Veuillez vérifier les erreurs ci-dessus.")