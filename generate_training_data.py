# generate_training_data.py
import json
import os # Ajout de l'import os si vous utilisez des chemins relatifs pour les fichiers

# Chemin vers votre annuaire de noms (un nom complet par ligne)
CHEMIN_ANNUAIRE_TXT = "annuaire_pour_finetuning.txt" 

# Fichier de sortie pour les données générées (format JSON pour inspection)
CHEMIN_DONNEES_GENEREES_JSON = "donnees_entrainement_longues.json"

# Label pour les entités de l'annuaire
LABEL_ENTITE = "PER"

# NOUVELLES PHRASES TEMPLATES PLUS LONGUES (au moins 100 caractères)
PHRASES_TEMPLATES_LONGUES = [
    "Après une analyse approfondie du dossier soumis par {NOM}, le comité a décidé de reconsidérer sa proposition initiale pour ce projet d'envergure.",
    "Il a été officiellement confirmé que {NOM} sera en charge de présenter les résultats finaux de l'étude lors de la conférence internationale prévue à Strasbourg.",
    "Malgré les défis techniques et les contraintes budgétaires, {NOM} a démontré une persévérance et un leadership remarquables, menant son équipe vers une réussite collective.",
    "La contribution significative de {NOM} au développement et à l'optimisation de notre nouvelle plateforme technologique a été reconnue comme essentielle par tous les membres du conseil.",
    "Nous avons eu le privilège d'accueillir {NOM} pour une série d'entretiens approfondis la semaine dernière, et son profil unique correspond parfaitement aux exigences du poste.",
    "Pour toute question administrative ou demande relative à la facturation des services et aux échéances de paiement, veuillez vous adresser directement à {NOM}, notre gestionnaire financier principal.",
    "Le rapport d'activité trimestriel, qui sera bientôt publié, mentionne spécifiquement l'implication déterminante de {NOM} dans l'atteinte des objectifs ambitieux que nous nous étions fixés.",
    "Si vous souhaitez obtenir une copie certifiée conforme des minutes de la dernière assemblée générale du conseil d'administration, {NOM} est la personne désignée à contacter auprès du secrétariat.",
    "L'article scientifique récemment co-publié par {NOM} dans la prestigieuse revue internationale 'Advanced Research Journal' a suscité un vif intérêt et de nombreux débats au sein de la communauté scientifique.",
    "C'est avec une immense fierté et un grand plaisir que la direction annonce aujourd'hui la promotion de {NOM} au poste stratégique de directeur des opérations européennes, une nomination effective immédiatement.",
    "Le témoignage de {NOM} lors du procès a été crucial pour l'issue de l'affaire, apportant des éclaircissements indispensables aux jurés.",
    "Les compétences en gestion de projet de {NOM} ont permis de livrer le produit dans les temps impartis, malgré un cahier des charges particulièrement exigeant.",
    "Une enquête interne sera menée par {NOM} afin de déterminer les causes exactes de l'incident survenu hier dans nos locaux principaux.",
    "Suite à sa brillante allocution, {NOM} a reçu une ovation debout de la part de l'ensemble des participants et des organisateurs de l'événement.",
    "Il est impératif que {NOM} valide toutes les modifications apportées au document avant sa diffusion officielle à l'ensemble des parties prenantes."
]

def charger_noms_annuaire(chemin_fichier):
    """Charge les noms depuis un fichier texte (un nom par ligne)."""
    noms = []
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                nom = ligne.strip()
                if nom:  # Ignorer les lignes vides
                    noms.append(nom)
    except FileNotFoundError:
        print(f"Erreur : Le fichier annuaire '{chemin_fichier}' est introuvable.")
    return noms

def generer_donnees_spacy(liste_noms, phrases_templates, label_entite="PER"):
    """
    Génère des données d'entraînement pour SpaCy NER.
    Chaque nom de la liste est inséré dans chaque template de phrase.
    """
    donnees_spacy_format = []
    for nom_propre in liste_noms:
        for template in phrases_templates:
            # Remplacer le placeholder {NOM} par le nom propre actuel
            phrase_contexte = template.replace("{NOM}", nom_propre)
            
            # Trouver la position de début et de fin du nom propre dans la phrase
            # Cette méthode simple fonctionne si le nom propre n'est pas modifié par le template
            # et s'il n'y a pas de caractères spéciaux dans le nom qui interfèrent avec .find()
            start_index = phrase_contexte.find(nom_propre)
            
            if start_index != -1:
                end_index = start_index + len(nom_propre)
                # Créer l'annotation pour cette entité
                annotation = {"entities": [(start_index, end_index, label_entite)]}
                donnees_spacy_format.append((phrase_contexte, annotation))
            else:
                # Avertissement si le nom n'est pas trouvé (peut arriver si le nom contient
                # des caractères que .replace ou .find gèrent mal, ou si le template est complexe)
                print(f"Attention : Le nom '{nom_propre}' n'a pas été trouvé dans la phrase générée : '{phrase_contexte}'. "
                      "Vérifiez le nom ou le template.")
                
    return donnees_spacy_format

if __name__ == "__main__":
    # 1. Charger la liste des noms propres depuis votre fichier annuaire .txt
    noms_a_fine_tuner = charger_noms_annuaire(CHEMIN_ANNUAIRE_TXT)

    if noms_a_fine_tuner:
        print(f"Chargement de {len(noms_a_fine_tuner)} noms depuis '{CHEMIN_ANNUAIRE_TXT}'.")

        # 2. Générer les données d'entraînement au format SpaCy
        donnees_pour_spacy = generer_donnees_spacy(noms_a_fine_tuner, PHRASES_TEMPLATES_LONGUES, LABEL_ENTITE)
        
        print(f"Génération de {len(donnees_pour_spacy)} exemples d'entraînement.")

        # 3. Afficher quelques exemples pour vérification (optionnel)
        if donnees_pour_spacy:
            print("\nQuelques exemples de données générées :")
            for i in range(min(3, len(donnees_pour_spacy))): # Affiche les 3 premiers
                print(donnees_pour_spacy[i])
        
        # 4. Sauvegarder les données générées dans un fichier JSON (pour inspection/utilisation ultérieure)
        try:
            with open(CHEMIN_DONNEES_GENEREES_JSON, "w", encoding="utf-8") as f_out:
                json.dump(donnees_pour_spacy, f_out, ensure_ascii=False, indent=2)
            print(f"\nDonnées d'entraînement sauvegardées avec succès dans '{CHEMIN_DONNEES_GENEREES_JSON}'")
        except IOError as e:
            print(f"Erreur lors de la sauvegarde des données JSON : {e}")
        except Exception as e_gen:
            print(f"Une erreur inattendue est survenue lors de la sauvegarde : {e_gen}")
            
    else:
        print("Aucun nom n'a été chargé depuis l'annuaire. Vérifiez le fichier et le chemin.")