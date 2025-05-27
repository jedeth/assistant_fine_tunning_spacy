# assistant_finetuning_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import subprocess # Pour lancer des commandes externes comme spacy train
import threading # Pour exécuter spacy train sans geler l'interface
import json
from tkinter import messagebox # Pour lire/écrire des configurations si besoin
import spacy
import sys

# --- Fonctions Utilitaires (à placer ici ou dans un module séparé) ---

def generer_config_spacy(config_path, train_path, dev_path, base_model, params=None):
    """
    Génère un fichier config.cfg pour l'entraînement SpaCy,
    configuré pour le fine-tuning des composants tok2vec et ner.
    params: dictionnaire optionnel pour surcharger les paramètres de training.
    """
    if params is None:
        params = {}

    # Assurer que les chemins sont correctement formatés pour le fichier config
    # (surtout pour Windows, où les backslashes doivent être échappés ou la chaîne doit être brute)
    train_path_config = train_path.replace("\\", "\\\\")
    dev_path_config = dev_path.replace("\\", "\\\\")

    config_content = f"""
[paths]
train = "{train_path_config}"
dev = "{dev_path_config}"
vectors = "{base_model}"
init_tok2vec = null

[system]
gpu_allocator = null
 # Adaptez si vous utilisez un GPU
seed = 0

[nlp]
lang = "fr"
pipeline = ["tok2vec", "ner"]
batch_size = 1000
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.tok2vec]
factory = "tok2vec"
source = "{base_model}"
# On source le tok2vec de base_model
# Pas besoin de la sous-section [components.tok2vec.model] ici,
# car 'source' indique à SpaCy de prendre l'architecture ET les poids du modèle source.

[components.ner]
factory = "ner"
source = "{base_model}"
 # On source le NER de base_model
# Pas besoin de la sous-section [components.ner.model] pour un fine-tuning simple.
# SpaCy utilisera l'architecture du composant NER du modèle source.
# Si vous vouliez une architecture NER différente tout en initialisant avec des poids
# d'un autre modèle, la configuration serait plus complexe.
# Pour le fine-tuning standard, sourcer le composant est la voie.

# La section [components.ner.model.tok2vec] (le listener) est implicitement configurée
# lorsque le composant 'ner' est sourcé et qu'un 'tok2vec' est dans le pipeline.
# Il n'est généralement pas nécessaire de la définir explicitement si la structure est simple.

[corpora]
[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
 # 0 signifie pas de limite de longueur pour les documents d'entraînement

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
 # 0 signifie pas de limite

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = {params.get('dropout', 0.1)}
patience = {params.get('patience', 1600)}
max_epochs = {params.get('max_epochs', 0)}
 # 0 pour un entraînement basé sur max_steps
max_steps = {params.get('max_steps', 5000)}
 # Nombre d'étapes (peut être ajusté)
eval_frequency = {params.get('eval_frequency', 200)}
frozen_components = []
 # Important : vide pour fine-tuner tok2vec et ner
score_weights = {{"ents_f": 1.0}}
 # Score principal pour model-best

# Paramètres de l'optimiseur Adam, souvent laissés par défaut pour un premier essai
[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = {params.get('learn_rate', 0.001)}
 # Taux d'apprentissage

# Configuration du batcher
[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
# get_length = null
#  # Pas nécessaire de le définir explicitement

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
# t = 0.0
#  # Pas nécessaire de le définir explicitement

[initialize]
vectors = ${{paths.vectors}}
# init_tok2vec = null
#  # Normalement pas besoin si tok2vec est sourcé.
# Si vous avez des problèmes, commentez cette ligne ou assurez-vous qu'elle est bien null.
# La section [initialize.components] n'est pas nécessaire si le sourcing des composants fonctionne.
    """
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Fichier config.cfg généré : {config_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la génération de config.cfg : {e}")
        return False

# --- Classe Principale de l'Interface ---
class AssistantFineTuningApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Assistant de Fine-Tuning SpaCy NER")
        self.root.geometry("800x700")

        self.chemin_annuaire_var = tk.StringVar()
        self.chemin_phrases_modeles_var = tk.StringVar()
        self.modele_spacy_base_var = tk.StringVar(value="fr_core_news_md")
        self.chemin_modele_sortie_var = tk.StringVar(value="./modele_finetune_assistant_v2") # Nom de dossier légèrement différent

        self.max_steps_var = tk.StringVar(value="5000") 
        self.learn_rate_var = tk.StringVar(value="0.001")

        self.creer_widgets()

    def creer_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        data_config_frame = ttk.LabelFrame(main_frame, text="1. Configuration des Données", padding="10")
        data_config_frame.pack(fill=tk.X, pady=5)

        ttk.Label(data_config_frame, text="Fichier Annuaire Noms Propres (.txt):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(data_config_frame, textvariable=self.chemin_annuaire_var, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(data_config_frame, text="Parcourir...", command=self.choisir_fichier_annuaire).grid(row=0, column=2, padx=5)

        ttk.Label(data_config_frame, text="Fichier Phrases Modèles (.txt, {NOM} dedans):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(data_config_frame, textvariable=self.chemin_phrases_modeles_var, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(data_config_frame, text="Parcourir...", command=self.choisir_fichier_phrases_modeles).grid(row=1, column=2, padx=5)
        
        data_config_frame.columnconfigure(1, weight=1)

        model_config_frame = ttk.LabelFrame(main_frame, text="2. Configuration du Modèle et de l'Entraînement", padding="10")
        model_config_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_config_frame, text="Modèle SpaCy de base:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        modeles = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"]
        ttk.Combobox(model_config_frame, textvariable=self.modele_spacy_base_var, values=modeles, state="readonly", width=47).grid(row=0, column=1, sticky=tk.EW, padx=5)

        ttk.Label(model_config_frame, text="Dossier de sortie du modèle affiné:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(model_config_frame, textvariable=self.chemin_modele_sortie_var, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(model_config_frame, text="Parcourir...", command=self.choisir_dossier_sortie).grid(row=1, column=2, padx=5)
        
        ttk.Label(model_config_frame, text="Max Étapes (max_steps):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(model_config_frame, textvariable=self.max_steps_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(model_config_frame, text="Taux d'Apprentissage (learn_rate):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(model_config_frame, textvariable=self.learn_rate_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        model_config_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X, pady=5)

        self.btn_preparer_donnees = ttk.Button(action_frame, text="Préparer les Données d'Entraînement", command=self.preparer_donnees)
        self.btn_preparer_donnees.pack(side=tk.LEFT, padx=5)

        self.btn_lancer_finetuning = ttk.Button(action_frame, text="Lancer le Fine-Tuning", command=self.lancer_finetuning, state=tk.DISABLED)
        self.btn_lancer_finetuning.pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Logs de l'Entraînement", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
        self.log_text_area.pack(fill=tk.BOTH, expand=True)

    def choisir_fichier_annuaire(self):
        filepath = filedialog.askopenfilename(title="Sélectionner le fichier d'annuaire (.txt)", filetypes=[("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")])
        if filepath:
            self.chemin_annuaire_var.set(filepath)
    
    def choisir_fichier_phrases_modeles(self):
        filepath = filedialog.askopenfilename(title="Sélectionner le fichier de phrases modèles (.txt)", filetypes=[("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")])
        if filepath:
            self.chemin_phrases_modeles_var.set(filepath)

    def choisir_dossier_sortie(self):
        folderpath = filedialog.askdirectory(title="Sélectionner le dossier de sortie pour le modèle affiné")
        if folderpath:
            self.chemin_modele_sortie_var.set(folderpath)
            
    def ajouter_log(self, message):
        self.log_text_area.config(state=tk.NORMAL)
        self.log_text_area.insert(tk.END, message + "\n")
        self.log_text_area.see(tk.END)
        self.log_text_area.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def preparer_donnees(self):
        chemin_annuaire = self.chemin_annuaire_var.get()
        chemin_phrases = self.chemin_phrases_modeles_var.get()

        if not chemin_annuaire or not chemin_phrases:
            messagebox.showerror("Erreur", "Veuillez spécifier le fichier d'annuaire ET le fichier de phrases modèles.")
            return

        self.ajouter_log("Préparation des données en cours...")
        
        def charger_noms(chemin):
            noms = []
            try:
                with open(chemin, 'r', encoding='utf-8') as f:
                    for ligne in f:
                        nom = ligne.strip()
                        if nom: noms.append(nom)
            except Exception as e:
                self.ajouter_log(f"Erreur chargement annuaire '{chemin}': {e}")
            return noms

        def charger_phrases_templates(chemin):
            templates = []
            try:
                with open(chemin, 'r', encoding='utf-8') as f:
                    for ligne in f:
                        template = ligne.strip()
                        if template and "{NOM}" in template: templates.append(template)
            except Exception as e:
                 self.ajouter_log(f"Erreur chargement phrases modèles '{chemin}': {e}")
            return templates
            
        def generer_donnees_format_spacy(liste_noms, phrases_templates, label_entite="PER"):
            donnees_spacy_format = []
            for nom_propre in liste_noms:
                for template in phrases_templates:
                    phrase_contexte = template.replace("{NOM}", nom_propre)
                    start_index = phrase_contexte.find(nom_propre)
                    if start_index != -1:
                        end_index = start_index + len(nom_propre)
                        annotation = {"entities": [(start_index, end_index, label_entite)]}
                        donnees_spacy_format.append((phrase_contexte, annotation))
                    else:
                        self.ajouter_log(f"Attention : Nom '{nom_propre}' non trouvé dans : '{phrase_contexte}'")
            return donnees_spacy_format

        noms_propres = charger_noms(chemin_annuaire)
        phrases_modeles = charger_phrases_templates(chemin_phrases)

        if not noms_propres or not phrases_modeles:
            self.ajouter_log("Erreur : Annuaire ou phrases modèles vides ou non chargés.")
            messagebox.showerror("Erreur Données", "Impossible de charger l'annuaire ou les phrases modèles.")
            return

        donnees_json_pour_conversion = generer_donnees_format_spacy(noms_propres, phrases_modeles)
        self.ajouter_log(f"{len(donnees_json_pour_conversion)} exemples générés au format JSON.")
        
        try:
            nlp_tokenizer = spacy.load(self.modele_spacy_base_var.get())
        except OSError:
            nlp_tokenizer = spacy.blank("fr")
            self.ajouter_log(f"Avertissement: Modèle {self.modele_spacy_base_var.get()} non trouvé, utilisation de 'fr' pour tokenisation.")
        except Exception as e:
             self.ajouter_log(f"Erreur chargement modèle {self.modele_spacy_base_var.get()} pour tokenisation: {e}")
             messagebox.showerror("Erreur Modèle", f"Impossible de charger {self.modele_spacy_base_var.get()}. Vérifiez qu'il est installé.")
             return


        if len(donnees_json_pour_conversion) < 10:
            self.ajouter_log("Pas assez de données pour une division train/dev significative. Mettre tout en entraînement.")
            donnees_train_final = donnees_json_pour_conversion
            donnees_dev_final = donnees_json_pour_conversion[:max(1, int(len(donnees_json_pour_conversion)*0.1))] 
        else:
            try:
                from sklearn.model_selection import train_test_split
                donnees_train_final, donnees_dev_final = train_test_split(donnees_json_pour_conversion, test_size=0.2, random_state=42)
            except ImportError:
                self.ajouter_log("Erreur: scikit-learn non trouvé. Veuillez l'installer (`pip install scikit-learn`) pour diviser les données.")
                messagebox.showerror("Dépendance Manquante", "scikit-learn est requis. Veuillez l'installer.")
                return
            except Exception as e_split:
                self.ajouter_log(f"Erreur lors de la division des données: {e_split}")
                return

        self.ajouter_log(f"{len(donnees_train_final)} exemples pour train.spacy, {len(donnees_dev_final)} pour dev.spacy.")

        self.train_spacy_path = os.path.join(os.getcwd(), "train.spacy")
        self.dev_spacy_path = os.path.join(os.getcwd(), "dev.spacy")

        from spacy.tokens import DocBin
        
        try:
            db_train = DocBin()
            for texte, annotations in donnees_train_final:
                doc = nlp_tokenizer.make_doc(texte)
                entites_valides = []
                for debut, fin, label in annotations.get("entities", []):
                    span = doc.char_span(debut, fin, label=label, alignment_mode="contract")
                    if span is None:
                        self.ajouter_log(f"Attention (train): Span non créé pour '{texte[debut:fin]}' dans '{texte}'")
                    else:
                        entites_valides.append(span)
                try:
                    doc.ents = entites_valides
                except ValueError as e: self.ajouter_log(f"Err assign ents (train) '{texte[:30]}...': {e}")
                db_train.add(doc)
            db_train.to_disk(self.train_spacy_path)
            self.ajouter_log(f"Fichier train.spacy créé : {self.train_spacy_path}")

            db_dev = DocBin()
            for texte, annotations in donnees_dev_final:
                doc = nlp_tokenizer.make_doc(texte)
                entites_valides = []
                for debut, fin, label in annotations.get("entities", []):
                    span = doc.char_span(debut, fin, label=label, alignment_mode="contract")
                    if span is None:
                        self.ajouter_log(f"Attention (dev): Span non créé pour '{texte[debut:fin]}' dans '{texte}'")
                    else:
                        entites_valides.append(span)
                try:
                    doc.ents = entites_valides
                except ValueError as e: self.ajouter_log(f"Err assign ents (dev) '{texte[:30]}...': {e}")
                db_dev.add(doc)
            db_dev.to_disk(self.dev_spacy_path)
            self.ajouter_log(f"Fichier dev.spacy créé : {self.dev_spacy_path}")
        except Exception as e_docbin:
            self.ajouter_log(f"Erreur lors de la création des fichiers DocBin : {e_docbin}")
            messagebox.showerror("Erreur DocBin", f"Erreur lors de la création des fichiers .spacy: {e_docbin}")
            return

        self.ajouter_log("Préparation des données terminée !")
        self.btn_lancer_finetuning.config(state=tk.NORMAL)
        messagebox.showinfo("Succès", "Fichiers train.spacy et dev.spacy générés.")


    def lancer_finetuning_thread(self):
        try:
            self.btn_preparer_donnees.config(state=tk.DISABLED)
            self.btn_lancer_finetuning.config(state=tk.DISABLED)
            self.ajouter_log("Lancement du fine-tuning...")

            base_model_choisi = self.modele_spacy_base_var.get()
            dossier_sortie_modele = self.chemin_modele_sortie_var.get()
            # Utiliser un nom de fichier config spécifique pour cet assistant
            config_file_path = os.path.join(os.getcwd(), "config_assistant_ft.cfg") 

            params_training = {
                "max_steps": int(self.max_steps_var.get()),
                "learn_rate": float(self.learn_rate_var.get())
            }

            # S'assurer que les attributs train_spacy_path et dev_spacy_path existent
            if not hasattr(self, 'train_spacy_path') or not hasattr(self, 'dev_spacy_path'):
                messagebox.showerror("Erreur", "Les chemins vers train.spacy et dev.spacy ne sont pas définis. Veuillez d'abord préparer les données.")
                self.btn_preparer_donnees.config(state=tk.NORMAL)
                self.btn_lancer_finetuning.config(state=tk.NORMAL)
                return

            if not generer_config_spacy(config_file_path, self.train_spacy_path, self.dev_spacy_path, base_model_choisi, params_training):
                self.ajouter_log("Erreur: Impossible de générer le fichier de configuration.")
                messagebox.showerror("Erreur Config", "Impossible de générer le fichier de configuration.")
                self.btn_preparer_donnees.config(state=tk.NORMAL)
                self.btn_lancer_finetuning.config(state=tk.NORMAL)
                return

            if not os.path.exists(self.train_spacy_path) or not os.path.exists(self.dev_spacy_path):
                self.ajouter_log("Erreur : Fichiers train.spacy ou dev.spacy manquants.")
                messagebox.showerror("Erreur", "Fichiers de données .spacy manquants.")
                self.btn_preparer_donnees.config(state=tk.NORMAL)
                self.btn_lancer_finetuning.config(state=tk.NORMAL)
                return

            if not dossier_sortie_modele:
                messagebox.showerror("Erreur", "Veuillez spécifier un dossier de sortie pour le modèle affiné.")
                self.btn_preparer_donnees.config(state=tk.NORMAL)
                self.btn_lancer_finetuning.config(state=tk.NORMAL)
                return

            commande = [
                sys.executable, 
                "-m", "spacy", "train",
                config_file_path,
                "--output", dossier_sortie_modele,
            ]
            
            self.ajouter_log(f"Commande : {' '.join(commande)}")

            console_encoding = sys.stdout.encoding if sys.stdout.encoding else 'utf-8'
            if os.name == 'nt':
                try:
                    import ctypes
                    oem_cp = ctypes.windll.kernel32.GetOEMCP()
                    if oem_cp != 0: 
                        console_encoding = f'cp{oem_cp}'
                    else: 
                        console_encoding = 'mbcs' 
                except Exception as e_ctypes:
                    self.ajouter_log(f"Avertissement: Impossible d'utiliser ctypes GetOEMCP ({e_ctypes}). Utilisation 'mbcs'.")
                    console_encoding = 'mbcs' 
            
            self.ajouter_log(f"Utilisation de l'encodage console (logs subprocess) : {console_encoding}")

            process = subprocess.Popen(
                commande, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding=console_encoding, 
                errors='replace', 
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            while True:
                try:
                    output_line = process.stdout.readline()
                except Exception as e_readline:
                    self.ajouter_log(f"Erreur readline: {e_readline}")
                    output_line = "" 

                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    self.ajouter_log(output_line.strip())
            
            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                best_model_path = os.path.join(dossier_sortie_modele, 'model-best')
                self.ajouter_log(f"Fine-tuning terminé avec succès ! Modèle sauvegardé dans : {best_model_path}")
                messagebox.showinfo("Succès", f"Fine-tuning terminé ! Modèle sauvegardé dans : {best_model_path}")
            else:
                self.ajouter_log(f"Erreur lors du fine-tuning (code de retour: {return_code}).")
                messagebox.showerror("Erreur Fine-tuning", "Une erreur s'est produite. Consultez les logs.")

        except Exception as e:
            self.ajouter_log(f"Erreur exceptionnelle lors du lancement du fine-tuning : {e}")
            messagebox.showerror("Erreur", f"Erreur : {e}")
        finally:
            self.btn_preparer_donnees.config(state=tk.NORMAL)
            self.btn_lancer_finetuning.config(state=tk.NORMAL)

    def lancer_finetuning(self):
        thread = threading.Thread(target=self.lancer_finetuning_thread, daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantFineTuningApp(root)
    root.mainloop()