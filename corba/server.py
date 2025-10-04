# #!/usr/bin/env python3
# import os, sys, json, pandas as pd, Ice

# # Définir le chemin du dossier racine du projet (un niveau au-dessus du dossier 'corba')
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.join(current_dir, "..")
# sys.path.append(root_dir)

# # Importer les modules du modèle
# from model.trainer import train_xgboost_model
# from model.predictor import load_model, predict_with_explanation

# # Ajout du chemin pour le dossier slice
# slice_path = os.path.join(current_dir, "..", "slice")
# sys.path.append(slice_path)
# import MyPredictor  # Module généré par slice

# # Définition des chemins (création des dossiers si nécessaire)
# DATA_DIR = os.path.join("data")
# os.makedirs(DATA_DIR, exist_ok=True)
# DATA_PATH = os.path.join(DATA_DIR, "data.csv")
# MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.pkl")
# RULES_PATH = os.path.join(DATA_DIR, "rules.pkl")  # Chemin pour les règles d'association
# BINS_PATH = os.path.join(DATA_DIR, "bins.pkl")      # Chemin pour sauvegarder les intervalles (bins)

# class PredictorI(MyPredictor.Predictor):
#     def __init__(self):
#         self.model = None

#     def sendTrainingData(self, csvData, current=None):
#         client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
#         print(f"[CORBA Server] Requête sendTrainingData reçue de : client_id = {client_id}")
#         try:
#             from io import StringIO
#             new_data = pd.read_csv(StringIO(csvData))
#             if os.path.exists(DATA_PATH):
#                 existing_data = pd.read_csv(DATA_PATH)
#                 merged_data = pd.concat([existing_data, new_data], ignore_index=True)
#             else:
#                 merged_data = new_data
#             merged_data.to_csv(DATA_PATH, index=False)
#             return "Données reçues et fusionnées avec succès."
#         except Exception as e:
#             return "Erreur lors de la fusion des données : " + str(e)

#     def trainModel(self, current=None):
#         client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
#         print(f"[CORBA Server] Requête trainModel reçue de : client_id = {client_id}")
#         try:
#             if not os.path.exists(DATA_PATH):
#                 return "Aucune donnée disponible pour l'entraînement."
#             # Entraîner le modèle et extraire les règles ET sauvegarder les bins utilisés
#             train_xgboost_model(DATA_PATH, MODEL_PATH, rules_path=RULES_PATH, bins_path=BINS_PATH)
#             self.model = None  # Réinitialiser le modèle pour forcer le rechargement
#             return "Modèle entraîné avec succès."
#         except Exception as e:
#             return "Erreur lors de l'entraînement : " + str(e)

#     def predict(self, inputData, current=None):
#         client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
#         print(f"[CORBA Server] Requête predict reçue de : client_id = {client_id}")
#         try:
#             if self.model is None:
#                 # Charger le modèle, les règles et les bins
#                 self.model = load_model(MODEL_PATH, rules_path=RULES_PATH, bins_path=BINS_PATH)
#             features_dict = json.loads(inputData)
#             prediction_result = predict_with_explanation(self.model, features_dict)
#             return json.dumps(prediction_result)
#         except Exception as e:
#             return json.dumps({"error": str(e)})

# with Ice.initialize(sys.argv) as communicator:
#     adapter = communicator.createObjectAdapterWithEndpoints("PredictorAdapter", "default -p 10000")
#     servant = PredictorI()
#     adapter.add(servant, Ice.stringToIdentity("Predictor"))
#     adapter.activate()
#     print("Serveur Ice démarré et prêt à recevoir des requêtes sur le port 10000.")
#     communicator.waitForShutdown()


##!/usr/bin/env python3
import os, sys, json, pandas as pd, Ice

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

from model.trainer import train_association_classifier  # Assurez-vous que cette fonction est définie dans trainer.py
from model.predictor import load_model, predict_with_apriori

slice_path = os.path.join(current_dir, "..", "slice")
sys.path.append(slice_path)
import MyPredictor  # Module généré par slice

DATA_DIR = os.path.join("data")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.pkl")  # Vous pouvez l'utiliser même si non utilisé ici
RULES_PATH = os.path.join(DATA_DIR, "rules.pkl")
BINS_PATH = os.path.join(DATA_DIR, "bins.pkl")

class PredictorI(MyPredictor.Predictor):
    def __init__(self):
        self.model = None  # Ce modèle contiendra les règles et les bins

    def sendTrainingData(self, csvData, current=None):
        client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
        print(f"[CORBA Server] Requête sendTrainingData reçue de : client_id = {client_id}")
        try:
            from io import StringIO
            new_data = pd.read_csv(StringIO(csvData))
            if os.path.exists(DATA_PATH):
                existing_data = pd.read_csv(DATA_PATH)
                merged_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                merged_data = new_data
            merged_data.to_csv(DATA_PATH, index=False)
            return "Données reçues et fusionnées avec succès."
        except Exception as e:
            return "Erreur lors de la fusion des données : " + str(e)

    def trainModel(self, current=None):
        client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
        print(f"[CORBA Server] Requête trainModel reçue de : client_id = {client_id}")
        try:
            if not os.path.exists(DATA_PATH):
                return "Aucune donnée disponible pour l'entraînement."
            # Utiliser la fonction d'entraînement basée sur Apriori
            model_data = train_association_classifier(DATA_PATH, rules_path=RULES_PATH, bins_path=BINS_PATH)
            self.model = model_data
            return "Modèle (Apriori) entraîné avec succès."
        except Exception as e:
            return "Erreur lors de l'entraînement : " + str(e)

    def predict(self, inputData, current=None):
        client_id = current.ctx.get("client_id") if current and current.ctx else "client inconnu"
        print(f"[CORBA Server] Requête predict reçue de : client_id = {client_id}")
        try:
            if self.model is None:
                self.model = load_model(None, rules_path=RULES_PATH, bins_path=BINS_PATH)
            features_dict = json.loads(inputData)
            prediction_result = predict_with_apriori(self.model, features_dict)
            return json.dumps(prediction_result)
        except Exception as e:
            return json.dumps({"error": str(e)})

with Ice.initialize(sys.argv) as communicator:
    adapter = communicator.createObjectAdapterWithEndpoints("PredictorAdapter", "default -p 10000")
    servant = PredictorI()
    adapter.add(servant, Ice.stringToIdentity("Predictor"))
    adapter.activate()
    print("Serveur Ice (Apriori) démarré et prêt à recevoir des requêtes sur le port 10000.")
    communicator.waitForShutdown()
