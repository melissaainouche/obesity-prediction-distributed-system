#!/usr/bin/env python3
import sys, os

# Calcul du chemin absolu du dossier 'slice' en se basant sur le fichier client.py
current_dir = os.path.dirname(os.path.abspath(__file__))
slice_path = os.path.join(current_dir, "..", "slice")
sys.path.append(slice_path)

import Ice, json
import MyPredictor  # Module généré par slice
# Reste du code...

def connect_to_server():
    with Ice.initialize(sys.argv) as communicator:
        proxy = communicator.stringToProxy("Predictor:default -p 10000")
        predictor = MyPredictor.PredictorPrx.checkedCast(proxy)
        if not predictor:
            print("Proxy invalide")
            sys.exit(1)
        return predictor

def send_training_data(predictor, local_csv_path):
    with open(local_csv_path, "r") as f:
        csv_data = f.read()
    result = predictor.sendTrainingData(csv_data)
    print("sendTrainingData:", result)

def train_model(predictor):
    result = predictor.trainModel()
    print("trainModel:", result)

def predict_case(predictor, case_data):
    input_json = json.dumps(case_data)
    result_json = predictor.predict(input_json)
    result = json.loads(result_json)
    print("predict:", result)

if __name__ == "__main__":
    predictor = connect_to_server()
    
    # Exemples d'utilisation :
    # 1) Envoyer un fichier CSV local
    # send_training_data(predictor, "nouveaux_cas.csv")

    # 2) Lancer l'entraînement
    # train_model(predictor)

    # 3) Faire une prédiction
    # new_case = {"Gender": "Male", "Age": 9, "Weight": 32}
    # predict_case(predictor, new_case)
