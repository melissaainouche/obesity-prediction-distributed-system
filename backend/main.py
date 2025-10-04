# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import sys, os, Ice, json, uuid

# # Ajout du dossier 'slice' dans le sys.path pour accéder aux modules générés par slice2py
# current_dir = os.path.dirname(os.path.abspath(__file__))
# slice_path = os.path.join(current_dir, "..", "slice")
# sys.path.append(slice_path)
# import MyPredictor  # Module généré par slice

# app = FastAPI()

# # Configuration du middleware CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # À restreindre en production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialisation globale d'Ice
# communicator = Ice.initialize(sys.argv)

# # Montage des fichiers statiques depuis le dossier "front"
# static_dir = os.path.join(current_dir, "..", "front")
# app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# # Route pour servir la page d'accueil
# @app.get("/")
# def read_index():
#     return FileResponse(os.path.join(static_dir, "index.html"))

# # Dictionnaires globaux pour gérer les sessions clients
# client_connections = {}   # client_id => proxy Ice (connexion persistante)
# client_transactions = {}  # client_id => liste des transactions

# # Modèles de données
# class CSVData(BaseModel):
#     csv_str: str

# class CaseData(BaseModel):
#     Age: float
#     Height: float
#     Weight: float
#     FCVC: float
#     NCP: float
#     CH2O: float
#     FAF: float
#     TUE: float
#     Gender: str
#     family_history_with_overweight: str
#     FAVC: str
#     CAEC: str
#     SMOKE: str
#     SCC: str
#     CALC: str
#     MTRANS: str

# # Chemins pour les données, le modèle, les règles et les bins
# DATA_DIR = os.path.join("data")
# os.makedirs(DATA_DIR, exist_ok=True)
# DATA_PATH = os.path.join(DATA_DIR, "data.csv")
# MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.pkl")
# RULES_PATH = os.path.join(DATA_DIR, "rules.pkl")
# BINS_PATH = os.path.join(DATA_DIR, "bins.pkl")

# # Endpoint pour établir une session (simulateur d'un client persistant)
# @app.post("/connect")
# def connect():
#     proxy = communicator.stringToProxy("Predictor:default -p 10000")
#     predictor = MyPredictor.PredictorPrx.checkedCast(proxy)
#     if not predictor:
#         raise HTTPException(status_code=500, detail="Proxy invalide")
#     client_id = str(uuid.uuid4())
#     client_connections[client_id] = predictor
#     client_transactions[client_id] = []
#     print(f"Nouvelle connexion établie : client_id={client_id}, serveur=Predictor:default -p 10000")
#     return {"client_id": client_id, "server": "Predictor:default -p 10000", "transactions": []}

# # Endpoint pour envoyer des données d'entraînement
# @app.post("/send_data")
# def send_data(data: CSVData, client_id: str):
#     if client_id not in client_connections:
#         raise HTTPException(status_code=400, detail="Client non connecté")
#     predictor = client_connections[client_id].ice_context({"client_id": client_id})
#     result = predictor.sendTrainingData(data.csv_str)
#     transaction = {"type": "send_data", "result": result}
#     client_transactions[client_id].append(transaction)
#     print(f"Transaction send_data pour client {client_id} : {result}")
#     return {"status": result, "transactions": client_transactions[client_id]}

# # Endpoint pour lancer l'entraînement du modèle
# @app.post("/train")
# def train(client_id: str):
#     if client_id not in client_connections:
#         raise HTTPException(status_code=400, detail="Client non connecté")
#     predictor = client_connections[client_id].ice_context({"client_id": client_id})
#     result = predictor.trainModel()
#     transaction = {"type": "train", "result": result}
#     client_transactions[client_id].append(transaction)
#     print(f"Transaction train pour client {client_id} : {result}")
#     return {"status": result, "transactions": client_transactions[client_id]}

# # Endpoint pour réaliser une prédiction avec explications (SHAP et règles)
# @app.post("/predict")
# def predict(data: CaseData, client_id: str):
#     if client_id not in client_connections:
#         raise HTTPException(status_code=400, detail="Client non connecté")
#     predictor = client_connections[client_id].ice_context({"client_id": client_id})
#     input_json = data.json()
#     result_json = predictor.predict(input_json)
#     result = json.loads(result_json)
#     transaction = {"type": "predict", "result": result}
#     client_transactions[client_id].append(transaction)
#     print(f"Transaction predict pour client {client_id} : {result}")
#     # Renvoi d'un objet imbriqué pour la prédiction
#     return {
#         "status": "ok",
#         "prediction": {
#             "prediction": result.get("prediction"),
#             "explanation": result.get("explanation")
#         },
#         "transactions": client_transactions[client_id]
#     }



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys, os, Ice, json, uuid

current_dir = os.path.dirname(os.path.abspath(__file__))
slice_path = os.path.join(current_dir, "..", "slice")
sys.path.append(slice_path)
import MyPredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

communicator = Ice.initialize(sys.argv)

static_dir = os.path.join(current_dir, "..", "front")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

client_connections = {}
client_transactions = {}

class CSVData(BaseModel):
    csv_str: str

class CaseData(BaseModel):
    Age: float
    Height: float
    Weight: float
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float
    Gender: str
    family_history_with_overweight: str
    FAVC: str
    CAEC: str
    SMOKE: str
    SCC: str
    CALC: str
    MTRANS: str

DATA_DIR = os.path.join("data")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.pkl")
RULES_PATH = os.path.join(DATA_DIR, "rules.pkl")
BINS_PATH = os.path.join(DATA_DIR, "bins.pkl")

@app.post("/connect")
def connect():
    proxy = communicator.stringToProxy("Predictor:default -p 10000")
    predictor = MyPredictor.PredictorPrx.checkedCast(proxy)
    if not predictor:
        raise HTTPException(status_code=500, detail="Proxy invalide")
    client_id = str(uuid.uuid4())
    client_connections[client_id] = predictor
    client_transactions[client_id] = []
    print(f"Nouvelle connexion établie : client_id={client_id}, serveur=Predictor:default -p 10000")
    return {"client_id": client_id, "server": "Predictor:default -p 10000", "transactions": []}

@app.post("/send_data")
def send_data(data: CSVData, client_id: str):
    if client_id not in client_connections:
        raise HTTPException(status_code=400, detail="Client non connecté")
    predictor = client_connections[client_id].ice_context({"client_id": client_id})
    result = predictor.sendTrainingData(data.csv_str)
    transaction = {"type": "send_data", "result": result}
    client_transactions[client_id].append(transaction)
    print(f"Transaction send_data pour client {client_id} : {result}")
    return {"status": result, "transactions": client_transactions[client_id]}

@app.post("/train")
def train(client_id: str):
    if client_id not in client_connections:
        raise HTTPException(status_code=400, detail="Client non connecté")
    predictor = client_connections[client_id].ice_context({"client_id": client_id})
    result = predictor.trainModel()
    transaction = {"type": "train", "result": result}
    client_transactions[client_id].append(transaction)
    print(f"Transaction train pour client {client_id} : {result}")
    return {"status": result, "transactions": client_transactions[client_id]}

@app.post("/predict")
def predict(data: CaseData, client_id: str):
    if client_id not in client_connections:
        raise HTTPException(status_code=400, detail="Client non connecté")
    predictor = client_connections[client_id].ice_context({"client_id": client_id})
    input_json = data.json()
    result_json = predictor.predict(input_json)
    result = json.loads(result_json)
    transaction = {"type": "predict", "result": result}
    client_transactions[client_id].append(transaction)
    print(f"Transaction predict pour client {client_id} : {result}")
    return {
        "status": "ok",
        "prediction": {
            "prediction": result.get("prediction"),
            "explanation": result.get("explanation")
        },
        "transactions": client_transactions[client_id]
    }
