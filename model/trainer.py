# import pandas as pd
# import xgboost as xgb
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import os

# # Outils d'association depuis mlxtend
# from mlxtend.frequent_patterns import apriori, association_rules

# def create_bins_dict(df, numeric_cols):
#     """
#     Crée un dictionnaire des intervalles (bins) pour chaque colonne numérique.
#     """
#     bins_dict = {}
#     for col in numeric_cols:
#         try:
#             # Utilisation de 4 intervalles
#             _, bins = pd.qcut(df[col], 4, duplicates='drop', retbins=True)
#             bins_dict[col] = bins
#         except Exception as e:
#             bins_dict[col] = None
#     return bins_dict

# def extract_association_rules(df, target_col, min_support=0.1, min_confidence=0.7, bins_dict=None):
#     """
#     Discrétise les variables numériques selon les bins fournis (ou via pd.qcut par défaut),
#     effectue un encodage one-hot, puis extrait des règles d'association.
#     Seules les règles dont la conséquence concerne le target_col sont retenues.
#     """
#     df_rules = df.copy()
#     for col in df_rules.columns:
#         if col != target_col and pd.api.types.is_numeric_dtype(df_rules[col]):
#             if bins_dict is not None and col in bins_dict and bins_dict[col] is not None:
#                 bins = bins_dict[col]
#                 df_rules[col] = pd.cut(df_rules[col], bins=bins, include_lowest=True).astype(str)
#             else:
#                 df_rules[col] = pd.qcut(df_rules[col], 4, duplicates='drop').astype(str)
#         else:
#             df_rules[col] = df_rules[col].astype(str)
#     df_encoded = pd.get_dummies(df_rules)
#     frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
#     # On garde les règles dont les consequents contiennent le target_col
#     rules = rules[rules['consequents'].apply(lambda x: any(target_col in item for item in x))]
#     return rules

# def train_xgboost_model(csv_path, model_path, rules_path=None, bins_path=None):
#     # Charger le dataset
#     df = pd.read_csv(csv_path)
    
#     # Supposons que la colonne cible s'appelle "NObeyesdad"
#     X = df.drop("NObeyesdad", axis=1)
#     y = df["NObeyesdad"]

#     # Encoder la cible
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
    
#     # Transformation des variables catégorielles pour X
#     X = pd.get_dummies(X)
    
#     # Division en ensembles d'entraînement et de test
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
#     # Entraînement du modèle XGBoost
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
#     model.fit(X_train, y_train)
    
#     preds = model.predict(X_test)
#     acc = accuracy_score(y_test, preds)
#     print("Accuracy du modèle :", acc)
    
#     # Créer et sauvegarder le dictionnaire des bins pour les variables numériques (hors cible)
#     numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "NObeyesdad"]
#     bins_dict = create_bins_dict(df, numeric_cols)
#     if bins_path is not None:
#         os.makedirs(os.path.dirname(bins_path), exist_ok=True)
#         with open(bins_path, "wb") as f:
#             pickle.dump(bins_dict, f)
#         print("Bins enregistrés dans :", bins_path)
    
#     # Extraire les règles d'association en utilisant les bins
#     rules = extract_association_rules(df, target_col="NObeyesdad", min_support=0.1, min_confidence=0.7, bins_dict=bins_dict)
#     if rules_path is not None:
#         os.makedirs(os.path.dirname(rules_path), exist_ok=True)
#         with open(rules_path, "wb") as f:
#             pickle.dump(rules, f)
#         print("Association rules extraites et sauvegardées dans :", rules_path)
    
#     # Sauvegarder le modèle et le LabelEncoder
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     with open(model_path, "wb") as f:
#         pickle.dump({"model": model, "le": le}, f)

import pandas as pd
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
import os

def create_bins_dict(df, numeric_cols):
    """
    Crée un dictionnaire des intervalles (bins) pour chaque colonne numérique.
    """
    bins_dict = {}
    for col in numeric_cols:
        try:
            # Utilisation de 4 intervalles
            _, bins = pd.qcut(df[col], 4, duplicates='drop', retbins=True)
            bins_dict[col] = bins
        except Exception as e:
            bins_dict[col] = None
    return bins_dict

def extract_association_rules(df, target_col, min_support=0.1, min_confidence=0.7, bins_dict=None):
    """
    Discrétise les variables numériques selon les bins fournis,
    effectue un encodage one-hot, puis extrait des règles d'association.
    Seules les règles dont les consequents contiennent target_col sont retenues.
    """
    df_rules = df.copy()
    for col in df_rules.columns:
        if col != target_col and pd.api.types.is_numeric_dtype(df_rules[col]):
            if bins_dict is not None and col in bins_dict and bins_dict[col] is not None:
                bins = bins_dict[col]
                df_rules[col] = pd.cut(df_rules[col], bins=bins, include_lowest=True).astype(str)
            else:
                df_rules[col] = pd.qcut(df_rules[col], 4, duplicates='drop').astype(str)
        else:
            df_rules[col] = df_rules[col].astype(str)
    df_encoded = pd.get_dummies(df_rules)
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['consequents'].apply(lambda x: any(target_col in item for item in x))]
    return rules

def train_association_classifier(csv_path, rules_path=None, bins_path=None):
    """
    Entraîne un classifieur basé sur l'extraction de règles d'association.
    Charge le dataset, crée les bins pour les variables numériques,
    extrait les règles d'association et sauvegarde le dictionnaire des bins et les règles.
    Renvoie un dictionnaire contenant ces informations.
    """
    df = pd.read_csv(csv_path)
    target_col = "NObeyesdad"
    
    # Création du dictionnaire des bins pour les colonnes numériques (hors cible)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
    bins_dict = create_bins_dict(df, numeric_cols)
    if bins_path is not None:
        os.makedirs(os.path.dirname(bins_path), exist_ok=True)
        with open(bins_path, "wb") as f:
            pickle.dump(bins_dict, f)
        print("Bins enregistrés dans :", bins_path)
    
    # Extraction des règles d'association en utilisant les bins
    rules = extract_association_rules(df, target_col=target_col, min_support=0.08, min_confidence=0.6, bins_dict=bins_dict)
    if rules_path is not None:
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        with open(rules_path, "wb") as f:
            pickle.dump(rules, f)
        print("Association rules extraites et sauvegardées dans :", rules_path)
    
    return {"rules": rules, "bins": bins_dict}

def train_xgboost_model(csv_path, model_path, rules_path=None, bins_path=None):
    """
    Entraîne un modèle XGBoost (pour comparaison) et extrait également les règles et bins.
    """
    # Charger le dataset
    df = pd.read_csv(csv_path)
    
    # Supposons que la colonne cible s'appelle "NObeyesdad"
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]
    
    # Ici, pour le modèle XGBoost, vous pouvez continuer votre pipeline habituel.
    # Nous ne détaillons pas l'entraînement XGBoost ici, car notre focus est sur l'extraction des règles.
    # Vous pouvez néanmoins entraîner votre modèle XGBoost si besoin.
    # Dans cet exemple, nous utilisons uniquement l'approche apriori pour la prédiction.
    
    # Extraire les règles et les bins
    model_data = train_association_classifier(csv_path, rules_path=rules_path, bins_path=bins_path)
    # Sauvegarder un "modèle" fictif : on sauvegarde uniquement les règles et bins
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print("Modèle (classifieur Apriori) sauvegardé dans :", model_path)
