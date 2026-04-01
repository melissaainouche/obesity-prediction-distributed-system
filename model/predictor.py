"""
Obesity Prediction - Rule-based Inference Engine
------------------------------------------------
Predicts obesity class using association rules extracted from training data.

Workflow:
1. Load precomputed bins and association rules from training phase
2. Transform input instance using bins (discretization)
3. Match instance items against rule antecedents
4. Return prediction from highest confidence rule with explanation

Functions:
- load_model(): Loads rules and bins from pickle files
- discretize_value(): Applies bins to numerical values
- transform_instance(): Converts instance to discrete format
- apply_association_rules(): Matches rules against instance
- predict_with_apriori(): Main prediction function
"""


# import pickle
# import shap
# import numpy as np
# import pandas as pd
# import os

# def load_model(model_path, rules_path=None, bins_path=None):
#     with open(model_path, "rb") as f:
#         model_data = pickle.load(f)
#     if rules_path is not None and os.path.exists(rules_path):
#         with open(rules_path, "rb") as f:
#             rules = pickle.load(f)
#         model_data["rules"] = rules
#     else:
#         model_data["rules"] = None
#     if bins_path is not None and os.path.exists(bins_path):
#         with open(bins_path, "rb") as f:
#             bins_dict = pickle.load(f)
#         model_data["bins"] = bins_dict
#     else:
#         model_data["bins"] = None
#     return model_data

# def discretize_value(attribute, value, bins):
#     for i in range(len(bins) - 1):
#         if bins[i] < value <= bins[i+1]:
#             return f"{attribute}_({bins[i]},{bins[i+1]}]"
#     return f"{attribute}_{value}"

# def transform_instance(instance, bins_dict):
#     transformed = {}
#     for key, val in instance.items():
#         if bins_dict is not None and key in bins_dict and bins_dict[key] is not None:
#             transformed[key] = discretize_value(key, val, bins_dict[key])
#         else:
#             transformed[key] = str(val)
#     return transformed

# def apply_association_rules(rules, instance, bins_dict):
#     # Transformer l'instance en utilisant les mêmes bins
#     transformed_instance = transform_instance(instance, bins_dict)
#     instance_items = set(f"{key}_{val}" for key, val in transformed_instance.items())
#     print("Instance items :", instance_items)
    
#     triggered_rules = []
#     for idx, rule in rules.iterrows():
#         antecedents = set(rule['antecedents'])
#         if antecedents.issubset(instance_items):
#             triggered_rules.append({
#                 "antecedents": antecedents,
#                 "consequents": rule['consequents'],
#                 "confidence": rule['confidence'],
#                 "lift": rule['lift']
#             })
#             print("Règle déclenchée :", antecedents, "=>", rule['consequents'])
#     if not triggered_rules:
#         print("Aucune règle déclenchée pour cette instance.")
#     return triggered_rules

# def predict_with_explanation(model_data, features_dict):
#     model = model_data["model"]
#     le = model_data["le"]
#     rules = model_data.get("rules")
#     bins_dict = model_data.get("bins")
    
#     # Pour la prédiction numérique et SHAP, on utilise les données brutes
#     df_input = pd.DataFrame([features_dict])
#     df_input = pd.get_dummies(df_input)
#     expected_columns = [
#         'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
#         'Gender_Female', 'Gender_Male',
#         'family_history_with_overweight_no', 'family_history_with_overweight_yes',
#         'FAVC_no', 'FAVC_yes',
#         'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
#         'SMOKE_no', 'SMOKE_yes',
#         'SCC_no', 'SCC_yes',
#         'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
#         'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
#         'MTRANS_Public_Transportation', 'MTRANS_Walking'
#     ]
#     df_input = df_input.reindex(columns=expected_columns, fill_value=0)
    
#     pred_int = model.predict(df_input)[0]
#     pred = le.inverse_transform([pred_int])[0]
    
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(df_input)
#     if isinstance(shap_values, list):
#         predicted_class = np.argmax(model.predict_proba(df_input))
#         shap_contrib = shap_values[predicted_class]
#     else:
#         shap_contrib = shap_values
#     if shap_contrib.ndim == 2:
#         shap_contrib = shap_contrib[0]
#     feature_importances = sorted(
#         zip(df_input.columns, shap_contrib),
#         key=lambda x: abs(x[1]),
#         reverse=True
#     )[:3]
#     shap_explanation = [f"{feat} => contribution SHAP = {float(np.ravel(val)[0]):.3f}" 
#                         for feat, val in feature_importances]
    
#     rules_explanation = []
#     if rules is not None and bins_dict is not None:
#         triggered = apply_association_rules(rules, features_dict, bins_dict)
#         for r in triggered:
#             antecedents = ", ".join(sorted(r["antecedents"]))
#             consequents = ", ".join(sorted(r["consequents"]))
#             rules_explanation.append(
#                 f"Si {antecedents} alors {consequents} (conf: {r['confidence']:.2f}, lift: {r['lift']:.2f})"
#             )
    
#     explanation_text = shap_explanation + rules_explanation
#     result = {
#         "prediction": str(pred),
#         "explanation": explanation_text
#     }
#     return result


import pickle
import pandas as pd
import os

def load_model(model_path=None, rules_path=None, bins_path=None):
    """
    Charge le classifieur basé sur les règles d'association.
    On charge les règles et les bins depuis les fichiers.
    """
    model_data = {}
    if rules_path is not None and os.path.exists(rules_path):
        with open(rules_path, "rb") as f:
            rules = pickle.load(f)
        model_data["rules"] = rules
    else:
        model_data["rules"] = None

    if bins_path is not None and os.path.exists(bins_path):
        with open(bins_path, "rb") as f:
            bins_dict = pickle.load(f)
        model_data["bins"] = bins_dict
    else:
        model_data["bins"] = None

    return model_data

def discretize_value(attribute, value, bins):
    for i in range(len(bins) - 1):
        if bins[i] < value <= bins[i+1]:
            # Utiliser le même format que pd.cut renvoie, par exemple "Age_(20,30]"
            return f"{attribute}_{str(pd.Interval(bins[i], bins[i+1], closed='right'))}"
    return f"{attribute}_{value}"

def transform_instance(instance, bins_dict):
    transformed = {}
    for key, val in instance.items():
        if bins_dict is not None and key in bins_dict and bins_dict[key] is not None:
            transformed[key] = discretize_value(key, val, bins_dict[key])
        else:
            transformed[key] = str(val)
    return transformed

def apply_association_rules(rules, instance, bins_dict):
    transformed_instance = transform_instance(instance, bins_dict)
    instance_items = set(f"{key}_{val}" for key, val in transformed_instance.items())
    print("Instance items :", instance_items)
    
    triggered_rules = []
    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        if antecedents.issubset(instance_items):
            triggered_rules.append({
                "antecedents": antecedents,
                "consequents": rule['consequents'],
                "confidence": rule['confidence'],
                "lift": rule['lift']
            })
            print("Règle déclenchée :", antecedents, "=>", rule['consequents'])
    if not triggered_rules:
        print("Aucune règle déclenchée pour cette instance.")
    return triggered_rules

def predict_with_apriori(model_data, features_dict):
    """
    Utilise uniquement les règles d'association pour classifier une instance.
    Sélectionne la règle avec la confiance maximale et retourne la conséquence (label cible).
    """
    rules = model_data.get("rules")
    bins_dict = model_data.get("bins")
    
    if rules is None or bins_dict is None:
        return {"prediction": "Indisponible", "explanation": ["Pas de règles disponibles"]}
    
    triggered = apply_association_rules(rules, features_dict, bins_dict)
    if not triggered:
        return {"prediction": "Indéterminé", "explanation": ["Aucune règle ne s'applique pour cette instance"]}
    
    best_rule = max(triggered, key=lambda r: r["confidence"])
    predicted_label = None
    for item in best_rule["consequents"]:
        if item.startswith("NObeyesdad_"):
            predicted_label = item.split("_", 1)[1]
            break
    if not predicted_label:
        predicted_label = "Indéterminé"
    
    explanation_text = [f"Si {', '.join(sorted(best_rule['antecedents']))} alors {', '.join(sorted(best_rule['consequents']))} (conf: {best_rule['confidence']:.2f}, lift: {best_rule['lift']:.2f})"]
    
    return {"prediction": predicted_label, "explanation": explanation_text}
