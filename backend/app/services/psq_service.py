import shap
from app.utils.load_models import psg_model
from app.utils.preprocess import preprocess_psg_features

def analyze_psg(psg_features: dict):
    X = preprocess_psg_features(psg_features)
    
    prediction = float(psg_model.predict(X)[0])
    
    explainer = shap.TreeExplainer(psg_model)
    shap_values = explainer.shap_values(X)[0].tolist()

    return prediction, shap_values
