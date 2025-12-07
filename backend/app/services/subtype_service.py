from app.utils.load_models import subtype_model, subtype_scaler, subtype_features, subtype_label_map
from app.utils.preprocess import prepare_subtype_vector

def classify_subtype(user_input: dict):
    vector = prepare_subtype_vector(user_input, subtype_features)
    vector_scaled = subtype_scaler.transform(vector)

    cluster = int(subtype_model.predict(vector_scaled)[0])
    subtype_label = subtype_label_map.get(str(cluster), "Unknown Subtype")

    return cluster, subtype_label
