from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import json
import os
import mne
import shap

# ========= CONFIG ========= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ========= FASTAPI APP ========= #

app = FastAPI(
    title="SleepScope Backend API",
    description="FastAPI backend for ISI, PHQ-9, insomnia subtypes, and PSG-based risk.",
    version="1.0.0",
)

# Allow CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, whitelist your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= ISI & PHQ-9 SCORING ========= #

def calculate_isi_score(responses: List[int]) -> Dict[str, Any]:
    total = int(sum(responses))
    if total <= 7:
        severity = "None"
    elif total <= 14:
        severity = "Mild"
    elif total <= 21:
        severity = "Moderate"
    else:
        severity = "Severe"
    return {"total": total, "severity": severity}

def calculate_phq9_score(responses: List[int]) -> Dict[str, Any]:
    total = int(sum(responses))
    if total <= 4:
        severity = "Minimal"
    elif total <= 9:
        severity = "Mild"
    elif total <= 14:
        severity = "Moderate"
    elif total <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"
    return {"total": total, "severity": severity}

# ========= LOAD SUBTYPE MODEL ========= #

SUBTYPE_FEATURES_PATH = os.path.join(MODELS_DIR, "subtype_features.json")
SUBTYPE_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "subtype_label_map.json")
SUBTYPE_MODEL_PATH = os.path.join(MODELS_DIR, "subtype_model.pkl")
SUBTYPE_SCALER_PATH = os.path.join(MODELS_DIR, "subtype_scaler.pkl")

subtype_model = None
subtype_scaler = None
subtype_features: List[str] = []
subtype_label_map: Dict[int, str] = {}

if os.path.exists(SUBTYPE_MODEL_PATH):
    subtype_model = joblib.load(SUBTYPE_MODEL_PATH)
    subtype_scaler = joblib.load(SUBTYPE_SCALER_PATH)
    with open(SUBTYPE_FEATURES_PATH, "r") as f:
        subtype_features = json.load(f)
    with open(SUBTYPE_LABEL_MAP_PATH, "r") as f:
        subtype_label_map = json.load(f)
    print("[INFO] Subtype model and metadata loaded.")
else:
    print("[WARN] Subtype model not found. /subtype/predict will not work.")


# ========= LOAD PSG MODEL + SHAP ========= #

PSG_MODEL_PATH = os.path.join(MODELS_DIR, "psg_model.pkl")

psg_model = None
psg_explainer = None

if os.path.exists(PSG_MODEL_PATH):
    psg_model = joblib.load(PSG_MODEL_PATH)
    # SHAP explainer will be created lazily (on first request) for speed
    print("[INFO] PSG model loaded.")
else:
    print("[WARN] PSG model not found. /psg/predict will not work.")


# ========= REQUEST SCHEMAS ========= #

class ISIRequest(BaseModel):
    user_id: Optional[str] = None
    responses: List[int]  # length 7

class PHQ9Request(BaseModel):
    user_id: Optional[str] = None
    responses: List[int]  # length 9

class SubtypeRequest(BaseModel):
    # features from unified schema
    sleep_duration: float = 0.0
    sleep_quality: float = 0.0
    sleepiness: float = 0.0
    stress_academic: float = 0.0
    stress_work: float = 0.0
    stress_finance: float = 0.0
    stress_general: float = 0.0
    depression_score: float = 0.0
    depression_label: float = 0.0
    anxiety_score: float = 0.0
    activity: float = 0.0
    steps: float = 0.0
    bmi: float = 0.0
    heart_rate: float = 0.0


# ========= HYPNOGRAM PARSING & PSG FEATURES ========= #

STAGE_MAP = {
    'W': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 3,
    'R': 4,
    'M': -1,
    '?': -1,
}

def read_hypnogram_edf(hyp_path: str) -> np.ndarray:
    """Convert EDF+ hypnogram annotations to 30-sec epoch stage array."""
    ann = mne.read_annotations(hyp_path)
    hyp = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        label = desc.replace("Sleep stage ", "").strip()
        stage = STAGE_MAP.get(label, -1)
        epochs = int(dur / 30)
        hyp.extend([stage] * epochs)
    return np.array(hyp)

def extract_psg_feature_vector(psg_path: str, hyp_path: str) -> Dict[str, float]:
    """Extract the same PSG features you used during training."""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    hyp = read_hypnogram_edf(hyp_path)
    valid = hyp[hyp >= 0]

    if len(valid) == 0:
        raise ValueError("No valid sleep stages found in hypnogram.")

    # Durations
    TST_hours = (np.sum(valid != 0) * 30) / 3600.0
    WASO_minutes = (np.sum(valid == 0) * 30) / 60.0
    N1_minutes = np.sum(valid == 1) * 30 / 60.0
    N2_minutes = np.sum(valid == 2) * 30 / 60.0
    N3_minutes = np.sum(valid == 3) * 30 / 60.0
    REM_minutes = np.sum(valid == 4) * 30 / 60.0

    # Sleep latency (first non-W stage)
    try:
        first_sleep_idx = np.where(valid != 0)[0][0]
        SOL_minutes = first_sleep_idx * 30 / 60.0
    except Exception:
        SOL_minutes = np.nan

    total_time_hours = len(valid) * 30 / 3600.0
    if total_time_hours > 0:
        SE = (TST_hours / total_time_hours) * 100.0
    else:
        SE = np.nan

    return {
        "TST_hours": float(TST_hours),
        "WASO_minutes": float(WASO_minutes),
        "SOL_minutes": float(SOL_minutes if not np.isnan(SOL_minutes) else 0.0),
        "N1_minutes": float(N1_minutes),
        "N2_minutes": float(N2_minutes),
        "N3_minutes": float(N3_minutes),
        "REM_minutes": float(REM_minutes),
        "Sleep_Efficiency": float(SE if not np.isnan(SE) else 0.0),
        "Total_Time_hours": float(total_time_hours),
    }


# ========= ROUTES ========= #

@app.get("/")
def root():
    return {"message": "SleepScope backend is running."}


# --- ISI --- #
@app.post("/isi/predict")
def isi_predict(req: ISIRequest):
    if len(req.responses) != 7:
        raise HTTPException(status_code=400, detail="ISI requires exactly 7 responses.")
    if any((r < 0 or r > 4) for r in req.responses):
        raise HTTPException(status_code=400, detail="ISI responses must be between 0 and 4.")

    result = calculate_isi_score(req.responses)
    return {"success": True, "data": result}


# --- PHQ-9 --- #
@app.post("/phq9/predict")
def phq9_predict(req: PHQ9Request):
    if len(req.responses) != 9:
        raise HTTPException(status_code=400, detail="PHQ-9 requires exactly 9 responses.")
    if any((r < 0 or r > 3) for r in req.responses):
        raise HTTPException(status_code=400, detail="PHQ-9 responses must be between 0 and 3.")

    result = calculate_phq9_score(req.responses)
    return {"success": True, "data": result}


# --- SUBTYPE PREDICTION --- #
@app.post("/subtype/predict")
def subtype_predict(req: SubtypeRequest):
    if subtype_model is None or subtype_scaler is None:
        raise HTTPException(status_code=500, detail="Subtype model not loaded on server.")

    # Build feature row in correct order
    row = []
    feature_dict = req.dict()

    for feat in subtype_features:
        # If feature is present in our conceptual schema, use it; else 0
        # We used df_cluster columns when saving subtype_features.json, so they must match these keys.
        val = feature_dict.get(feat, 0.0)
        row.append(val)

    X = np.array([row])
    X_scaled = subtype_scaler.transform(X)
    cluster_idx = int(subtype_model.predict(X_scaled)[0])
    label = subtype_label_map.get(cluster_idx, f"Cluster {cluster_idx}")

    return {
        "success": True,
        "data": {
            "cluster": cluster_idx,
            "subtype_label": label,
            "features_used": feature_dict,
        },
    }


# --- PSG PREDICTION --- #
@app.post("/psg/predict")
async def psg_predict(
    psg_file: UploadFile = File(...),
    hyp_file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    if psg_model is None:
        raise HTTPException(status_code=500, detail="PSG model not loaded on server.")

    # Save uploaded files temporarily
    tmp_dir = os.path.join(BASE_DIR, "tmp_psg")
    os.makedirs(tmp_dir, exist_ok=True)
    psg_path = os.path.join(tmp_dir, psg_file.filename)
    hyp_path = os.path.join(tmp_dir, hyp_file.filename)

    with open(psg_path, "wb") as f:
        f.write(await psg_file.read())
    with open(hyp_path, "wb") as f:
        f.write(await hyp_file.read())

    # Extract features
    try:
        feat_dict = extract_psg_feature_vector(psg_path, hyp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PSG features: {e}")

    # Build DataFrame in same feature order as during training
    feature_names = [
        "TST_hours",
        "WASO_minutes",
        "SOL_minutes",
        "N1_minutes",
        "N2_minutes",
        "N3_minutes",
        "REM_minutes",
        "Sleep_Efficiency",
        "Total_Time_hours",
    ]
    X = pd.DataFrame([[feat_dict[k] for k in feature_names]], columns=feature_names)

    # Predict insomnia risk (synthetic ISI-like)
    y_pred = float(psg_model.predict(X)[0])

    # SHAP explanation (lazy init)
    global psg_explainer
    if psg_explainer is None:
        psg_explainer = shap.TreeExplainer(psg_model)

    shap_vals = psg_explainer.shap_values(X)
    shap_vals = shap_vals[0]  # single sample

    # Top contributing features
    contrib = sorted(
        [{"feature": name, "shap_value": float(val)} for name, val in zip(feature_names, shap_vals)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "success": True,
        "data": {
            "psg_features": feat_dict,
            "predicted_insomnia_risk": y_pred,
            "shap_contributions": contrib,
        },
    }


# You can add more endpoints here later for:
# - saving user scores to Firestore
# - global correlation
# - dashboard summaries, etc.


# ========= RUN (for local debug) ========= #
# Run with: uvicorn main:app --reload
