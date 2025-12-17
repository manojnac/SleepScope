import numpy as np
import mne

def read_hypnogram(path):
    ann = mne.read_annotations(path)
    STAGE_MAP = {'W':0,'1':1,'2':2,'3':3,'4':3,'R':4,'M':-1,'?':-1}
    
    hyp = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        label = desc.replace("Sleep stage ", "").strip()
        stage = STAGE_MAP.get(label, -1)
        epochs = int(dur / 30)
        hyp.extend([stage] * epochs)

    return np.array(hyp)


def preprocess_psg_features(psg_file, hyp_file=None):
    """
    Extracts the same 10 features used during model training.
    """
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

    if hyp_file is None:
        raise ValueError("Hypnogram file is required for correct feature extraction.")

    hyp = read_hypnogram(hyp_file)
    valid = hyp[hyp >= 0]

    # Durations
    TST = (np.sum(valid != 0) * 30) / 3600
    WASO = (np.sum(valid == 0) * 30) / 60
    N1 = np.sum(valid == 1) * 30 / 60
    N2 = np.sum(valid == 2) * 30 / 60
    N3 = np.sum(valid == 3) * 30 / 60
    REM = np.sum(valid == 4) * 30 / 60

    total_time = len(valid) * 30 / 3600
    SE = (TST / total_time) * 100

    # Ratios (must match training order)
    N1_ratio = N1 / (TST * 60)
    N2_ratio = N2 / (TST * 60)
    N3_ratio = N3 / (TST * 60)
    REM_ratio = REM / (TST * 60)

    WASO_rate = WASO / (total_time * 60)
    SOL = next((i for i, x in enumerate(valid) if x == 1), -1)
    SOL_minutes = SOL * 30 / 60
    SOL_rate = SOL_minutes / total_time

    Fragmentation = (N1 + WASO) / (TST * 60)

    # Return EXACT 10 feature values in the correct order:
    return {
        "TST_hours": TST,
        "WASO_rate": WASO_rate,
        "SOL_rate": SOL_rate,
        "N1_ratio": N1_ratio,
        "N2_ratio": N2_ratio,
        "N3_ratio": N3_ratio,
        "REM_ratio": REM_ratio,
        "Sleep_Efficiency": SE,
        "Fragmentation": Fragmentation,
        "Total_Time_hours": total_time
    }
