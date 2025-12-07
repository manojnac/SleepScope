from app.utils.scoring import get_severity, ISI_SEVERITY

def compute_isi(responses):
    total = sum(responses)
    severity = get_severity(total, ISI_SEVERITY)
    return total, severity
