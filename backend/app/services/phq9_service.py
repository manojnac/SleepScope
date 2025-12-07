from app.utils.scoring import get_severity, PHQ9_SEVERITY

def compute_phq9(responses):
    total = sum(responses)
    severity = get_severity(total, PHQ9_SEVERITY)
    return total, severity
