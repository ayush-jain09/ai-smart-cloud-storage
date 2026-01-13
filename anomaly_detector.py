from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest


def compute_user_risk(logs, files):
    """
    Returns a UI-friendly risk object using anomaly detection.
    No external dataset is required – the model learns from the user's own activity.
    """

    now = datetime.utcnow()
    start = now - timedelta(days=20)

    days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(21)]
    day_index = {d: i for i, d in enumerate(days)}

    actions = ["upload", "download", "search", "delete", "login"]
    X = np.zeros((21, len(actions)), dtype=float)

    for log in logs:
        day = log.created_at.strftime("%Y-%m-%d")
        if day in day_index and log.action in actions:
            X[day_index[day], actions.index(log.action)] += 1

    # Not enough activity yet
    if X.sum() < 8:
        return {
            "level": "Low",
            "label": "Low confidence (insufficient data)",
            "score": 0.15,
            "why": "Use the system for a few days to enable behavior-based AI security insights."
        }

    model = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42
    )

    model.fit(X)

    recent = X[-1].reshape(1, -1)
    anomaly_score = -model.score_samples(recent)[0]

    score = min(max((anomaly_score - 0.35) / 0.9, 0.0), 1.0)

    if score < 0.35:
        level = "Low"
        label = "Normal activity"
    elif score < 0.7:
        level = "Medium"
        label = "Unusual activity"
    else:
        level = "High"
        label = "Potential suspicious behavior"

    return {
        "level": level,
        "label": label,
        "score": score,
        "why": "AI detected deviation from your normal file-access behavior."
    }
