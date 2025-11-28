# feature/time_features.py

from datetime import datetime

def extract_time_features(record):
    """
    Extracts hour, weekday, weekend flags.
    Works even if timestamp is missing or invalid.
    """
    ts = record.get("@timestamp")
    out = {
        "hour": None,
        "weekday": None,
        "is_weekend": None,
    }

    if not ts:
        return out

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        out["hour"] = dt.hour
        out["weekday"] = dt.weekday()
        out["is_weekend"] = dt.weekday() >= 5
    except:
        pass

    return out
