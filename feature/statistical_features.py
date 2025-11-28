# feature/statistical_features.py

import json
import numpy as np

def extract_statistical_features(record):
    emb_raw = record.get("embedding")
    out = {
        "embedding_norm": None,
        "message_length": len(record.get("clean_message", "")),
    }

    if emb_raw:
        try:
            vec = np.array(json.loads(emb_raw))
            out["embedding_norm"] = float(np.linalg.norm(vec))
        except:
            pass

    return out
