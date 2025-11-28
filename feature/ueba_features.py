# feature/ueba_features.py

import json
import re

def extract_username(record):
    # Try ner-based extraction first
    ents = record.get("entities", "[]")

    try:
        ents = json.loads(ents)
    except:
        ents = []

    for e in ents:
        if e["type"] in ["USER", "PER", "PERSON"]:
            return e["entity"]

    # Regex fallback
    msg = record.get("raw_message", "")
    m = re.search(r"user\s+([a-zA-Z0-9_\-]+)", msg)
    if m:
        return m.group(1)

    return None


def extract_ip(record):
    msg = record.get("raw_message", "")
    m = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", msg)
    return m.group(0) if m else None


def extract_process(record):
    return record.get("process", None)


def build_ueba_features(record):
    """
    Basic UEBA features (more will be added once we have historical context)
    """
    username = extract_username(record)
    ip = extract_ip(record)
    proc = extract_process(record)

    return {
        "username": username,
        "has_username": username is not None,

        "src_ip": ip,
        "has_ip": ip is not None,

        "process_name": proc,
        "has_process": proc is not None,

        # Early UEBA indicators
        "entity_count": sum([
            username is not None,
            ip is not None,
            proc is not None,
        ]),
    }
