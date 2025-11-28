# nlp/normalize.py

import re

def clean_message(msg: str) -> str:
    """
    Normalize logs for semantic processing:
    - lowercasing
    - remove PIDs, numbers, hex
    - remove IPs
    - remove paths
    - remove duplicate spaces
    """

    if msg is None:
        return ""

    text = msg.lower()

    # remove IP addresses
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " IP ", text)

    # remove numbers (PIDs etc.)
    text = re.sub(r"\b\d+\b", " ", text)

    # remove paths
    text = re.sub(r"/[a-zA-Z0-9_\-./]+", " PATH ", text)

    # remove parentheses content
    text = re.sub(r"\(.*?\)", " ", text)

    # clean symbols
    text = re.sub(r"[^a-zA-Z ]+", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
