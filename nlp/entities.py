# nlp/entities.py

import re
import json
from transformers import pipeline


class LogEntityExtractor:

    def __init__(self, ml_model="AaltoNLP/ner-model-log"):
        """
        A hybrid NER system:
        - Regex/entity extraction (primary)
        - Log-pattern extraction
        - ML fallback
        """
        print(f"[NER] Loading ML model (fallback): {ml_model}")
        try:
            self.ner_pipe = pipeline( "ner", model=ml_model, aggregation_strategy="simple" )
        except Exception as e:
            print("[NER] ML model load failed, fallback to regex only.")
            self.ner_pipe = None

    # ------------------------------------------------------------------
    # REGEX EXTRACTORS
    # ------------------------------------------------------------------

    def extract_ips(self, text):
        ipv4 = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
        ipv6 = re.findall(r"\b[a-fA-F0-9:]{2,}\b", text)
        return list(set(ipv4 + ipv6))

    def extract_usernames(self, text):
        patterns = [
            r"user\s+([a-zA-Z0-9_\-]+)",
            r"for\s+([a-zA-Z0-9_\-]+)\s+from",
            r"session opened for user\s+([a-zA-Z0-9_\-]+)",
            r"session closed for user\s+([a-zA-Z0-9_\-]+)",
            r"for\s+invalid user\s+([a-zA-Z0-9_\-]+)"
        ]
        users = []
        for p in patterns:
            users += re.findall(p, text)
        return list(set(users))

    def extract_hostnames(self, text):
        return re.findall(r"\b[a-zA-Z][a-zA-Z0-9\.\-]{2,}\b", text)

    def extract_process(self, record):
        return record.get("process", None)

    def extract_pid(self, text):
        m = re.search(r"\[(\d+)\]", text)
        return int(m.group(1)) if m else None

    def extract_paths(self, text):
        return re.findall(r"/[a-zA-Z0-9_\-./]+", text)

    def extract_ports(self, text):
        ports = re.findall(r"port\s+(\d+)", text)
        rports = re.findall(r"rport\s+(\d+)", text)
        return list(set(ports + rports))

    # ------------------------------------------------------------------
    # LOG PATTERN EXTRACTORS (POWERFUL)
    # ------------------------------------------------------------------

    def extract_patterns(self, text):
        patterns = {}

        # Failed SSH login
        if "Failed password for" in text:
            m = re.search(r"Failed password for (?:invalid user )?(\w+)", text)
            if m:
                patterns["ssh_failed_user"] = m.group(1)

        # Successful SSH login
        if "Accepted password for" in text:
            m = re.search(r"Accepted password for ([\w\-]+)", text)
            if m:
                patterns["ssh_success_user"] = m.group(1)

        # Public key auth
        if "Accepted publickey for" in text:
            m = re.search(r"Accepted publickey for ([\w\-]+)", text)
            if m:
                patterns["ssh_pubkey_user"] = m.group(1)

        return patterns

    # ------------------------------------------------------------------
    # ML FALLBACK
    # ------------------------------------------------------------------

    def extract_ml(self, text):
        if self.ner_pipe is None:
            return []

        try:
            ents = self.ner_pipe(text)
        except:
            return []

        extracted = []
        for e in ents:
            extracted.append({
                "value": e["word"],
                "type": e["entity_group"],
                "score": float(e["score"])
            })
        return extracted

    # ------------------------------------------------------------------
    # FINAL COMBINED EXTRACTION
    # ------------------------------------------------------------------

    def extract(self, record):
        text = record.get("raw_message", "") or record.get("clean_message", "")

        entities = {
            "ips": self.extract_ips(text),
            "usernames": self.extract_usernames(text),
            "hostnames": self.extract_hostnames(text),
            "process": self.extract_process(record),
            "pid": self.extract_pid(text),
            "paths": self.extract_paths(text),
            "ports": self.extract_ports(text),
            "patterns": self.extract_patterns(text),
            "ml": self.extract_ml(text)
        }

        # Filter noisy hostnames
        entities["hostnames"] = [
            h for h in entities["hostnames"]
            if not re.match(r"\d+\.\d+\.\d+\.\d+", h)
        ]

        return entities
