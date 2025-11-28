# main.py

import argparse
import json

from utils.io_manager import IOManager
from nlp.normalize import clean_message
from nlp.embedder import Embedder
from feature.feature_builder import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_type", default=None, help="es/file")
    parser.add_argument("--out", dest="output_type", default=None, help="es/file")
    parser.add_argument("--config", dest="config_path", default="config.yml")

    args = parser.parse_args()

    # Load IO manager
    io = IOManager(args.config_path)

    # Override config using CLI args
    io.override_config(
        input_type=args.input_type,
        output_type=args.output_type,
        config_path=args.config_path,
    )

    # Read input logs (ES or CSV)
    records = io.read()

    # Load NLP components
    cfg = io.config
    embedder = Embedder(cfg["nlp"]["embedding_model"])

    results = []

    for rec in records:
        msg = str(rec.get("message", ""))
        clean = clean_message(msg)

        # Embedding
        emb = embedder.encode([clean])[0]

        # Base record (NER removed)
        base = {
            "@timestamp": rec.get("@timestamp", ""),
            "hostname": rec.get("hostname", ""),
            "process": rec.get("process", ""),
            "raw_message": msg,
            "clean_message": clean,
            "embedding": json.dumps(emb.tolist()),
            # no NER field
        }

        # Features (NER-based ones become null or defaults)
        feats = build_features(base)
        base.update(feats)

        results.append(base)

    # Output using IO manager
    io.write(results)


if __name__ == "__main__":
    main()
