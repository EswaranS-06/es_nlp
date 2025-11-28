# main.py

import argparse
import json
import pandas as pd

from utils.io_manager import IOManager
from nlp.normalize import clean_message
from nlp.embedder import Embedder
from feature.feature_builder import build_features

from ml.ml_pipeline import MLPipeline
from ml.model_store import ModelStore


def main():
    parser = argparse.ArgumentParser()

    # Input/output control
    parser.add_argument("--in", dest="input_type", help="es/file")
    parser.add_argument("--out", dest="output_type", help="es/file")
    parser.add_argument("--config", dest="config_path", default="config.yml")

    # ML actions
    parser.add_argument("--train-ml", action="store_true")
    parser.add_argument("--predict-ml", action="store_true")

    # Auto-labeling for LightGBM
    parser.add_argument("--auto-label", action="store_true")
    parser.add_argument("--label-threshold", type=float, default=0.8)

    args = parser.parse_args()

    # IO Manager
    io = IOManager(args.config_path)
    io.override_config(
        input_type=args.input_type,
        output_type=args.output_type,
        config_path=args.config_path
    )

    # ===============================================================
    # STEP 1 — READ RAW LOGS
    # ===============================================================
    raw_logs = io.read()
    print(f"[MAIN] Loaded {len(raw_logs)} unprocessed logs from input index.")

    # NLP Embedding model
    cfg = io.config
    embedder = Embedder(cfg["nlp"]["embedding_model"])

    processed_logs = []

    # ===============================================================
    # STEP 2 — CLEAN → EMBED → FEATURES
    # ===============================================================
    print("[MAIN] Processing logs (cleaning, embedding, features)...")

    for rec in raw_logs:
        msg = str(rec.get("message", rec.get("raw_message", "")))
        clean = clean_message(msg)

        # Embedding (used for ML only)
        embedding_vec = embedder.encode([clean])[0]

        enriched = {
            "_id": rec.get("_id"),  # keep ES ID for reference only
            "@timestamp": rec.get("@timestamp", ""),
            "hostname": rec.get("hostname", ""),
            "process": rec.get("process", ""),
            "raw_message": msg,
            "clean_message": clean,

            # Store embedding internally (DO NOT write this to ES)
            "embedding": embedding_vec.tolist(),
        }

        # Feature extraction
        feats = build_features(enriched)
        enriched.update(feats)

        processed_logs.append(enriched)

    print(f"[MAIN] Preprocessing complete for {len(processed_logs)} logs.")

    # Save processed logs if using file mode
    if args.input_type == "file":
        io.write(processed_logs)

    df_struct = pd.DataFrame(processed_logs)

    # ===============================================================
    # STEP 3 — TRAIN ML MODELS
    # ===============================================================
    if args.train_ml:
        print("\n[ML] Training anomaly detection models...")

        ml = MLPipeline()
        store = ModelStore("models")

        metrics = ml.train(
            df=df_struct,
            auto_label=args.auto_label,
            label_threshold=args.label_threshold,
        )

        store.save(ml, metadata=metrics)
        print("[ML] Training complete.")
        print("[ML] Stored model metadata:", metrics)

    # ===============================================================
    # STEP 4 — PREDICT ML SCORES
    # ===============================================================
    if args.predict_ml:
        print("\n[ML] Performing anomaly scoring...")

        store = ModelStore("models")
        data = store.load()

        ml = MLPipeline()

        # Restore IsolationForest + scaler
        ml.isolation_forest = data.get("isolation_forest")
        ml.scaler = data.get("scaler")

        # Restore LightGBM
        if "lgb_booster" in data:
            booster = data["lgb_booster"]

            class LGBWrapper:
                def __init__(self, booster):
                    self.booster_ = booster

                def predict(self, X):
                    return self.booster_.predict(X)

            ml.lgb_model = LGBWrapper(booster)

        if "lgb_meta" in data:
            ml.lgb_train_features = data["lgb_meta"]["feature_names"]

        scored = ml.predict(df_struct)

        # Save to CSV always
        scored.to_csv("anomaly_scored_logs.csv", index=False)
        print("[ML] Saved anomaly_scored_logs.csv")

        # ===============================================================
        # STEP 5 — WRITE ENRICHED LOGS TO NEW ES INDEX (NO EMBEDDINGS)
        # ===============================================================
        if args.output_type == "es":
            print("[MAIN] Writing enriched logs to Elasticsearch index WITHOUT embeddings...")

            output_records = scored.to_dict(orient="records")

            # Remove embedding (to avoid ES errors)
            for rec in output_records:
                rec.pop("embedding", None)

            io.write_to_es(output_records)

            print("[MAIN] Successfully wrote enriched logs into output index.")


if __name__ == "__main__":
    main()
