# utils/io_manager.py

import yaml
import pandas as pd
from elasticsearch import Elasticsearch, helpers


class IOManager:

    def __init__(self, config_path="config.yml"):
        self.config = self.load_config(config_path)
        self.es = None

        # Fields that indicate the log is already enriched by ML
        self.ml_fields = [
            "iso_score",
            "lgbm_score",
            "adwin_flag",
            "fusion_score",
            "is_anomaly",
            "embedding"   # If embedding exists → skip processing
        ]

    # -----------------------------------------------------
    # Load YAML configuration
    # -----------------------------------------------------
    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------
    # Override CLI arguments
    # -----------------------------------------------------
    def override_config(self, input_type, output_type, config_path=None):
        if config_path:
            self.config = self.load_config(config_path)

        if input_type:
            self.config["input"]["type"] = input_type
        if output_type:
            self.config["output"]["type"] = output_type

    # -----------------------------------------------------
    # Connect to Elasticsearch
    # -----------------------------------------------------
    def connect_es(self):
        es_host = self.config["elasticsearch"]["host"]
        print(f"[IO] Connecting to Elasticsearch at {es_host}")
        self.es = Elasticsearch(es_host)

    # -----------------------------------------------------
    # Skip if ML fields exist
    # -----------------------------------------------------
    def _skip_if_processed(self, doc):
        return any(f in doc for f in self.ml_fields)

    # -----------------------------------------------------
    # Read raw logs from Elasticsearch (input_index)
    # -----------------------------------------------------
    def read_from_es(self):
        index = self.config["elasticsearch"]["input_index"]
        size = self.config["elasticsearch"]["size"]
        scroll = self.config["elasticsearch"]["scroll_timeout"]

        if self.es is None:
            self.connect_es()

        print(f"[IO] Reading from ES index: {index}")

        results = []
        resp = self.es.search(
            index=index,
            scroll=scroll,
            size=size,
            body={"query": {"match_all": {}}}
        )

        scroll_id = resp["_scroll_id"]

        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break

            for h in hits:
                src = h["_source"]
                src["_id"] = h["_id"]   # Keep ES ID for reference

                if self._skip_if_processed(src):
                    continue

                results.append(src)

            resp = self.es.scroll(scroll_id=scroll_id, scroll=scroll)

        print(f"[IO] Retrieved {len(results)} unprocessed logs from ES.")
        return results

    # -----------------------------------------------------
    # Read logs from CSV
    # -----------------------------------------------------
    def read_from_csv(self):
        path = self.config["input"]["file"]
        print(f"[IO] Reading from CSV: {path}")
        df = pd.read_csv(path)

        raw = df.to_dict(orient="records")
        filtered = [r for r in raw if not self._skip_if_processed(r)]

        print(f"[IO] Loaded {len(filtered)} fresh logs (skipped {len(raw)-len(filtered)} processed logs).")
        return filtered

    # -----------------------------------------------------
    # Write enriched logs to new Elasticsearch index
    # -----------------------------------------------------
    def write_to_es(self, records):
        index = self.config["elasticsearch"]["output_index"]

        if self.es is None:
            self.connect_es()

        print(f"[IO] Writing {len(records)} enriched logs to ES index: {index}")

        actions = []

        for rec in records:
            doc = rec.copy()

            # Ensure _id is removed (ES rejects it inside _source)
            doc.pop("_id", None)

            # Embedding MUST NOT be sent to ES (skip)
            doc.pop("embedding", None)

            # Fix timestamp issues
            ts = doc.get("@timestamp", None)
            if ts in ("", None, "null", "NaT"):
                doc["@timestamp"] = "1970-01-01T00:00:00Z"   # Fallback valid date


            actions.append({
                "_op_type": "index",
                "_index": index,
                "_source": doc
            })

        try:
            helpers.bulk(self.es, actions)
        except Exception as e:
            from pprint import pprint
            print("\n\n========== ELASTICSEARCH BULK ERROR DETAILS ==========")
            if hasattr(e, "errors"):
                pprint(e.errors)         # ⭐ SHOWS THE REAL CAUSE
            else:
                print(str(e))
            print("=====================================================\n\n")
            raise

        print("[IO] Successfully wrote enriched logs to ES.")

    # -----------------------------------------------------
    # Write enriched logs to a CSV file
    # -----------------------------------------------------
    def write_to_csv(self, records):
        path = self.config["output"]["file"]
        print(f"[IO] Writing enriched logs to CSV: {path}")
        pd.DataFrame(records).to_csv(path, index=False)

    # -----------------------------------------------------
    # Public read() entrypoint
    # -----------------------------------------------------
    def read(self):
        t = self.config["input"]["type"]

        if t == "es":
            return self.read_from_es()
        elif t == "file":
            return self.read_from_csv()
        else:
            raise ValueError("Invalid input type. use 'es' or 'file'")

    # -----------------------------------------------------
    # Public write() entrypoint
    # -----------------------------------------------------
    def write(self, records):
        t = self.config["output"]["type"]

        if t == "es":
            self.write_to_es(records)
        elif t == "file":
            self.write_to_csv(records)
        else:
            raise ValueError("Invalid output type. use 'es' or 'file'")
