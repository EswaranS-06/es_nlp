# utils/io_manager.py

import yaml
import pandas as pd
from elasticsearch import Elasticsearch
import json


class IOManager:

    def __init__(self, config_path="config.yml"):
        self.config = self.load_config(config_path)
        self.es = None

    # -----------------------------------------------------
    # Load configuration YAML
    # -----------------------------------------------------
    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------
    # Apply CLI overrides to config
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
    # Read from Elasticsearch
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
                results.append(h["_source"])

            resp = self.es.scroll(scroll_id=scroll_id, scroll=scroll)

        print(f"[IO] Retrieved {len(results)} records from ES.")
        return results

    # -----------------------------------------------------
    # Read from CSV
    # -----------------------------------------------------
    def read_from_csv(self):
        path = self.config["input"]["file"]
        print(f"[IO] Reading from CSV: {path}")
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    # -----------------------------------------------------
    # Write to CSV
    # -----------------------------------------------------
    def write_to_csv(self, records):
        path = self.config["output"]["file"]
        print(f"[IO] Writing to CSV: {path}")
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)

    # -----------------------------------------------------
    # Write to Elasticsearch
    # -----------------------------------------------------
    def write_to_es(self, records):
        index = self.config["elasticsearch"]["output_index"]
        if self.es is None:
            self.connect_es()

        print(f"[IO] Writing results to ES index: {index}")

        for rec in records:
            self.es.index(index=index, document=rec)

        print("[IO] Finished writing to ES.")

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def read(self):
        input_type = self.config["input"]["type"]
        if input_type == "es":
            return self.read_from_es()
        elif input_type == "file":
            return self.read_from_csv()
        else:
            raise ValueError("Invalid input type. Use 'es' or 'file'.")

    def write(self, records):
        output_type = self.config["output"]["type"]
        if output_type == "es":
            self.write_to_es(records)
        elif output_type == "file":
            self.write_to_csv(records)
        else:
            raise ValueError("Invalid output type. Use 'es' or 'file'.")
