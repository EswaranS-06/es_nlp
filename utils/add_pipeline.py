import os
import json
import requests
import yaml

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def add_pipeline(pipeline_file, pipeline_name, es_host):
    url = f"{es_host}/_ingest/pipeline/{pipeline_name}"

    with open(pipeline_file, "r") as f:
        data = json.load(f)

    print(f"[INFO] Uploading pipeline: {pipeline_name}")

    res = requests.put(url, json=data)
    print(res.status_code, res.text)

def load_all_pipelines():
    cfg = load_config()
    es_host = cfg["elasticsearch"]["host"]

    pattern_path = "es_patterns"

    for filename in os.listdir(pattern_path):
        if filename.endswith(".json"):
            pipeline_file = os.path.join(pattern_path, filename)
            pipeline_name = filename.replace(".json", "")
            add_pipeline(pipeline_file, pipeline_name, es_host)

if __name__ == "__main__":
    load_all_pipelines()
