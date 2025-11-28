import argparse
import os
import yaml
import requests
import json
from tqdm import tqdm

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def send_to_es(raw_line, cfg, pipeline_name):
    es_url = (
        f"{cfg['elasticsearch']['host']}/"
        f"{cfg['elasticsearch']['index']}/_doc"
        f"?pipeline={pipeline_name}"
    )

    doc = {"raw_log": raw_line}

    res = requests.post(
        es_url,
        data=json.dumps(doc),
        headers={"Content-Type": "application/json"}
    )

    return res.status_code in (200, 201)

def ingest_file(filepath, cfg, pipeline, success, failed):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    print(f"\n[INFO] Ingesting file: {filepath}")
    for line in tqdm(lines, desc="Ingesting", unit="line"):
        line = line.strip()
        if not line:
            continue

        if send_to_es(line, cfg, pipeline):
            success[0] += 1
        else:
            failed[0] += 1

def ingest_folder(folder, cfg, pipeline):
    success = [0]
    failed = [0]

    print(f"[INFO] Reading folder: {folder}")

    for file in os.listdir(folder):
        if file.endswith(".log") or file.endswith(".txt"):
            ingest_file(os.path.join(folder, file), cfg, pipeline, success, failed)

    print("\n============================")
    print("      INGEST SUMMARY")
    print("============================")
    print(f"Successful: {success[0]}")
    print(f"Failed:     {failed[0]}")
    print("============================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="logs/")
    parser.add_argument("--file", default="")
    parser.add_argument("--pipeline", default="syslog_pipeline")

    args = parser.parse_args()
    cfg = load_config()

    if args.file:
        success = [0]
        failed = [0]
        ingest_file(args.file, cfg, args.pipeline, success, failed)
        print(f"\nSuccessful: {success[0]}  Failed: {failed[0]}")
    else:
        ingest_folder(args.folder, cfg, args.pipeline)

if __name__ == "__main__":
    main()
