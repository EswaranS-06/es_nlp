import os
import joblib
import json
import lightgbm as lgb

class ModelStore:
    def __init__(self, path="models"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save(self, pipeline, metadata=None):
        joblib.dump(pipeline.isolation_forest, os.path.join(self.path, "isolation_forest.joblib"))

        if pipeline.scaler is not None:
            joblib.dump(pipeline.scaler, os.path.join(self.path, "scaler.joblib"))

        if pipeline.lgb_model is not None:
            pipeline.lgb_model.booster_.save_model(os.path.join(self.path, "lightgbm.txt"))
            joblib.dump(
                {"feature_names": pipeline.lgb_train_features},
                os.path.join(self.path, "lgb_meta.joblib")
            )

        if metadata:
            with open(os.path.join(self.path, "metadata.json"), "w") as f:
                json.dump(metadata, f)

    def load(self):
        ms = {}
        if os.path.exists(os.path.join(self.path, "isolation_forest.joblib")):
            ms["isolation_forest"] = joblib.load(os.path.join(self.path, "isolation_forest.joblib"))

        if os.path.exists(os.path.join(self.path, "scaler.joblib")):
            ms["scaler"] = joblib.load(os.path.join(self.path, "scaler.joblib"))

        if os.path.exists(os.path.join(self.path, "lightgbm.txt")):
            booster = lgb.Booster(model_file=os.path.join(self.path, "lightgbm.txt"))
            ms["lgb_booster"] = booster

        if os.path.exists(os.path.join(self.path, "lgb_meta.joblib")):
            ms["lgb_meta"] = joblib.load(os.path.join(self.path, "lgb_meta.joblib"))

        return ms
