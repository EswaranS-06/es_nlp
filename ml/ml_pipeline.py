# ml/ml_pipeline.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from river.drift import ADWIN


class MLPipeline:
    def __init__(self):
        self.isolation_forest = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.adwin = ADWIN()
        self.lgb_train_features = None

        self.weights = {
            "isolation": 0.4,
            "lgbm": 0.5,
            "adwin": 0.1
        }

    # ============================================================
    # FEATURE PREPARATION
    # ============================================================
    def _prepare_features(self, df, feature_cols=None, fit=False):
        exclude = ["timestamp", "@timestamp", "raw_message",
                   "clean_message", "message", "label", "embedding"]

        if feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if c not in exclude and np.issubdtype(df[c].dtype, np.number)
            ]

        X = df[feature_cols].fillna(0).astype(float).values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, feature_cols

    # ============================================================
    # AUTO LABELING
    # ============================================================
    def _auto_label(self, df, X, threshold=0.8):
        iso_raw = self.isolation_forest.decision_function(X)
        iso_score = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

        df["label"] = (iso_score >= threshold).astype(int)
        print(f"[Auto-Label] Applied threshold {threshold} → {df['label'].sum()} anomalies")

        return df

    # ============================================================
    # TRAINING PIPELINE
    # ============================================================
    def train(self, df, auto_label=True, label_threshold=0.8):
        print("\n[TRAIN] Starting ML training...")

        X, feature_cols = self._prepare_features(df, fit=True)
        self.lgb_train_features = feature_cols

        print("[TRAIN] Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination="auto",
            n_jobs=-1,
            random_state=42
        ).fit(X)

        has_label = "label" in df.columns

        if not has_label and auto_label:
            print("[TRAIN] Auto-labeling enabled...")
            df = self._auto_label(df, X, threshold=label_threshold)
            has_label = True

        if not has_label:
            print("[TRAIN] No labels available → LGBM skipped.")
            self.lgb_model = None
            return {"cv_auc_mean": None, "cv_pr_mean": None}

        normal = df[df.label == 0]
        anomaly = df[df.label == 1]

        if len(anomaly) == 0:
            print("[WARN] No anomalies after auto-label → LGBM skipped.")
            self.lgb_model = None
            return {"cv_auc_mean": None, "cv_pr_mean": None}

        normal_sampled = normal.sample(min(len(normal), len(anomaly) * 4), random_state=42)
        df_bal = pd.concat([normal_sampled, anomaly]).sample(frac=1, random_state=42)

        X_bal, _ = self._prepare_features(df_bal, feature_cols=feature_cols, fit=False)
        y_bal = df_bal["label"].astype(int).values

        print("[TRAIN] Running LightGBM CV...")
        tscv = TimeSeriesSplit(n_splits=5)

        aucs, prs = [], []

        for train_idx, val_idx in tscv.split(X_bal):
            X_train, X_val = X_bal[train_idx], X_bal[val_idx]
            y_train, y_val = y_bal[train_idx], y_bal[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                {"objective": "binary", "metric": "auc", "verbosity": -1},
                train_data,
                valid_sets=[valid_data]
            )

            y_pred = model.predict(X_val)

            aucs.append(roc_auc_score(y_val, y_pred))
            prs.append(average_precision_score(y_val, y_pred))

        cv_auc = float(np.mean(aucs))
        cv_pr = float(np.mean(prs))

        print(f"[TRAIN] CV-AUC = {cv_auc:.4f}")
        print(f"[TRAIN] CV-PR  = {cv_pr:.4f}")

        print("[TRAIN] Training final LightGBM model...")
        full_data = lgb.Dataset(X_bal, label=y_bal)
        self.lgb_model = lgb.train(
            {"objective": "binary", "metric": "auc"},
            full_data
        )
        self.lgb_model.booster_ = self.lgb_model

        return {"cv_auc_mean": cv_auc, "cv_pr_mean": cv_pr}

    # ============================================================
    # PREDICTION PIPELINE
    # ============================================================
    def predict(self, df):
        print("\n[PREDICT] Running ML inference...")

        df_out = df.copy()
        X, _ = self._prepare_features(df, feature_cols=self.lgb_train_features, fit=False)

        iso_raw = self.isolation_forest.decision_function(X)
        iso_score = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)
        df_out["iso_score"] = iso_score

        if self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict(X)
            lgb_score = (lgb_pred - lgb_pred.min()) / (lgb_pred.max() - lgb_pred.min() + 1e-9)
        else:
            lgb_score = np.zeros(len(df))

        df_out["lgbm_score"] = lgb_score

        ad_flags = []
        for v in iso_score:
            self.adwin.update(v)
            ad_flags.append(1 if self.adwin.drift_detected else 0)

        df_out["adwin_flag"] = ad_flags

        w = self.weights
        df_out["fusion_score"] = (
            w["isolation"] * df_out["iso_score"] +
            w["lgbm"] * df_out["lgbm_score"] +
            w["adwin"] * df_out["adwin_flag"]
        )

        df_out["is_anomaly"] = (df_out["fusion_score"] >= 0.5).astype(int)

        print("[PREDICT] Inference complete.")
        return df_out
