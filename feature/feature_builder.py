# feature/feature_builder.py

from feature.time_features import extract_time_features
from feature.statistical_features import extract_statistical_features
from feature.ueba_features import build_ueba_features


def build_features(record):
    """
    Build complete feature set.
    Each extractor is fault-tolerant.
    Missing fields = handled gracefully.
    """

    features = {}

    # Time-based features
    features.update(extract_time_features(record))

    # Statistical features
    features.update(extract_statistical_features(record))

    # UEBA entity features
    features.update(build_ueba_features(record))

    return features
