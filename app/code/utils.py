"""
Helpers shared by the Dash pages and the unit tests.

The A1/A2 save/load helpers are unchanged. Everything below the marker is new for A3
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# The saved model contains instances of the LogisticRegression class from
# logistic_regression.py (the Task 1/2 deliverable). Unpickling it needs that module on
# the import path. It lives next to this file inside the Docker image, and two folders up
# at the repo root when running locally from app/code, so both locations are added here.
# This is what lets `python main.py` and the pickle load work without duplicating the module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def save(filename: str, obj: object):
    joblib.dump(obj, filename)


def load(filename: str) -> object:
    return joblib.load(filename)


# ===========================================================================
# A3: loading the price-class model
# ===========================================================================

# the columns the model was trained on, in the order the preprocessor expects
INPUT_COLUMNS = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                 'owner', 'mileage', 'engine', 'max_power', 'seats']
CATEGORICAL_COLUMNS = ['brand', 'fuel', 'seller_type', 'transmission']
NUMERIC_COLUMNS = [c for c in INPUT_COLUMNS if c not in CATEGORICAL_COLUMNS]

# read from the environment so the image does not hard-code the server address
MODEL_NAME = os.environ.get('APP_MODEL_NAME', 'st126956-a3-model')
MODEL_STAGE = os.environ.get('APP_MODEL_STAGE', 'Staging')
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'car_price_classifier.pkl')

# human-readable price bands, filled in when the local bundle is loaded
PRICE_BANDS = None


def build_input_row(**values) -> pd.DataFrame:
    """Build the one-row DataFrame the model expects from whatever the user filled in.

    A blank dropdown arrives as None and a blank number as None/NaN. Both are allowed:
    the preprocessor imputes them with the median / most frequent value it learned during
    training, which is what makes "leave any field empty" work.

    The logged model signature declares every column optional, with the
    numbers as double, so the numeric columns have to be float64 (an int column cannot hold
    NaN) and the categorical ones have to stay object even when their only value is None.
    """
    frame = pd.DataFrame([{c: values.get(c, None) for c in INPUT_COLUMNS}])
    for c in CATEGORICAL_COLUMNS:
        frame[c] = frame[c].astype(object)
    for c in NUMERIC_COLUMNS:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').astype('float64')
    return frame


class LocalClassifier:
    """Wraps the pickled preprocessor + weights so it has the same .predict(DataFrame)
    interface as a model loaded from MLflow. Used when the MLflow server is unreachable."""

    def __init__(self, bundle):
        self.preprocessor = bundle['preprocessor']
        self.model = bundle['model']
        self.price_bins = bundle.get('price_bins')
        self.config = bundle.get('config', {})
        self.metrics = bundle.get('metrics', {})

    def _prepare(self, model_input: pd.DataFrame) -> np.ndarray:
        frame = model_input.copy()
        for c in INPUT_COLUMNS:
            if c not in frame.columns:
                frame[c] = np.nan
        Xp = self.preprocessor.transform(frame[INPUT_COLUMNS])
        return np.concatenate([np.ones((Xp.shape[0], 1)), Xp], axis=1)   # intercept column

    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(model_input))

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare(model_input))


_cache = {}


def load_local_model() -> LocalClassifier:
    """The copy of the model that ships inside the image."""
    global PRICE_BANDS
    if 'local' not in _cache:
        bundle = load(LOCAL_MODEL_PATH)
        PRICE_BANDS = bundle.get('price_bins')
        _cache['local'] = LocalClassifier(bundle)
    return _cache['local']


def load_mlflow_model(stage: str = None):
    """Load <MODEL_NAME> at the given stage from the MLflow model registry.

    Raises if the server cannot be reached - callers decide whether to fall back.
    """
    import mlflow
    stage = stage or MODEL_STAGE
    key = f"mlflow:{stage}"
    if key not in _cache:
        # MLFLOW_TRACKING_URI comes from the container environment, so the server address
        # is never baked into the image
        _cache[key] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{stage}")
    return _cache[key]


def load_model(stage: str = None, prefer_mlflow: bool = True):
    """Return something with .predict(DataFrame) -> array of class labels.

    Tries the MLflow registry first and falls back to the local copy. The fallback is what
    keeps the web app and the CI pipeline working while the class MLflow server is down,
    and it also means the deployed site does not go offline if the server goes away later.
    """
    if prefer_mlflow and os.environ.get('MLFLOW_TRACKING_URI'):
        try:
            model = load_mlflow_model(stage)
            print(f"[utils] loaded {MODEL_NAME} at {stage or MODEL_STAGE} from MLflow")
            return model
        except Exception as exc:
            print(f"[utils] MLflow unavailable ({type(exc).__name__}: {str(exc)[:120]}), "
                  f"using the local model instead")
    return load_local_model()


def class_label(class_index: int) -> str:
    """Turn the predicted class into the price band it stands for."""
    load_local_model()          # makes sure PRICE_BANDS is populated
    names = ['Budget', 'Lower-mid', 'Upper-mid', 'Premium']
    if PRICE_BANDS is None:
        return names[int(class_index)]
    lo, hi = PRICE_BANDS[int(class_index)], PRICE_BANDS[int(class_index) + 1]
    return f"{names[int(class_index)]} ({lo:,.0f} - {hi:,.0f})"
