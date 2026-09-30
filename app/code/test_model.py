"""
Unit tests for the A3 price-class model.

The assignment asks for two unit tests:
    1. the model takes the expected input
    2. the output of the model has the expected shape

Those are `test_model_takes_expected_input` and `test_model_output_shape` below. The other
tests are extra checks that were useful while building the app, mostly around the "any field
may be left blank" behaviour that broke once already when MLflow inferred an integer schema.

Run with:   pytest test_model.py -v
"""

import numpy as np
import pandas as pd
import pytest

from utils import (INPUT_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
                   build_input_row, load_model)

# One fully specified car, used by most of the tests.
SAMPLE_CAR = dict(brand='Maruti', year=2018.0, km_driven=45000.0, fuel='Petrol',
                  seller_type='Individual', transmission='Manual', owner=1.0,
                  mileage=21.5, engine=1197.0, max_power=82.0, seats=5.0)

N_CLASSES = 4


@pytest.fixture(scope='module')
def model():
    """The model the web app serves: MLflow registry if reachable, local copy otherwise."""
    return load_model()


# ---------------------------------------------------------------------------
# Required test 1: the model takes the expected input
# ---------------------------------------------------------------------------
def test_model_takes_expected_input(model):
    """The model must accept a DataFrame with the 11 raw car columns and return a
    prediction without raising. This is the exact object the Dash callback builds."""
    X = build_input_row(**SAMPLE_CAR)

    # the input really is the shape the model was trained on
    assert list(X.columns) == INPUT_COLUMNS, f"expected {INPUT_COLUMNS}, got {list(X.columns)}"
    assert X.shape == (1, 11), f"expected one row of 11 columns, got {X.shape}"
    # dtypes have to match the logged signature, otherwise MLflow rejects the request
    for c in NUMERIC_COLUMNS:
        assert X[c].dtype == np.float64, f"{c} should be float64, got {X[c].dtype}"
    for c in CATEGORICAL_COLUMNS:
        assert X[c].dtype == object, f"{c} should be object, got {X[c].dtype}"

    prediction = model.predict(X)
    assert prediction is not None


# ---------------------------------------------------------------------------
# Required test 2: the output has the expected shape
# ---------------------------------------------------------------------------
def test_model_output_shape(model):
    """One input row must give exactly one prediction, and n rows must give n
    predictions - all of them valid class labels in 0..3."""
    single = model.predict(build_input_row(**SAMPLE_CAR))
    assert np.asarray(single).shape == (1,), \
        f"expecting the shape to be (1,) but got {np.asarray(single).shape}"

    # and it has to stay one-prediction-per-row for a batch
    batch = pd.concat([build_input_row(**SAMPLE_CAR) for _ in range(5)], ignore_index=True)
    many = np.asarray(model.predict(batch))
    assert many.shape == (5,), f"expecting the shape to be (5,) but got {many.shape}"

    # the values are class labels, not prices
    assert set(np.unique(many)).issubset(set(range(N_CLASSES))), \
        f"predictions must be class labels 0..{N_CLASSES - 1}, got {np.unique(many)}"


# ---------------------------------------------------------------------------
# Extra checks
# ---------------------------------------------------------------------------
def test_model_loads(model):
    """The model object exists and exposes predict()."""
    assert model is not None
    assert hasattr(model, 'predict')


def test_all_fields_blank(model):
    """The form may be submitted completely empty: every value is imputed.
    This is the case that failed when MLflow inferred `year` as a long column."""
    prediction = model.predict(build_input_row())
    assert np.asarray(prediction).shape == (1,)
    assert int(prediction[0]) in range(N_CLASSES)


def test_partially_filled_input(model):
    """A realistic half-filled form still predicts."""
    prediction = model.predict(build_input_row(brand='Hyundai', year=2015.0, km_driven=80000.0))
    assert int(prediction[0]) in range(N_CLASSES)


def test_unknown_brand_is_handled(model):
    """A brand that was not in the training data must not crash the app;
    the encoder maps it to all-zero dummies, i.e. the baseline brand."""
    prediction = model.predict(build_input_row(**{**SAMPLE_CAR, 'brand': 'Tesla'}))
    assert int(prediction[0]) in range(N_CLASSES)


def test_expensive_car_ranks_above_cheap_car(model):
    """A sanity check on the learned ordering: a recent, powerful car should not be put in
    a lower price band than an old, low-powered one."""
    cheap = model.predict(build_input_row(
        brand='Maruti', year=2010.0, km_driven=120000.0, fuel='Petrol',
        seller_type='Individual', transmission='Manual', owner=3.0,
        mileage=19.0, engine=796.0, max_power=47.0, seats=5.0))[0]
    pricey = model.predict(build_input_row(
        brand='BMW', year=2020.0, km_driven=15000.0, fuel='Diesel',
        seller_type='Dealer', transmission='Automatic', owner=1.0,
        mileage=15.0, engine=2998.0, max_power=250.0, seats=5.0))[0]
    assert int(pricey) >= int(cheap), f"cheap car -> class {cheap}, expensive car -> class {pricey}"
