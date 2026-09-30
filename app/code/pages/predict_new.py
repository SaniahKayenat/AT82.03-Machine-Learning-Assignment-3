"""
New prediction page (A2 model).

"""

import os

import dash
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State

from utils import load

dash.register_page(__name__, path="/predict-new", name="Predict Price (New Model)")

# ---------------------------------------------------------------------------
# Load the A2 bundle
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "car_price_model_a2.pkl")
bundle = load(MODEL_PATH)
preprocessor = bundle["preprocessor"]   # the ColumnTransformer fitted in the notebook
theta = bundle["theta"]                 # weights, theta[0] is the intercept

# the same column order the preprocessor was fitted on
INPUT_COLUMNS = ["brand", "year", "km_driven", "fuel", "seller_type", "transmission",
                 "owner", "mileage", "engine", "max_power", "seats"]

# ---------------------------------------------------------------------------
# Dropdown options (same categories as the old page)
# ---------------------------------------------------------------------------
BRAND_OPTIONS = [
    "Ambassador", "Ashok", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun", "Fiat",
    "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep", "Kia", "Land",
    "Lexus", "MG", "Mahindra", "Maruti", "Mercedes-Benz", "Mitsubishi", "Nissan",
    "Opel", "Peugeot", "Renault", "Skoda", "Tata", "Toyota", "Volkswagen", "Volvo",
]
FUEL_OPTIONS = ["Diesel", "Petrol"]
SELLER_TYPE_OPTIONS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSION_OPTIONS = ["Manual", "Automatic"]
OWNER_OPTIONS = [
    {"label": "First Owner", "value": 1},
    {"label": "Second Owner", "value": 2},
    {"label": "Third Owner", "value": 3},
    {"label": "Fourth & Above Owner", "value": 4},
]


def field(label_text, component):
    """A labeled form field. Every input is optional, as in the old page."""
    return dbc.Col([dbc.Label(label_text), component], md=6, className="mb-3")


# ---------------------------------------------------------------------------
# Message explaining the new model (required by the assignment)
# ---------------------------------------------------------------------------
intro = dbc.Alert(
    [
        html.H5("About this page", className="alert-heading"),
        html.P(
            "This page uses the model written from scratch in Assignment 2. It is a linear "
            "regression trained with stochastic gradient descent on polynomial features, "
            "with Xavier initialisation and a learning rate of 0.001. The settings were "
            "chosen out of 144 combinations compared with 5-fold cross-validation."
        ),
        html.P("It is better than the old model in several ways!!"),
        html.Ul(
            [
                html.Li(
                    "It is transparent. Every feature has one coefficient I can read, so "
                    "the price can be explained The old Random "
                    "Forest is 500 trees and cannot be read this way."
                ),
                html.Li(
                    "It is far smaller and faster, a few kilobytes of weights and a single "
                    "dot product per prediction, instead of a 57 MB forest."
                ),
                html.Li(
                    "Its accuracy is very close to the old model (test r\u00b2 0.89 against 0.92 "
                    "on the log-price scale), so you give up very little for that."
                ),
            ]
        ),
        html.Hr(),
        html.P(
            "How to use it: fill in whatever you know about the car and press Predict Price. "
            "Any field you leave blank is filled in automatically with the median (numbers) or "
            "the most common value (categories) from the training data, so you can even submit "
            "an empty form.",
            className="mb-0",
        ),
    ],
    color="info",
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
form = dbc.Form(
    [
        html.H5("Fill in details"),
        dbc.Row([
            field("Brand", dcc.Dropdown(id="new-input-brand",
                  options=[{"label": b, "value": b} for b in BRAND_OPTIONS],
                  placeholder="Select a brand")),
            field("Year of manufacture", dbc.Input(id="new-input-year", type="number",
                  min=1983, max=2026, step=1, placeholder="e.g. 2018")),
        ]),
        dbc.Row([
            field("Kilometers driven", dbc.Input(id="new-input-km", type="number",
                  min=0, step=1, placeholder="e.g. 45000")),
            field("Fuel type", dcc.Dropdown(id="new-input-fuel",
                  options=[{"label": f, "value": f} for f in FUEL_OPTIONS],
                  placeholder="Select fuel type")),
        ]),
        dbc.Row([
            field("Seller type", dcc.Dropdown(id="new-input-seller",
                  options=[{"label": s, "value": s} for s in SELLER_TYPE_OPTIONS],
                  placeholder="Select seller type")),
            field("Transmission", dcc.Dropdown(id="new-input-transmission",
                  options=[{"label": t, "value": t} for t in TRANSMISSION_OPTIONS],
                  placeholder="Select transmission")),
        ]),
        dbc.Row([
            field("Ownership history", dcc.Dropdown(id="new-input-owner",
                  options=OWNER_OPTIONS, placeholder="Select ownership history")),
        ]),
        html.H5("Technical specs", className="mt-3"),
        dbc.Row([
            field("Mileage (kmpl)", dbc.Input(id="new-input-mileage", type="number",
                  min=0, step=0.1, placeholder="e.g. 21.5")),
            field("Engine size (CC)", dbc.Input(id="new-input-engine", type="number",
                  min=0, step=1, placeholder="e.g. 1197")),
        ]),
        dbc.Row([
            field("Max power (bhp)", dbc.Input(id="new-input-power", type="number",
                  min=0, step=0.1, placeholder="e.g. 82")),
            field("Seats", dbc.Input(id="new-input-seats", type="number",
                  min=2, max=14, step=1, placeholder="e.g. 5")),
        ]),
        dbc.Button("Predict Price", id="new-predict-button", color="success",
                   n_clicks=0, className="mt-2"),
        html.Div(id="new-prediction-output", className="mt-4"),
    ],
    className="mb-3",
)

layout = dbc.Container(
    [
        html.H2("Predict a Car's Selling Price (New Model)"),
        intro,
        form,
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def predict_price_new(brand=None, year=None, km_driven=None, fuel=None, seller_type=None,
                      transmission=None, owner=None, mileage=None, engine=None,
                      max_power=None, seats=None):
    """
    Predict a car's selling price with the A2 from-scratch model.

    Anything left as None becomes NaN and is imputed by the saved preprocessor,
    exactly as it was during training.

    Returns
    -------
    float : predicted selling price in the original currency units.
    """
    values = {"brand": brand, "year": year, "km_driven": km_driven, "fuel": fuel,
              "seller_type": seller_type, "transmission": transmission, "owner": owner,
              "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats}
    row = pd.DataFrame([{c: (values[c] if values[c] is not None else np.nan)
                         for c in INPUT_COLUMNS}])

    # impute -> polynomial terms -> scale -> one-hot, the same pipeline used in training
    X = preprocessor.transform(row)
    # add the intercept column, because theta[0] is the bias
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # X is one row, so the product has one element ([0] is needed on numpy 2)
    log_price = float((X @ theta)[0])   # the model predicts log(selling_price)
    return float(np.exp(log_price))   # back to rupees


# ---------------------------------------------------------------------------
# Callback: read the form -> predict -> show the result
# ---------------------------------------------------------------------------
@callback(
    Output("new-prediction-output", "children"),
    Input("new-predict-button", "n_clicks"),
    State("new-input-brand", "value"),
    State("new-input-year", "value"),
    State("new-input-km", "value"),
    State("new-input-fuel", "value"),
    State("new-input-seller", "value"),
    State("new-input-transmission", "value"),
    State("new-input-owner", "value"),
    State("new-input-mileage", "value"),
    State("new-input-engine", "value"),
    State("new-input-power", "value"),
    State("new-input-seats", "value"),
    prevent_initial_call=True,
)
def on_new_predict_click(n_clicks, brand, year, km_driven, fuel, seller_type,
                         transmission, owner, mileage, engine, power, seats):
    all_fields = {
        "Brand": brand, "Year": year, "Kilometers driven": km_driven,
        "Fuel type": fuel, "Seller type": seller_type,
        "Transmission": transmission, "Ownership history": owner,
        "Mileage": mileage, "Engine": engine, "Max power": power, "Seats": seats,
    }
    blank_fields = [name for name, value in all_fields.items() if value in (None, "")]

    try:
        price = predict_price_new(
            brand=brand, year=year, km_driven=km_driven, fuel=fuel,
            seller_type=seller_type, transmission=transmission, owner=owner,
            mileage=mileage, engine=engine, max_power=power, seats=seats,
        )
    except Exception as exc:                      # show the problem instead of a blank page
        return dbc.Alert(f"Could not predict: {exc}", color="danger")

    message = f"Estimated selling price:  {price:,.0f} rupees."
    if blank_fields:
        message += f" (auto-filled: {', '.join(blank_fields)})"
    return dbc.Alert(message, color="success")
