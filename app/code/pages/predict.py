"""
Prediction page: a form for the car's details, a Predict button, and a
result box.
"""

import os

import dash
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State

from utils import load

dash.register_page(__name__, path="/predict")

# ---------------------------------------------------------------------------
# Load the trained pipeline 
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "car_price_model.pkl")
model = load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Dropdown option lists -- taken from the categories seen during training.
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


def field(label_text, component, required=False):
    """A labeled form field. I kept all inputs optional as per the requirement of the assignment"""
    note = (
        html.Span(" *", className="text-danger")
        if required
        else html.Span("", className="text-muted")
    )
    return dbc.Col(
        [dbc.Label([label_text, note]), component],
        md=6,
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
form = dbc.Form(
    [
        html.H5(" Fill in as many details as you know"),
        dbc.Row(
            [
                field(
                    "Brand",
                    dcc.Dropdown(
                        id="input-brand",
                        options=[{"label": b, "value": b} for b in BRAND_OPTIONS],
                        placeholder="Select a brand",
                    ),
                ),
                field(
                    "Year of manufacture",
                    dbc.Input(id="input-year", type="number", min=1983, max=2026, step=1,
                               placeholder="e.g. 2018"),
                ),
            ]
        ),
        dbc.Row(
            [
                field(
                    "Kilometers driven",
                    dbc.Input(id="input-km", type="number", min=0, step=1,
                               placeholder="e.g. 45000"),
                ),
                field(
                    "Fuel type",
                    dcc.Dropdown(
                        id="input-fuel",
                        options=[{"label": f, "value": f} for f in FUEL_OPTIONS],
                        placeholder="Select fuel type",
                    ),
                ),
            ]
        ),
        dbc.Row(
            [
                field(
                    "Seller type",
                    dcc.Dropdown(
                        id="input-seller",
                        options=[{"label": s, "value": s} for s in SELLER_TYPE_OPTIONS],
                        placeholder="Select seller type",
                    ),
                ),
                field(
                    "Transmission",
                    dcc.Dropdown(
                        id="input-transmission",
                        options=[{"label": t, "value": t} for t in TRANSMISSION_OPTIONS],
                        placeholder="Select transmission",
                    ),
                ),
            ]
        ),
        dbc.Row(
            [
                field(
                    "Ownership history",
                    dcc.Dropdown(
                        id="input-owner",
                        options=OWNER_OPTIONS,
                        placeholder="Select ownership history",
                    ),
                ),
            ]
        ),
        html.H5("Technical specs", className="mt-3"),
        dbc.Row(
            [
                field(
                    "Mileage (kmpl)",
                    dbc.Input(id="input-mileage", type="number", min=0, step=0.1,
                               placeholder="e.g. 21.5"),
                    required=False,
                ),
                field(
                    "Engine size (CC)",
                    dbc.Input(id="input-engine", type="number", min=0, step=1,
                               placeholder="e.g. 1197"),
                    required=False,
                ),
            ]
        ),
        dbc.Row(
            [
                field(
                    "Max power (bhp)",
                    dbc.Input(id="input-power", type="number", min=0, step=0.1,
                               placeholder="e.g. 82"),
                    required=False,
                ),
                field(
                    "Seats",
                    dbc.Input(id="input-seats", type="number", min=2, max=14, step=1,
                               placeholder="e.g. 5"),
                    required=False,
                ),
            ]
        ),
        dbc.Button("Predict Price", id="predict-button", color="primary",
                   n_clicks=0, className="mt-2"),
        html.Div(id="prediction-output", className="mt-4"),
    ],
    className="mb-3",
)

layout = dbc.Container(
    [
        html.H2("Predict a Car's Selling Price"),
        form,
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def predict_price(brand=None, year=None, km_driven=None, fuel=None, seller_type=None,
                   transmission=None, owner=None, mileage=None, engine=None,
                   max_power=None, seats=None):
    """
    Predict a car's selling price from raw user input.

    Any optional numeric argument left as None is converted to NaN and
    imputed automatically by the saved pipeline.

    Returns
    -------
    float : predicted selling price in the original currency units.
    """
    row = pd.DataFrame([{
        "brand": brand if brand is not None else np.nan,
        "year": year if year is not None else np.nan,
        "km_driven": km_driven if km_driven is not None else np.nan,
        "fuel": fuel if fuel is not None else np.nan,
        "seller_type": seller_type if seller_type is not None else np.nan,
        "transmission": transmission if transmission is not None else np.nan,
        "owner": owner if owner is not None else np.nan,
        "mileage": mileage if mileage is not None else np.nan,
        "engine": engine if engine is not None else np.nan,
        "max_power": max_power if max_power is not None else np.nan,
        "seats": seats if seats is not None else np.nan,
    }])
    log_price = model.predict(row)[0]
    return float(np.exp(log_price))


# ---------------------------------------------------------------------------
# Callback: gather form input -> validate -> predict -> render result
# ---------------------------------------------------------------------------
@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("input-brand", "value"),
    State("input-year", "value"),
    State("input-km", "value"),
    State("input-fuel", "value"),
    State("input-seller", "value"),
    State("input-transmission", "value"),
    State("input-owner", "value"),
    State("input-mileage", "value"),
    State("input-engine", "value"),
    State("input-power", "value"),
    State("input-seats", "value"),
    prevent_initial_call=True,
)
def on_predict_click(n_clicks, brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, power, seats):
    all_fields = {
        "Brand": brand, "Year": year, "Kilometers driven": km_driven,
        "Fuel type": fuel, "Seller type": seller_type,
        "Transmission": transmission, "Ownership history": owner,
        "Mileage": mileage, "Engine": engine, "Max power": power, "Seats": seats,
    }
    blank_fields = [name for name, value in all_fields.items() if value in (None, "")]

    price = predict_price(
        brand=brand, year=year, km_driven=km_driven, fuel=fuel,
        seller_type=seller_type, transmission=transmission, owner=owner,
        mileage=mileage, engine=engine, max_power=power, seats=seats,
    )

    message = f"Estimated selling price:  {price:,.0f} rupees."
    if blank_fields:
        message += f" (auto-filled: {', '.join(blank_fields)})"
    return dbc.Alert(message, color="success")
