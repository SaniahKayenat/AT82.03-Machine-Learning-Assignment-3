"""
A3 prediction page: predicts which *price band* a car falls into, instead of a price.
"""

import dash
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State

from utils import build_input_row, load_model, class_label

dash.register_page(__name__, path="/predict-class", name="Price Class (A3)")

# loaded once when the page is imported, not on every prediction
model = load_model()

# ---------------------------------------------------------------------------
# Dropdown options - the categories seen during training
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
    """A labelled form field. Every input is optional, as on the earlier pages."""
    return dbc.Col([dbc.Label(label_text), component], md=6, className="mb-3")


intro = dbc.Alert(
    [
        html.H5("About this page", className="alert-heading"),
        html.P(
            "This page answers a different question from the other two. Instead of guessing an "
            "exact number, it sorts the car into one of four price bands, which is closer to how "
            "people actually think about a used car: is it a budget car, a mid-range one, or a "
            "premium one?"
        ),
        html.P(
            "Behind it is a multinomial logistic regression implemented from scratch for "
            "Assignment 3 with softmax, cross-entropy loss and gradient descent, and an optional "
            "ridge penalty. The settings were chosen by comparing 27 combinations of gradient "
            "descent method, learning rate and penalty strength, all tracked in MLflow. It gets "
            "about 73 % of the test cars into the right band, against 25 % for guessing."
        ),
        html.Hr(),
        html.P(
            "Fill in whatever you know and press Predict. Anything left blank is filled in "
            "automatically from the training data, so even an empty form returns an answer.",
            className="mb-0",
        ),
    ],
    color="info",
)

form = dbc.Form(
    [
        html.H5("Fill in details"),
        dbc.Row([
            field("Brand", dcc.Dropdown(id="cls-brand",
                  options=[{"label": b, "value": b} for b in BRAND_OPTIONS],
                  placeholder="Select a brand")),
            field("Year of manufacture", dbc.Input(id="cls-year", type="number",
                  min=1983, max=2026, step=1, placeholder="e.g. 2018")),
        ]),
        dbc.Row([
            field("Kilometers driven", dbc.Input(id="cls-km", type="number",
                  min=0, step=1, placeholder="e.g. 45000")),
            field("Fuel type", dcc.Dropdown(id="cls-fuel",
                  options=[{"label": f, "value": f} for f in FUEL_OPTIONS],
                  placeholder="Select fuel type")),
        ]),
        dbc.Row([
            field("Seller type", dcc.Dropdown(id="cls-seller",
                  options=[{"label": s, "value": s} for s in SELLER_TYPE_OPTIONS],
                  placeholder="Select seller type")),
            field("Transmission", dcc.Dropdown(id="cls-transmission",
                  options=[{"label": t, "value": t} for t in TRANSMISSION_OPTIONS],
                  placeholder="Select transmission")),
        ]),
        dbc.Row([
            field("Ownership history", dcc.Dropdown(id="cls-owner",
                  options=OWNER_OPTIONS, placeholder="Select ownership history")),
        ]),
        html.H5("Technical specs", className="mt-3"),
        dbc.Row([
            field("Mileage (kmpl)", dbc.Input(id="cls-mileage", type="number",
                  min=0, step=0.1, placeholder="e.g. 21.5")),
            field("Engine size (CC)", dbc.Input(id="cls-engine", type="number",
                  min=0, step=1, placeholder="e.g. 1197")),
        ]),
        dbc.Row([
            field("Max power (bhp)", dbc.Input(id="cls-power", type="number",
                  min=0, step=0.1, placeholder="e.g. 82")),
            field("Seats", dbc.Input(id="cls-seats", type="number",
                  min=2, max=14, step=1, placeholder="e.g. 5")),
        ]),
        dbc.Button("Predict Price Class", id="cls-button", color="warning",
                   n_clicks=0, className="mt-2"),
        html.Div(id="cls-output", className="mt-4"),
    ],
    className="mb-3",
)

layout = dbc.Container(
    [
        html.H2("Predict a Car's Price Class"),
        intro,
        form,
    ],
    fluid=True,
    className="py-4",
)


def predict_class(brand=None, year=None, km_driven=None, fuel=None, seller_type=None,
                  transmission=None, owner=None, mileage=None, engine=None,
                  max_power=None, seats=None):
    """Predict the price class (0-3) for one car. Kept separate from the callback so the
    unit tests can call it without a running server."""
    row = build_input_row(brand=brand, year=year, km_driven=km_driven, fuel=fuel,
                          seller_type=seller_type, transmission=transmission, owner=owner,
                          mileage=mileage, engine=engine, max_power=max_power, seats=seats)
    return int(np.asarray(model.predict(row))[0])


@callback(
    Output("cls-output", "children"),
    Input("cls-button", "n_clicks"),
    State("cls-brand", "value"),
    State("cls-year", "value"),
    State("cls-km", "value"),
    State("cls-fuel", "value"),
    State("cls-seller", "value"),
    State("cls-transmission", "value"),
    State("cls-owner", "value"),
    State("cls-mileage", "value"),
    State("cls-engine", "value"),
    State("cls-power", "value"),
    State("cls-seats", "value"),
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

    try:
        predicted = predict_class(
            brand=brand, year=year, km_driven=km_driven, fuel=fuel,
            seller_type=seller_type, transmission=transmission, owner=owner,
            mileage=mileage, engine=engine, max_power=power, seats=seats,
        )
    except Exception as exc:            # show the problem instead of a blank page
        return dbc.Alert(f"Could not predict: {exc}", color="danger")

    message = f"Predicted price class: {predicted} - {class_label(predicted)}"
    if blank_fields:
        message += f"  (auto-filled: {', '.join(blank_fields)})"
    return dbc.Alert(message, color="success")
