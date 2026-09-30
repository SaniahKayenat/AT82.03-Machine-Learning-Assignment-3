"""
Car Price Prediction - web app

Entry point for the multi-page Dash app.

Run locally with:   python3 main.py
Run in Docker with: docker compose up --build   (more in /README.md)
"""

import os

import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # exposed in case you want to run this behind gunicorn later

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Old model (A1)", href="/predict")),
        dbc.NavItem(dbc.NavLink("New model (A2)", href="/predict-new")),
        dbc.NavItem(dbc.NavLink("Price class (A3)", href="/predict-class")),
    ],
    brand="Chaky's Car Price Predictor",
    brand_href="/",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    navbar,
    dash.page_container,
])

# HOST and PORT come from the environment (see docker-compose.yaml); Dash falls back to
# 127.0.0.1:8050 when they are not set. DEBUG is off by default so the deployed site does
# not expose the debugger; set DEBUG=1 when developing locally.
if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8050)),
        debug=os.getenv("DEBUG", "0") == "1",
    )
