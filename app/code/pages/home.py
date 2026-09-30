import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")


def model_card(title, body, href, button_text, color, outline=False):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H5(title, className="card-title"),
                html.P(body, className="card-text"),
                dbc.Button(button_text, href=href, color=color, outline=outline),
            ])
        ),
        md=4,
        className="mb-3",
    )


layout = dbc.Container(
    [
        html.H1("Predict which budget class a car should be sold for!"),
        html.P(
            "The company uses this tool to get a quick estimate "
            "of what price a used car should have.",
            className="lead",
        ),
        html.Hr(),
        html.H4("Three models to choose from"),
        html.P(
            "All three pages ask for the same information about the car. They differ in what "
            "they do with it and in how the model behind them was built."
        ),
        dbc.Row([
            model_card(
                "Old model (A1)",
                "A scikit-learn Random Forest with 500 trees, chosen in Assignment 1. It predicts "
                "an exact price. Slightly the most accurate of the three, but it is a black box "
                "and the saved model is 57 MB.",
                "/predict", "Use the old model", "primary", outline=True),
            model_card(
                "New model (A2)",
                "A linear regression I wrote from scratch in Assignment 2, trained with stochastic "
                "gradient descent on polynomial features. Also predicts an exact price, almost as "
                "accurately, but you can read every coefficient and it is a few kilobytes.",
                "/predict-new", "Use the new model", "success"),
            model_card(
                "Price class (A3)",
                "A multinomial logistic regression written from scratch in Assignment 3. Instead of "
                "a number it gives a price band - budget, lower-mid, upper-mid or premium - which is "
                "closer to how people actually compare used cars.",
                "/predict-class", "Predict a price class", "warning"),
        ]),
        html.Hr(),
        html.H4("How it works"),
        html.Ol([
            html.Li("Pick one of the three prediction pages above and fill in what you know about the car."),
            html.Li("If you don't know some technical specs, just leave those fields blank. They "
                    "will be estimated for you automatically."),
            html.Li("You can leave all fields empty if you want and it will still predict for you. "
                    "But ideally, you should fill as many fields as you can."),
            html.Li("Click the predict button to get an instant answer."),
        ]),
        html.P(
            "All three models were trained on the same 6,800+ real used-car listings.",
            className="text-muted",
        ),
    ],
    fluid=True,
    className="py-4",
)
