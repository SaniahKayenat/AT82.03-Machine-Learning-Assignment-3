# A3: Predicting Car Price (Classification)

AT82.03 Machine Learning — Assignment 3 (st126956)

The same used-car data as [A1](https://github.com/SaniahKayenat/ML-Coding-Assignment-1) and A2, but treated as a
**4-class classification** problem and solved with a multinomial logistic regression written from scratch.

**Live site:** http://web-st126956.ml.brain.cs.ait.ac.th/ (also on https)

## Folder structure

```
.
├── A3_car_price_classification.ipynb   # Tasks 1, 2 and Task 3 objectives 1-2
├── logistic_regression.py              # the model classes (Tasks 1 and 2)
├── Cars.csv
├── .github/
│   └── workflows/
│       └── build-test.yml              # Task 3 objective 3: CI/CD
└── app/
    ├── .Dockerfile
    ├── docker-compose.yaml             # local build / CI
    ├── docker-compose_forDeploy.yaml   # the file that runs on ml-brain
    └── code/
        ├── main.py                     # navbar + page container
        ├── utils.py                    # model loading + input building
        ├── conftest.py                 # puts the repo root on the import path for pytest
        ├── requirements.txt
        ├── test_model.py               # the unit tests CI runs
        ├── models/
        │   ├── car_price_model.pkl         # A1 Random Forest
        │   ├── car_price_model_a2.pkl      # A2 from-scratch regression
        │   └── car_price_classifier.pkl    # A3 from-scratch classifier
        └── pages/
            ├── home.py
            ├── predict.py              # A1 model, exact price
            ├── predict_new.py          # A2 model, exact price
            └── predict_class.py        # A3 model, price class
```

## Task 1 — Classification

`selling_price` is discretised into 4 bands with **`pd.qcut`**, giving roughly 25 % of the cars per class.
`pd.cut` was rejected after testing it: because the price distribution is heavily right-skewed (skew = 4.19),
equal-width bins put 6,726 of the 6,806 cars (98.8 %) into class 0 and exactly **one** car into class 3. A model
could then score 98.8 % accuracy while learning nothing, and most of the metrics would be undefined.

The `LogisticRegression` class from `02 - Multinomial Logistic Regression.ipynb` was extended with the whole
classification report, built from a single confusion matrix:

| Added                                                  | What it returns                                 |
| ------------------------------------------------------ | ----------------------------------------------- |
| `confusion_matrix`, `support`                          | the (4, 4) matrix, and the true count per class |
| `accuracy`                                             | correct predictions / all predictions           |
| `precision`, `recall`, `f1_score`                      | one score per class                             |
| `macro_precision`, `macro_recall`, `macro_f1`          | plain average over the 4 classes                |
| `weighted_precision`, `weighted_recall`, `weighted_f1` | average weighted by support                     |
| `classification_report`                                | the same table sklearn prints                   |

**Results:** test accuracy **0.7254** against a 0.25 baseline. Classes 0 and 3 (cheapest and most expensive)
reach f1 ≈ 0.84 and 0.81; the two middle classes only reach 0.61 and 0.63, because they have neighbours on both
sides and the boundary between them is an artefact of the quantile split. Almost every error is in a cell next to the diagonal.

**What `support` means:** the number of samples that truly belong to each class in `y_true` — the row sums of
the confusion matrix, not the number of predictions made for that class. It says how much each per-class score
can be trusted, and it is the weighting factor behind the weighted averages. Because `qcut` made the four
supports nearly equal, the macro and weighted rows are almost identical here; with `pd.cut` they would have
diverged sharply.

## Task 2 — Ridge Logistic Regression

The penalty follows the pattern in `03 - Regularization.ipynb`: a small object with `__call__` (added to the
loss) and `derivation` (added to the gradient), which the model plugs in. Choosing whether to regularise is just
a matter of which object is passed:

```python
LogisticRegression(k, n)                                # no penalty (default)
LogisticRegression(k, n, regularization=RidgePenalty(0.01))
RidgeLogisticRegression(k, n, l=0.01)                   # same thing, shorter
```

The **intercept row is not penalised** as it only sets the base rate of each class, so shrinking it would bias
every prediction instead of simplifying the model.

`||W||` falls monotonically from 9.22 to 0.23 as λ goes from 0 to 1. The best λ is 0.001 (test accuracy 0.7276
against 0.7254 unpenalised); anything from 0.01 up underfits badly. The gain is small, which matches A2: with
~5,400 training rows and 40 features there is not much overfitting for a penalty to fix.

As a sanity check, sklearn's `LogisticRegression` scores 0.7386–0.7416 on the same split. Landing about a point
below it is the expected result for plain gradient descent against L-BFGS.

## Task 3 — Deployment

### Objective 1: logging to the MLflow server

27 configurations (3 gradient-descent methods × 3 learning rates × 3 λ values), each logging its parameters and
the seven classification metrics from Task 1. The dataset is **not** logged.

Best configuration: `batch`, α = 0.1, no penalty — test accuracy 0.7269.

> Per the announcement of 21 September,
> `mlflow.ml.brain.cs.ait.ac.th` was not running, which affects objectives 1 and 2 . The notebook
> tries the server first and falls back to a local `mlflow.db` if it cannot be reached, so it always runs. Set
> `USE_SERVER = True` (it is the default) and re-run from the MLflow section once the server is back; nothing
> else has to change.

### Objective 2: the model registry

The best model is logged as an **`mlflow.pyfunc` model that carries its own preprocessor**, so `predict()` takes
a raw DataFrame of car details and returns the price class. `code_paths=["logistic_regression.py"]` packages the
model classes with it, which is what lets the CI container load it without any training code.

It is registered as **`st126956-a3-model`** and moved to **Staging**.

One thing that needed care: the model **signature is declared by hand** rather than inferred. Left to infer,
MLflow types `year`, `km_driven` and `owner` as `long`, and a long column cannot hold a missing value — so the
moment a user submitted the form with one of those blank, MLflow rejected the request before the model ran. That
would have broken the "leave any field empty" behaviour carried over from A1/A2. Declaring every column
`double`/`string` and `required=False` fixes it.

### Objective 3: CI/CD

`.github/workflows/build-test.yml` runs on every push:

1. build the image and start the container;
2. run `pytest test_model.py` **inside** the container;
3. only on `main`, and only if the tests passed, log in to Docker Hub and push the image;
4. then SSH to ml-brain through the bazooka jump host and `docker compose pull && up -d`.

The deploy job declares `needs: build-test`, which is what makes the deployment conditional: if a test fails,
GitHub never starts the deploy job, so a broken commit cannot reach the server.

**The two required unit tests** are in `app/code/test_model.py`:

- `test_model_takes_expected_input` — the model accepts the 11-column DataFrame the Dash callback builds, with
  the dtypes the signature requires.
- `test_model_output_shape` — one input row gives exactly one prediction, _n_ rows give _n_, and the values are
  class labels in 0–3.

Five extra tests cover the blank-field cases, an unknown brand, and a sanity check that a recent premium car is
not ranked below an old budget one.

**Model loading.** `utils.load_model()` tries the MLflow registry first and falls back to the copy inside the
image. That keeps both the site and CI working while the MLflow server is down, and means the deployed site does
not go offline if the server disappears later.

## Running it

```bash
# notebook
pip install -r app/code/requirements.txt
jupyter lab A3_car_price_classification.ipynb

# the app, locally
cd app/code && python main.py           # http://127.0.0.1:8050

# the app, in Docker (from the repo root)
docker compose -f app/docker-compose.yaml up --build

# the tests
cd app/code && pytest test_model.py -v
```

## GitHub settings the workflow needs

| Kind     | Name                  | Value                                                                               |
| -------- | --------------------- | ----------------------------------------------------------------------------------- |
| Secret   | `DOCKERHUB_USERNAME`  | `psyduckait`                                                                        |
| Secret   | `DOCKERHUB_TOKEN`     | an access token from Docker Hub → Account Settings → Personal access tokens         |
| Secret   | `USERNAME`            | `st126956`                                                                          |
| Secret   | `KEY`                 | the **private** SSH key that matches the public key submitted to the course         |
| Secret   | `MLFLOW_TRACKING_URI` | `http://mlflow.ml.brain.cs.ait.ac.th/` (may be left empty while the server is down) |
| Secret   | `APP_MODEL_NAME`      | `st126956-a3-model`                                                                 |
| Variable | `HOST`                | `ml.brain.cs.ait.ac.th`                                                             |
| Variable | `PROXY_HOST`          | `bazooka.cs.ait.ac.th`                                                              |
