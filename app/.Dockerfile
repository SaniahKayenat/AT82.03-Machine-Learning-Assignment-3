FROM python:3.11-slim

WORKDIR /root/code

# Install dependencies first so Docker can cache this layer across rebuilds.
# NOTE: the build context is the REPOSITORY ROOT (see docker-compose.yaml), not app/,
# because the image also needs logistic_regression.py which lives at the root.
COPY app/code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The application code (main.py, pages/, models/, utils.py, tests)
COPY app/code /root/code

# A3: the model classes. The saved model contains instances of these, so the module has to
# be importable for joblib/cloudpickle to load it back.
COPY logistic_regression.py /root/code/logistic_regression.py

EXPOSE 8050

# Runs the Dash server. HOST/PORT are read from the environment (see docker-compose.yaml).
CMD ["python3", "main.py"]
