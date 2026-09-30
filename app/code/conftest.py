"""
pytest configuration.

The saved model contains instances of the LogisticRegression class from
`logistic_regression.py`, which lives at the repository root (it is the Task 1/2
deliverable). Unpickling it therefore needs that module on the import path.

Inside the Docker image the file is copied next to the app, so this is a no-op there.
When running `pytest` locally from `app/code` it is what makes the import work, so the
module is not duplicated in two places in the repo.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

for path in (HERE, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
