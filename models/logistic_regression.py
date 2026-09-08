from os import pipe

from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
from sklearn.preprocessing import MinMaxScaler
import ipdb


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    # torch tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def train_logistic_regression(
    X_train, y_train, X_test, y_test,
    feature_cols_names=None,
    *,
    random_state=42,
    no_scaling=False,
    model_name=None,
    penalty="l1",          # "l1" or "elasticnet" or "l2"
    regularization=0.1,                 # inverse regularization strength
    l1_ratio=None,         # only used if penalty="elasticnet"
):
    # --- Ensure NumPy and 2D (flatten) ---
    X_train = to_numpy(X_train)
    X_test  = to_numpy(X_test)
    y_train = to_numpy(y_train).ravel()
    y_test  = to_numpy(y_test).ravel()

    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    X_train = X_train.reshape(n_train, -1)
    X_test  = X_test.reshape(n_test,  -1)

    # --- Sanity: both classes must be present in y_train ---
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train has a single class. Use a stratified split so both classes appear in training.")

    # --- Output path ---
    out_path = None
    if model_name:
        out_path = Path(model_name)
        out_path.mkdir(parents=True, exist_ok=True)

    steps = []
    if not no_scaling:
        steps.append(("scaler", StandardScaler()))
        print("Using feature scaling.")
    steps.append(("clf", LogisticRegression(
        penalty=penalty,
        C=regularization,
        solver="liblinear",
        max_iter=10000,
        random_state=random_state,
    )))
    pipe = Pipeline(steps)

    # --- Fit ---
    pipe.fit(X_train, y_train)

    # --- Scores (probabilities for PR-AUC etc.) ---
    y_scores_train = pipe.predict_proba(X_train)[:, 1]
    y_scores_test  = pipe.predict_proba(X_test)[:, 1]
    # save as csv


    X_train_scaled = pipe.named_steps["scaler"].transform(X_train)
    explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_train_scaled)
    # distribution check
    return y_scores_train, y_scores_test, pipe, explainer

