from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pickle
import streamlit as st


# Centralised paths so we avoid hard-coded absolutes.
PROJECT_ROOT = Path(__file__).resolve().parent
# Data and model currently live one level above the project root (C:\\Users\\HP).
DATA_ROOT = PROJECT_ROOT.parents[1]

MODEL_PATH = DATA_ROOT / "ventureguard_model.pkl"
DATA_PATH = DATA_ROOT / "big_startup_secsees_dataset.csv"

FEATURE_COLUMNS = ["funding_total_usd", "funding_rounds"]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pre-trained RandomForest model used for VentureGuard AI."""
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data(show_spinner=False)
def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Load the startup dataset and prepare:
    - original dataframe with a binary success label
    - numeric feature matrix used by the model
    - median values for each feature (for imputation and explanations)
    """
    df = pd.read_csv(DATA_PATH)

    # Binary outcome aligned with the training notebook
    df["success"] = df["status"].apply(
        lambda x: 1 if x in ["operating", "acquired"] else 0
    )

    features = df[FEATURE_COLUMNS].copy()

    # Mirror the training-time preprocessing
    features.replace("-", pd.NA, inplace=True)
    features["funding_total_usd"] = pd.to_numeric(
        features["funding_total_usd"], errors="coerce"
    )
    features["funding_rounds"] = pd.to_numeric(
        features["funding_rounds"], errors="coerce"
    )

    medians = features.median(numeric_only=True).to_dict()
    features = features.fillna(medians)

    return df, features, medians


def get_failure_probability(
    total_funding_usd: float, funding_rounds: int
) -> Tuple[float, float]:
    """
    Run a single prediction and return:
    - failure probability in [0, 1]
    - success probability in [0, 1]
    """
    model = load_model()
    _, _, medians = load_dataset()

    data = pd.DataFrame(
        [
            {
                "funding_total_usd": total_funding_usd,
                "funding_rounds": funding_rounds,
            }
        ]
    )

    # Apply the same style of imputation used during training
    for col in FEATURE_COLUMNS:
        if pd.isna(data.loc[0, col]):
            data.loc[0, col] = medians.get(col, 0.0)

    proba = model.predict_proba(data)[0]

    # Class 0 => failure, Class 1 => success (as per training label)
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        idx_failure = classes.index(0)
        idx_success = classes.index(1)
    except ValueError:
        # Fallback: assume first column is failure, second is success
        idx_failure, idx_success = 0, 1

    failure_prob = float(proba[idx_failure])
    success_prob = float(proba[idx_success])

    return failure_prob, success_prob


def bucket_risk_level(failure_probability: float) -> str:
    """
    Map a failure probability in [0, 1] into a discrete VC-style label.
    """
    if failure_probability < 0.33:
        return "Low Risk"
    if failure_probability < 0.66:
        return "Medium Risk"
    return "High Risk"


def describe_relative_position(total_funding_usd: float, funding_rounds: int) -> str:
    """
    Generate a short qualitative explanation based on how the startup
    compares to the historical dataset along key dimensions.
    """
    _, _, medians = load_dataset()

    median_funding = medians.get("funding_total_usd", 0.0)
    median_rounds = medians.get("funding_rounds", 1.0)

    funding_ratio = total_funding_usd / median_funding if median_funding else 1.0
    rounds_ratio = funding_rounds / median_rounds if median_rounds else 1.0

    funding_msg = (
        "well below what similar surviving companies raised historically"
        if funding_ratio < 0.5
        else "broadly in line with typical funding levels"
        if funding_ratio < 1.5
        else "meaningfully above the median for historical survivors"
    )

    rounds_msg = (
        "early in its funding journey"
        if rounds_ratio <= 1.0
        else "at a mid-stage in terms of funding rounds"
        if rounds_ratio <= 3.0
        else "already quite seasoned in terms of funding cycles"
    )

    return (
        f"The company has raised capital that is {funding_msg}, and is {rounds_msg}. "
        "The model only considers total capital raised and number of rounds, so other "
        "drivers of venture risk (team quality, market structure, unit economics, etc.) "
        "are not captured here."
    )

