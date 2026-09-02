"""Strain: what a session cost, and predicting it from movement alone.

Two different quantities live in this module and the distinction between them
is the point of the whole project.

**Observed strain** is what the heart says it cost. It is Banister's TRIMP —
training impulse — which weights every minute by heart-rate reserve under an
exponential, so a minute at 85% of reserve counts for far more than three
minutes at 30%. The exponent is what makes it a model of physiological cost
rather than a stopwatch, and it is the reason a strain score is not just
duration times intensity. Computing it requires a chest strap or an optical
sensor that is behaving.

**Predicted strain** is what the movement says it should have cost. It is
fitted here from accelerometry, the discovered state mix and the subject's
own body — never from heart rate. That is the useful direction commercially:
accelerometers are cheap, always on and never lose contact, while heart rate is
the channel that is missing when the strap is in a drawer, the wrist is cold,
or the sensor is on the wrong side of a tattoo. A model that recovers strain
from motion covers those gaps, and the gap between the two numbers is itself a
signal — a session that felt harder than it looked is the shape of fatigue,
heat, illness or a body that has not recovered.

Predictions are validated leave-one-subject-out. Windows from one person are
serially correlated and share a body, so a random split would put neighbouring
windows from the same session on both sides of it and report a score that says
nothing about the next person to put the watch on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import schema
from .cluster import STATES

# Banister's exponent. The published coefficient differs by sex because the
# lactate-versus-reserve curve it was fitted to does.
TRIMP_B = {"M": 1.92, "F": 1.67}

# The 0-21 display scale. TRIMP is unbounded and its useful range is
# multiplicative, so a linear axis wastes most of its length on the top end
# nobody reaches. These two constants are the whole map and both are stated
# rather than fitted: SCALE_MAX is the TRIMP of an all-out endurance day, which
# anchors 21, and SCALE_SHAPE sets how fast the curve rises out of nothing.
#
# The 0-21 range is a presentation convention borrowed from consumer wearables.
# It is not a validated clinical instrument and nothing here calibrates it
# against one - TRIMP underneath it is the quantity with literature behind it.
SCALE_MAX = 300.0
SCALE_SHAPE = 25.0


def trimp_rate(hr_reserve: np.ndarray | float, sex: str) -> np.ndarray | float:
    """Banister TRIMP accumulated per minute at a given fraction of reserve."""
    reserve = np.clip(hr_reserve, 0.0, 1.0)
    return reserve * 0.64 * np.exp(TRIMP_B.get(sex, TRIMP_B["M"]) * reserve)


def to_scale(trimp: np.ndarray | float) -> np.ndarray | float:
    """Map accumulated TRIMP onto the 0-21 strain scale."""
    trimp = np.clip(trimp, 0.0, None)
    return 21.0 * np.log1p(trimp / SCALE_SHAPE) / np.log1p(SCALE_MAX / SCALE_SHAPE)


def add_observed_strain(windows: pd.DataFrame) -> pd.DataFrame:
    """Add the per-window TRIMP rate and the TRIMP each window contributed."""
    out = windows.copy()
    sex = out["subject"].map(lambda s: schema.SUBJECTS[s]["sex"])
    out["trimp_rate"] = [trimp_rate(r, x) for r, x in zip(out["hr_reserve"], sex)]
    # Weighted by stride, not span: overlapping windows must partition the
    # session or every minute is counted twice.
    out["trimp"] = out["trimp_rate"] * (out["stride_s"] / 60.0)
    return out


# Movement, body and discovered state. Heart rate is deliberately absent: the
# whole point is to predict the cardiac cost without measuring it.
MOTION_INPUTS = [
    "sma_hand", "sma_chest", "sma_ankle",
    "dyn_std_ankle", "dyn_std_chest", "jerk_ankle",
    "cadence_hz", "cadence_power", "spectral_entropy",
    "chest_tilt", "ankle_tilt",
]
BODY_INPUTS = ["age", "height_cm", "weight_kg", "bmi", "is_female", "hr_rest", "hr_max"]


def predictors(windows: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature frame the strain model is fitted on."""
    frame = windows.copy()
    info = frame["subject"].map(schema.SUBJECTS)
    for field in ("age", "height_cm", "weight_kg", "hr_rest", "hr_max"):
        frame[field] = [row[field] for row in info]
    frame["is_female"] = [1.0 if row["sex"] == "F" else 0.0 for row in info]
    frame["bmi"] = frame["weight_kg"] / (frame["height_cm"] / 100.0) ** 2

    # Resting and maximum heart rate are body facts, taken once from a profile
    # or a fitness test. They are not the live heart-rate channel the model is
    # standing in for, and using them is what lets one model serve a cohort
    # whose reserve spans 54 to 75 bpm at rest.
    for state in STATES:
        frame[f"state_{state}"] = (frame["state"] == state).astype("float64")

    columns = MOTION_INPUTS + BODY_INPUTS + [f"state_{s}" for s in STATES]
    return frame[columns]


@dataclass
class StrainModel:
    """A fitted predictor of TRIMP rate from movement, and its honest scores."""

    model: object
    columns: list[str]
    scores: dict
    per_subject: pd.DataFrame
    # Held-out predicted TRIMP rate for every window, aligned to the frame the
    # model was fitted on. Each value comes from the fold in which that
    # window's subject was the test set, so a curve drawn from these is a
    # prediction for a person the model had not seen - which is the only
    # version worth showing anybody.
    held_out_rate: np.ndarray | None = None

    def predict_rate(self, windows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(predictors(windows)[self.columns]), 0.0, None)


def _candidates():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "boosted": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.07, max_iter=400,
            min_samples_leaf=40, l2_regularization=1.0, random_state=610),
    }


def fit(windows: pd.DataFrame) -> StrainModel:
    """Fit strain from movement, choosing the model by leave-one-subject-out score."""
    X = predictors(windows)
    y = windows["trimp_rate"].to_numpy(dtype="float64")
    groups = windows["subject"].to_numpy()
    splitter = LeaveOneGroupOut()

    results, folds = {}, {}
    for name, estimator in _candidates().items():
        predicted = np.full(y.size, np.nan)
        for train, test in splitter.split(X, y, groups):
            estimator.fit(X.iloc[train], y[train])
            predicted[test] = estimator.predict(X.iloc[test])
        predicted = np.clip(predicted, 0.0, None)
        results[name] = {
            "r2": float(r2_score(y, predicted)),
            "mae_rate": float(mean_absolute_error(y, predicted)),
        }
        folds[name] = predicted

    best = max(results, key=lambda n: results[n]["r2"])
    held_out = folds[best]

    # The number that actually matters is not the per-window rate but the
    # session total it integrates to, on the scale a person is shown.
    per_subject = []
    minutes = windows["stride_s"].to_numpy() / 60.0
    for subject in sorted(set(groups)):
        mask = groups == subject
        observed = to_scale(float((y[mask] * minutes[mask]).sum()))
        predicted = to_scale(float((held_out[mask] * minutes[mask]).sum()))
        per_subject.append({"subject": subject,
                            "observed_strain": observed,
                            "predicted_strain": predicted,
                            "error": predicted - observed})
    per_subject = pd.DataFrame(per_subject)

    scores = {
        "chosen": best,
        "candidates": results,
        "cv": "leave-one-subject-out",
        "n_windows": int(y.size),
        "session_mae_strain": float(per_subject["error"].abs().mean()),
        "session_max_error": float(per_subject["error"].abs().max()),
    }

    # The returned model is refit on everyone; the scores above are the
    # held-out ones, so nothing reported was scored on data it had seen.
    final = _candidates()[best]
    final.fit(X, y)
    return StrainModel(model=final, columns=list(X.columns), scores=scores,
                       per_subject=per_subject, held_out_rate=held_out)
