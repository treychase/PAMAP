"""Finding active, static and recovery states without being told where they are.

The question is which of three states a window of wearable data belongs to:

    active     the body is doing mechanical work
    static     the body is still and the heart is at its baseline
    recovery   the body is still and the heart is not

The first split is easy and an accelerometer alone can make it. The second is
the one worth the work, because *nothing in the dataset labels it*. Sitting
down before a run and sitting down after one carry the same activity id, the
same posture and the same near-zero acceleration; what separates them is a
heart rate still paying off the bout and falling rather than sitting flat. So
the target is not recoverable by supervised learning from the labels that
exist — the label would be the same for both — which is why this is clustered
rather than classified.

Why the clustering is in two stages
-----------------------------------
The obvious approach is one k-means at k=3 over movement and cardiac features
together. It does not work, and the way it fails is worth stating because it
is the reason for everything below.

Movement intensity spans three orders of magnitude from lying still to rope
jumping. Heart-rate reserve among *still* windows spans a few tenths. k-means
minimises total within-cluster distance, so given three clusters to spend it
buys the reduction that is on offer: it splits the enormous active spread into
moderate and vigorous, and leaves every still window — baseline and recovering
alike — in one lump. The static/recovery distinction, which is the entire point
of the exercise, gets no cluster at all. Down-weighting the movement block so
each block contributes equal variance does not fix it; nine movement columns
against two cardiac ones is not the problem, the shape of the distributions is.
That result is reproduced in `flat_comparison` rather than hidden, because "the
obvious thing fails, here is what it does instead" is a finding.

What works is to match the structure of the question, which is hierarchical
rather than flat. Two states are distinguished by mechanical work and two by
cardiac state, so the split is made in two stages, each on the block of
features that defines it:

    stage 1   movement features only, k=2   ->  active  vs  still
    stage 2   cardiac features only, k=2    ->  static  vs  recovery
              (fitted on the still windows alone)

Each stage is still unsupervised — no activity label is consulted at any point,
and the three clusters are named afterwards from where their centres sit in
original units. What the two stages add is the statement that these two
questions are asked on different evidence, which is true of the physiology and
is what a single distance metric cannot express.

Two further scaling decisions:

**Blocks are standardised, then down-weighted by the square root of their
width**, so a stage that draws on several correlated columns is not implicitly
weighted by how many of them there are.

**Intensity features are log1p-transformed first.** Movement intensity is
heavy-tailed and a z-score of a distribution that shaped puts every still
window in a heap at the bottom. log1p is monotone, so it changes the geometry
k-means sees without changing what any feature means.

Posture — how far the torso and shin are tilted from upright — is computed and
reported but never fitted on. It is a clean bimodal split, near 0 degrees
upright against near 90 lying down, and any clustering handed it spends a whole
cluster separating lying from sitting. That distinction is orthogonal to this
taxonomy: a man on a mat and a man in a chair are both, for these purposes,
still. Posture is read back afterwards as a description, where it doubles as a
check — lying and sitting landing in the same state is the intended behaviour,
and it is visible in the centroid table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .features import FEATURE_BLOCKS

STATES = ("active", "static", "recovery")

# Intensity features are positive and heavy-tailed; the rest are already on
# sensible scales (angles, fractions of reserve, bpm/min, Hz).
_LOG_FEATURES = {
    "sma_hand", "sma_chest", "sma_ankle",
    "dyn_std_ankle", "dyn_std_chest", "jerk_ankle",
}

# Columns summarised for every cluster, in original units, so the naming step
# and the published centroid table read the same numbers.
_CENTROID_COLUMNS = [
    "sma_ankle", "sma_hand", "sma_chest", "jerk_ankle", "cadence_power",
    "hr_reserve", "hr_slope", "hr_mean", "chest_tilt", "ankle_tilt",
]

# The blocks each stage draws on. Posture is computed and reported but never
# fitted on - see the module docstring for why.
STAGE_BLOCKS = {"motion": ("motion",), "cardiac": ("cardiac",)}
CLUSTER_BLOCKS = ("motion", "cardiac")

RANDOM_STATE = 610


@dataclass
class Clustering:
    """A fitted state model and everything needed to report on it."""

    labels: np.ndarray                 # state name per window
    centroids: pd.DataFrame            # cluster centres in original units
    diagnostics: dict


def design_matrix(windows: pd.DataFrame, blocks) -> tuple[np.ndarray, list[str]]:
    """Scale the named feature blocks into the space a stage is clustered in."""
    columns: list[str] = []
    scaled: list[np.ndarray] = []

    for block in blocks:
        names = FEATURE_BLOCKS[block]
        values = windows[names].to_numpy(dtype="float64").copy()
        for position, name in enumerate(names):
            if name in _LOG_FEATURES:
                values[:, position] = np.log1p(np.clip(values[:, position], 0, None))

        centre = values.mean(axis=0)
        spread = values.std(axis=0)
        spread[spread < 1e-9] = 1.0          # a constant feature contributes nothing
        values = (values - centre) / spread

        scaled.append(values / np.sqrt(values.shape[1]))
        columns.extend(names)

    return np.hstack(scaled), columns


def _intensity(frame: pd.DataFrame) -> pd.Series:
    """One movement score per row, so "which is moving more" is one comparison."""
    return np.log1p(frame[["sma_ankle", "sma_hand", "sma_chest"]].clip(lower=0)).mean(axis=1)


def _split(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """One k=2 k-means, with the diagnostics that say whether to believe it."""
    km = KMeans(n_clusters=2, n_init=25, random_state=RANDOM_STATE).fit(matrix)
    gm = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=RANDOM_STATE, n_init=5).fit(matrix)
    report = {
        "n": int(matrix.shape[0]),
        "silhouette": float(silhouette_score(matrix, km.labels_,
                                             sample_size=min(4000, matrix.shape[0]),
                                             random_state=RANDOM_STATE)),
        # A Gaussian mixture assumes elliptical components and assigns softly,
        # so agreement is evidence the split is in the data rather than an
        # artefact of k-means' preference for round, equal-sized clusters.
        "gmm_ari": float(adjusted_rand_score(km.labels_, gm.predict(matrix))),
    }
    return km.labels_, report


def flat_comparison(windows: pd.DataFrame, k: int = 3) -> dict:
    """Fit the single-stage k-means this module rejects, and report what it did.

    Kept so the claim in the module docstring is checkable rather than asserted:
    at k=3 over movement and cardiac features together, the clusters come out
    ordered by intensity and the still windows stay in one piece.
    """
    matrix, _ = design_matrix(windows, CLUSTER_BLOCKS)
    km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE).fit(matrix)

    frame = windows.assign(_flat=km.labels_)
    centres = frame.groupby("_flat")[["sma_ankle", "hr_reserve", "hr_slope"]].mean()
    centres["intensity"] = _intensity(frame).groupby(frame["_flat"]).mean()
    centres["share"] = frame["_flat"].value_counts(normalize=True).sort_index()
    centres = centres.sort_values("intensity")

    still = frame["_flat"] == centres.index[0]
    return {
        "k": k,
        "silhouette": float(silhouette_score(matrix, km.labels_,
                                             sample_size=min(4000, matrix.shape[0]),
                                             random_state=RANDOM_STATE)),
        "centres": centres.round(3).reset_index().to_dict("records"),
        # The tell: the quietest cluster holds essentially all the still
        # windows, so nothing was spent separating baseline from recovery.
        "still_in_one_cluster": float(still.mean()),
    }


def model_selection(windows: pd.DataFrame, k_values=range(2, 9)) -> list[dict]:
    """Silhouette and mixture BIC across k for the single-stage fit."""
    matrix, _ = design_matrix(windows, CLUSTER_BLOCKS)
    report = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(matrix)
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=RANDOM_STATE, n_init=3).fit(matrix)
        report.append({
            "k": int(k),
            "silhouette": float(silhouette_score(matrix, km.labels_,
                                                 sample_size=min(4000, matrix.shape[0]),
                                                 random_state=RANDOM_STATE)),
            "bic": float(gm.bic(matrix)),
        })
    return report


def fit(windows: pd.DataFrame, with_selection: bool = True) -> Clustering:
    """Two-stage clustering into active, static and recovery."""
    labels = np.empty(len(windows), dtype=object)

    # --- stage 1: is the body doing mechanical work? ---------------------
    motion_matrix, _ = design_matrix(windows, STAGE_BLOCKS["motion"])
    motion_labels, stage1 = _split(motion_matrix)

    intensity = _intensity(windows).to_numpy()
    busier = max((0, 1), key=lambda c: intensity[motion_labels == c].mean())
    is_active = motion_labels == busier
    labels[is_active] = "active"

    # --- stage 2: among still windows, is the heart still paying? --------
    still = windows.loc[~is_active]
    cardiac_matrix, _ = design_matrix(still, STAGE_BLOCKS["cardiac"])
    cardiac_labels, stage2 = _split(cardiac_matrix)

    reserve = still["hr_reserve"].to_numpy()
    elevated = max((0, 1), key=lambda c: reserve[cardiac_labels == c].mean())
    still_names = np.where(cardiac_labels == elevated, "recovery", "static")
    labels[~is_active] = still_names

    labels = labels.astype(str)

    frame = windows.assign(_state=labels)
    centroids = frame.groupby("_state")[_CENTROID_COLUMNS].mean()
    centroids["share"] = frame["_state"].value_counts(normalize=True)
    centroids["intensity"] = _intensity(frame).groupby(frame["_state"]).mean()

    diagnostics = {
        "n_windows": int(len(windows)),
        "stage1": stage1 | {"question": "active vs still", "features": "motion"},
        "stage2": stage2 | {"question": "static vs recovery", "features": "cardiac"},
        "shares": {state: float((labels == state).mean()) for state in STATES},
        "flat": flat_comparison(windows),
    }

    if "activity_id" in windows:
        # How much of the state split the recorded activity label already
        # explains. A high number would mean the clustering had only rederived
        # the labels; the interesting result is that it is not high, because
        # the static/recovery split cuts straight across `sitting`.
        codes = pd.factorize(labels)[0]
        diagnostics["ari_vs_activity"] = float(
            adjusted_rand_score(windows["activity_id"].to_numpy(), codes))

    if with_selection:
        diagnostics["selection"] = model_selection(windows)

    return Clustering(labels=labels, centroids=centroids, diagnostics=diagnostics)


def contingency(windows: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Recorded activity by discovered state, as shares of each activity."""
    table = pd.crosstab(windows["activity"], pd.Series(labels, index=windows.index, name="state"))
    for state in STATES:
        if state not in table:
            table[state] = 0
    table = table[list(STATES)]
    return table.div(table.sum(axis=1), axis=0)
