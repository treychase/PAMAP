"""Tests for the pieces that would fail silently.

Most of these check properties rather than fixed values: that the loader
reconstructs a sampling grid without inventing data across a dropout, that the
feature extractor recovers a cadence it was given, that strain is monotone in
effort, and that the clustering names its clusters by physiology rather than by
the order k-means happened to return them in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamap import cluster, features, ingest, schema, strain, synth  # noqa: E402


# --- schema ---------------------------------------------------------------

def test_column_layout_matches_the_published_format():
    columns = schema.columns()
    assert len(columns) == schema.N_COLUMNS == 54
    assert columns[:3] == ["timestamp", "activity_id", "heart_rate"]
    # The dataset documents the ankle IMU as columns 38-54, so its first
    # acceleration axis is column 39 (1-based).
    assert columns.index("ankle_acc16_x") + 1 == 39
    assert schema.accel_columns("chest") == ("chest_acc16_x", "chest_acc16_y", "chest_acc16_z")


def test_every_subject_has_the_metadata_the_reserve_calculation_needs():
    for subject, info in schema.SUBJECTS.items():
        assert info["hr_max"] > info["hr_rest"] > 0, subject
        assert info["sex"] in ("M", "F")


# --- heart-rate reconstruction --------------------------------------------

def test_interpolation_fills_the_sampling_grid_but_not_a_dropout():
    t = np.arange(0, 60, 0.01)
    hr = np.full(t.size, np.nan)
    hr[::11] = 70.0                       # the strap's ~9 Hz grid
    hr[(t > 20) & (t < 40)] = np.nan      # a 20 second dropout
    frame = pd.DataFrame({"timestamp": t, "heart_rate": hr})

    filled = ingest.interpolate_heart_rate(frame).to_numpy()

    assert np.isfinite(filled[(t > 5) & (t < 15)]).all()      # gaps bridged
    assert np.isnan(filled[(t > 22) & (t < 38)]).all()        # dropout left alone
    assert np.allclose(filled[np.isfinite(filled)], 70.0)


def test_interpolation_survives_a_column_with_nothing_in_it():
    frame = pd.DataFrame({"timestamp": np.arange(10.0), "heart_rate": np.full(10, np.nan)})
    assert ingest.interpolate_heart_rate(frame).isna().all()


# --- features -------------------------------------------------------------

@pytest.mark.parametrize("cadence", [0.8, 1.4, 1.9, 2.75])
def test_cadence_is_recovered_at_the_fundamental_not_a_harmonic(cadence):
    """The regression this guards: taking |a| before the FFT doubles the cadence.

    Rectifying a zero-mean oscillation halves its period, so a magnitude
    spectrum reports 2f (or whichever harmonic wins) instead of f.
    """
    hz = schema.IMU_HZ
    t = np.arange(0, features.WINDOW_S, 1.0 / hz)
    rng = np.random.default_rng(0)
    motion = np.stack([
        3.0 * np.sin(2 * np.pi * cadence * t) + 0.9 * np.sin(4 * np.pi * cadence * t),
        1.5 * np.sin(2 * np.pi * cadence * t + 1.0),
        0.8 * np.sin(2 * np.pi * cadence * t + 2.0),
    ], axis=1) + rng.normal(0, 0.05, (t.size, 3))

    peak, power = features._cadence(motion, hz)
    assert peak == pytest.approx(cadence, abs=0.12)
    assert power > 0.3


def test_still_signal_reports_no_meaningful_periodicity():
    hz = schema.IMU_HZ
    rng = np.random.default_rng(1)
    motion = rng.normal(0, 0.04, (int(features.WINDOW_S * hz), 3))
    _, power = features._cadence(motion, hz)
    assert power < 0.15


def test_windows_never_straddle_a_change_of_activity():
    frame = _one_session("subject101")
    windows = features.window_subject(frame, "subject101")
    assert not windows.empty
    for _, row in windows.iterrows():
        start, stop = row["t_start"], row["t_start"] + row["duration_s"]
        span = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < stop)]
        assert span["activity_id"].nunique() == 1


def test_posture_tilt_puts_lying_a_right_angle_from_upright():
    windows = _windowed("subject101")
    lying = windows[windows["activity"] == "lying"]["chest_tilt"]
    walking = windows[windows["activity"] == "walking"]["chest_tilt"]
    assert not lying.empty and not walking.empty
    assert walking.mean() < 15
    assert lying.mean() > 60


# --- strain ---------------------------------------------------------------

def test_trimp_rate_rises_with_effort_and_is_zero_at_rest():
    reserves = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    rates = strain.trimp_rate(reserves, "M")
    assert rates[0] == 0.0
    assert np.all(np.diff(rates) > 0)
    # The exponential is the point: the top of the range is worth far more per
    # minute than a linear weighting would give it.
    assert rates[-1] / rates[2] > 2.5


def test_the_display_scale_is_monotone_and_bounded():
    trimp = np.linspace(0, strain.SCALE_MAX, 50)
    scaled = strain.to_scale(trimp)
    assert scaled[0] == 0.0
    assert np.all(np.diff(scaled) > 0)
    assert scaled[-1] == pytest.approx(21.0)
    assert strain.to_scale(0.0) == 0.0


def test_predictors_never_leak_heart_rate_into_the_model():
    """The whole claim of the strain model is that it does not see heart rate."""
    windows = _clustered()
    columns = strain.predictors(windows).columns
    banned = {"hr_mean", "hr_reserve", "hr_slope", "trimp", "trimp_rate", "heart_rate"}
    assert banned.isdisjoint(set(columns))
    # hr_rest and hr_max are profile facts, not the live channel, and are meant
    # to be there - the reserve denominator has to come from somewhere.
    assert {"hr_rest", "hr_max"} <= set(columns)


# --- clustering -----------------------------------------------------------

def test_states_are_named_from_physiology_not_from_cluster_order():
    windows = _clustered()
    fitted = cluster.fit(windows, with_selection=False)
    centroids = fitted.centroids

    assert set(fitted.labels) == set(cluster.STATES)
    # active moves most
    assert centroids.loc["active", "intensity"] > centroids.loc["static", "intensity"]
    assert centroids.loc["active", "intensity"] > centroids.loc["recovery", "intensity"]
    # recovery is a still body with a heart that is not still
    assert centroids.loc["recovery", "hr_reserve"] > centroids.loc["static", "hr_reserve"]
    assert centroids.loc["recovery", "hr_slope"] < centroids.loc["static", "hr_slope"]


def test_posture_is_excluded_from_the_distance_metric():
    windows = _clustered()
    _, columns = cluster.design_matrix(windows, cluster.CLUSTER_BLOCKS)
    assert "chest_tilt" not in columns and "ankle_tilt" not in columns
    # and it is still available to describe the result
    assert "chest_tilt" in fitted_centroid_columns()


def fitted_centroid_columns():
    return cluster._CENTROID_COLUMNS


# --- fixtures -------------------------------------------------------------

_SESSION_CACHE: dict[str, pd.DataFrame] = {}


def _one_session(subject: str) -> pd.DataFrame:
    if subject not in _SESSION_CACHE:
        frame = pd.DataFrame(synth.build_subject(subject), columns=schema.columns())
        frame["activity_id"] = frame["activity_id"].astype("int64")
        frame["heart_rate"] = ingest.interpolate_heart_rate(frame)
        frame["activity"] = frame["activity_id"].map(schema.ACTIVITIES)
        frame["subject"] = subject
        _SESSION_CACHE[subject] = frame
    return _SESSION_CACHE[subject]


_WINDOW_CACHE: dict[str, pd.DataFrame] = {}


def _windowed(subject: str) -> pd.DataFrame:
    if subject not in _WINDOW_CACHE:
        windows = features.window_subject(_one_session(subject), subject)
        _WINDOW_CACHE[subject] = features.add_posture_tilt(windows)
    return _WINDOW_CACHE[subject]


def _clustered() -> pd.DataFrame:
    # Two subjects is enough structure for the naming rules to be exercised
    # without the whole cohort's runtime.
    windows = pd.concat([_windowed("subject101"), _windowed("subject104")], ignore_index=True)
    windows = features.add_posture_tilt(windows)
    fitted = cluster.fit(windows, with_selection=False)
    return strain.add_observed_strain(windows.assign(state=fitted.labels))


def test_overlapping_windows_do_not_double_count_the_session():
    """Totals are built on the stride, never the span.

    Windows overlap by half, so weighting each by its full ten seconds covers
    every second of the session twice and inflates every total derived from
    them - minutes in a state, accumulated TRIMP, session strain.
    """
    windows = _windowed("subject101")
    assert (windows["stride_s"] == features.HOP_S).all()
    assert (windows["duration_s"] == features.WINDOW_S).all()

    wall_clock = _one_session("subject101")["timestamp"].iloc[-1]
    counted = windows["stride_s"].sum()
    # Dropped straddling windows mean this lands under wall clock, never over.
    assert counted <= wall_clock
    assert counted > 0.5 * wall_clock

    scored = strain.add_observed_strain(windows)
    by_span = float((scored["trimp_rate"] * scored["duration_s"] / 60).sum())
    assert float(scored["trimp"].sum()) == pytest.approx(by_span / 2, rel=1e-9)
