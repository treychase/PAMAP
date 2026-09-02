"""Turning 100 Hz waveforms into one row per window.

Nothing downstream works on raw samples. A single 100 Hz reading says almost
nothing — the interesting quantities are intensity, periodicity, posture and
cardiac response, and all four are properties of a stretch of time. So the
session is cut into overlapping windows and each becomes one row.

Ten seconds at half-overlap is the standard choice for accelerometer activity
recognition, and it is a real constraint rather than a convention: the window
has to hold several gait cycles for a cadence estimate to mean anything (ten
seconds is ~19 strides at walking pace, ~5 at the slowest movement here), and
it has to be short enough that a window does not straddle a change of activity.
Windows that do straddle one are dropped rather than assigned a majority label.

The features come in three blocks, and the split matters — `cluster.py` scales
by block so that a domain with more columns does not simply outvote one with
fewer:

    motion    intensity, periodicity and jerk, per unit and pooled
    cardiac   heart-rate reserve and its slope through the window
    posture   where gravity points, which separates postures at zero motion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

WINDOW_S = 10.0
HOP_S = 5.0

# Human movement, from the slowest deliberate sweep to a sprint cadence.
# Bounding the search stops a DC residue or high-frequency noise from being
# reported as somebody's stride rate.
CADENCE_BAND = (0.3, 5.0)

G = 9.80665

MOTION_FEATURES = [
    "sma_hand", "sma_chest", "sma_ankle",
    "dyn_std_ankle", "dyn_std_chest",
    "jerk_ankle", "cadence_hz", "cadence_power", "spectral_entropy",
]
CARDIAC_FEATURES = ["hr_reserve", "hr_slope"]
POSTURE_FEATURES = ["chest_tilt", "ankle_tilt"]

FEATURE_BLOCKS = {
    "motion": MOTION_FEATURES,
    "cardiac": CARDIAC_FEATURES,
    "posture": POSTURE_FEATURES,
}
ALL_FEATURES = MOTION_FEATURES + CARDIAC_FEATURES + POSTURE_FEATURES


def _spectrum(motion: np.ndarray, hz: float):
    """Power spectrum of a triaxial signal, summed over axes.

    The three axes are transformed separately and their power added, rather
    than the vector magnitude being transformed once. Taking the magnitude
    first would rectify each oscillation - |sin| has twice the period of sin -
    and report every cadence at double its true value, or at whichever harmonic
    happened to come out largest.
    """
    centred = motion - motion.mean(axis=0)
    # A Hann window keeps a cadence that is not an exact bin from smearing
    # across the whole spectrum and swamping the peak search.
    taper = np.hanning(centred.shape[0])[:, None]
    power = (np.abs(np.fft.rfft(centred * taper, axis=0)) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(centred.shape[0], d=1.0 / hz)
    return freqs, power


def _cadence(motion: np.ndarray, hz: float) -> tuple[float, float]:
    """Dominant movement frequency in Hz, and the share of power it carries.

    The power share is the useful half: walking is a metronome and puts most of
    its energy in one peak, while fidgeting in a chair has a dominant frequency
    too and almost no power behind it.
    """
    freqs, power = _spectrum(motion, hz)
    band = (freqs >= CADENCE_BAND[0]) & (freqs <= CADENCE_BAND[1])
    if not band.any() or power[band].sum() <= 0:
        return 0.0, 0.0

    total = power[1:].sum()          # drop the DC bin
    if total <= 0:
        return 0.0, 0.0

    peak = np.argmax(power[band])
    peak_hz = float(freqs[band][peak])

    # Count the peak plus its immediate neighbours, so a cadence sitting
    # between two bins is not scored as half as periodic as one that is not.
    peak_index = np.flatnonzero(band)[peak]
    lo, hi = max(1, peak_index - 1), min(power.size, peak_index + 2)
    return peak_hz, float(power[lo:hi].sum() / total)


def _spectral_entropy(motion: np.ndarray, hz: float) -> float:
    """Normalised entropy of the power spectrum: 0 is a pure tone, 1 is noise."""
    freqs, power = _spectrum(motion, hz)
    band = (freqs >= CADENCE_BAND[0]) & (freqs <= CADENCE_BAND[1])
    p = power[band]
    if p.sum() <= 0 or p.size < 2:
        return 1.0
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(band.sum()))


def _window_row(chunk: pd.DataFrame, hz: float, hr_rest: float, hr_max: float) -> dict | None:
    """Reduce one window to its feature row, or None if it is unusable."""
    out: dict[str, float] = {}

    dynamic: dict[str, np.ndarray] = {}
    for place in schema.IMU_PLACEMENTS:
        axes = chunk[list(schema.accel_columns(place))].to_numpy(dtype="float64")
        if not np.isfinite(axes).all():
            return None

        # Gravity is the constant part of the window and posture is its
        # direction; subtracting the per-axis mean leaves the movement. This
        # holds because a window is short enough that posture is fixed within
        # it, which is the same assumption that makes the window labellable.
        gravity = axes.mean(axis=0)
        motion = axes - gravity
        dynamic[place] = motion

        # Signal magnitude area: the standard accelerometer intensity measure,
        # mean absolute dynamic acceleration summed over the three axes.
        out[f"sma_{place}"] = float(np.abs(motion).sum(axis=1).mean())

        norm = np.linalg.norm(gravity)
        unit = gravity / norm if norm > 0 else np.zeros(3)
        for axis, value in zip("xyz", unit):
            out[f"{place}_grav_{axis}"] = float(value)

    for place in ("ankle", "chest"):
        out[f"dyn_std_{place}"] = float(np.linalg.norm(dynamic[place], axis=1).std())

    # Jerk - the rate of change of acceleration - separates a hard footfall
    # from a large smooth arm sweep, which SMA on its own does not.
    ankle_jerk = np.diff(dynamic["ankle"], axis=0) * hz
    out["jerk_ankle"] = float(np.sqrt((ankle_jerk ** 2).sum(axis=1).mean()))

    out["cadence_hz"], out["cadence_power"] = _cadence(dynamic["ankle"], hz)
    out["spectral_entropy"] = _spectral_entropy(dynamic["ankle"], hz)

    heart = chunk["heart_rate"].to_numpy(dtype="float64")
    finite = np.isfinite(heart)
    # Half the window is the least that gives a slope worth reporting.
    if finite.sum() < 0.5 * heart.size:
        return None

    seconds = chunk["timestamp"].to_numpy(dtype="float64")[finite]
    beats = heart[finite]
    out["hr_mean"] = float(beats.mean())
    out["hr_reserve"] = float((beats.mean() - hr_rest) / (hr_max - hr_rest))
    # bpm per minute: positive while the demand is rising, negative while the
    # heart is climbing back down after a bout.
    span = seconds[-1] - seconds[0]
    out["hr_slope"] = float(np.polyfit(seconds, beats, 1)[0] * 60.0) if span > 1.0 else 0.0

    return out


def window_subject(frame: pd.DataFrame, subject: str,
                   window_s: float = WINDOW_S, hop_s: float = HOP_S) -> pd.DataFrame:
    """Cut one subject's session into windows and extract features from each."""
    info = schema.SUBJECTS[subject]
    hz = schema.IMU_HZ
    size = int(round(window_s * hz))
    hop = int(round(hop_s * hz))

    activity = frame["activity_id"].to_numpy()
    rows = []

    for start in range(0, len(frame) - size + 1, hop):
        stop = start + size
        labels = activity[start:stop]
        # A window spanning a change of activity has no honest label and no
        # coherent posture, so it is dropped rather than assigned a majority.
        if labels[0] != labels[-1] or (labels != labels[0]).any():
            continue

        chunk = frame.iloc[start:stop]
        row = _window_row(chunk, hz, info["hr_rest"], info["hr_max"])
        if row is None:
            continue

        row["subject"] = subject
        row["activity_id"] = int(labels[0])
        row["activity"] = schema.ACTIVITIES.get(int(labels[0]), "unknown")
        row["t_start"] = float(chunk["timestamp"].iloc[0])
        # Two different durations, and confusing them double-counts the
        # session. `duration_s` is the span the features describe. `stride_s`
        # is how much time the window *accounts for*: consecutive windows
        # overlap by half, so weighting each one by its full span would cover
        # every second twice and inflate any total built from them - minutes
        # in a state, accumulated TRIMP, session strain. The strides partition
        # the session instead, which is what totals must be built on.
        row["duration_s"] = window_s
        row["stride_s"] = hop_s
        rows.append(row)

    return pd.DataFrame(rows)


# How periodic a window has to be before it counts as locomotion. Used only to
# pick each subject's upright reference posture, never as a label.
_UPRIGHT_CADENCE_POWER = 0.25


def add_posture_tilt(windows: pd.DataFrame) -> pd.DataFrame:
    """Add `chest_tilt` and `ankle_tilt`: degrees away from standing upright.

    An accelerometer at rest measures gravity, so the direction of the constant
    part of the signal is the posture the unit is in. What that direction means
    anatomically depends on how the unit was strapped on, which is fixed for a
    subject and arbitrary between them — so a raw axis reading is not
    comparable across people and hardcoding "y is vertical" is an assumption
    about the mounting rather than a measurement.

    The fix is to let each subject define their own upright. Windows with
    strong periodic movement are locomotion, and nobody walks lying down, so
    the median gravity direction over a subject's most periodic windows is that
    subject's upright reference. Every window is then scored as the angle
    between its gravity direction and that reference: near 0 degrees standing,
    sitting or walking, near 90 lying down.

    The reference is picked from the movement signal alone, so no activity
    label is consulted here and the clustering downstream stays unsupervised.
    """
    if windows.empty:
        return windows

    out = windows.copy()
    for place in ("chest", "ankle"):
        axes = [f"{place}_grav_{a}" for a in "xyz"]
        tilt = np.full(len(out), np.nan)

        for subject, index in out.groupby("subject").groups.items():
            block = out.loc[index, axes].to_numpy(dtype="float64")
            periodic = out.loc[index, "cadence_power"].to_numpy() >= _UPRIGHT_CADENCE_POWER
            # Fall back to the whole session if a subject never moves much:
            # the median is still dominated by whatever posture they held most.
            reference = np.median(block[periodic] if periodic.sum() >= 5 else block, axis=0)
            norm = np.linalg.norm(reference)
            if norm <= 0:
                tilt[out.index.get_indexer(index)] = 0.0
                continue
            reference = reference / norm

            cosine = np.clip(block @ reference, -1.0, 1.0)
            tilt[out.index.get_indexer(index)] = np.degrees(np.arccos(cosine))

        out[f"{place}_tilt"] = tilt

    return out
