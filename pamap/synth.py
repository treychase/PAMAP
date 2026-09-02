"""A stand-in cohort in the PAMAP2 schema, for when the real files are absent.

The dataset lives at archive.ics.uci.edu, which is not reachable from every
environment — a locked-down CI runner or an egress policy will refuse it. This
module writes `subjectNNN.dat` files with the same 54 columns, the same
sampling rates, the same activity ids and the same nine subjects, so the rest
of the package runs against them unmodified.

It is a stand-in, not the dataset. Numbers produced from it describe this
generator, and anything published off it has to say so. What it is *for* is
keeping the pipeline honest and runnable: the loader, the windowing, the
clustering and the strain fit all see input they cannot distinguish by shape
from the real thing, so the code that works here is the code that works there.

Two design choices matter, because the analysis downstream leans on them:

**Acceleration is synthesised in the sensor frame.** Each activity fixes where
gravity points for each of the three units — that is what posture *is* to an
accelerometer, and it is why lying and standing are separable at zero motion —
and adds a periodic component at the activity's cadence with harmonics, plus
broadband noise. Nothing is drawn from a per-activity summary statistic, so the
feature extractor has to recover cadence and intensity from the waveform the
way it would from a real one.

**Heart rate lags.** It chases the activity's target as a first-order system
with a fast onset constant and a slow recovery one, which is the standard
first-order approximation of the cardiac response. The consequence is the whole
point of the project: after a hard bout, a subject sitting still has a body at
rest and a heart that is not, and that state is physiologically distinct from
the same posture before the bout. Neither the activity id nor any label in the
file marks that difference. The clustering has to find it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import schema

G = 9.80665


@dataclass(frozen=True)
class Motion:
    """How one IMU sees one activity.

    gravity: unit vector along gravity in the sensor frame, i.e. posture.
    amplitude: RMS of the dynamic component per axis, m/s^2.
    cadence: fundamental frequency of the movement in Hz; 0 for aperiodic.
    jitter: broadband noise per axis, m/s^2.
    """

    gravity: tuple[float, float, float]
    amplitude: tuple[float, float, float]
    cadence: float
    jitter: float = 0.05


@dataclass(frozen=True)
class Activity:
    """One protocol activity: what it looks like, and what it costs."""

    activity_id: int
    minutes: float
    hr_reserve: float               # steady-state fraction of HR reserve
    motion: dict[str, Motion]
    temperature: float = 32.0


def _still(gravity, jitter=0.04) -> Motion:
    return Motion(gravity=gravity, amplitude=(0.0, 0.0, 0.0), cadence=0.0, jitter=jitter)


# Postures, as gravity directions per unit. Upright puts gravity down the
# long axis of the wrist and shin and down the chest; lying rolls it onto a
# different axis of all three at once, which is what makes lying and sitting
# separable to an accelerometer that cannot see motion.
_UP_HAND, _UP_CHEST, _UP_ANKLE = (0, -1, 0), (0, 0, -1), (0, -1, 0)
_LIE_HAND, _LIE_CHEST, _LIE_ANKLE = (-1, 0, 0), (0, -1, 0), (0, 0, -1)
_SIT_ANKLE = (-0.71, -0.71, 0.0)


PROTOCOL: tuple[Activity, ...] = (
    Activity(1, 3.0, 0.02, {                    # lying
        "hand": _still(_LIE_HAND), "chest": _still(_LIE_CHEST), "ankle": _still(_LIE_ANKLE)}),
    Activity(2, 3.0, 0.06, {                    # sitting
        "hand": _still(_UP_HAND), "chest": _still(_UP_CHEST), "ankle": _still(_SIT_ANKLE)}),
    Activity(3, 3.0, 0.11, {                    # standing, with postural sway
        "hand": Motion(_UP_HAND, (0.18, 0.10, 0.18), 0.25),
        "chest": Motion(_UP_CHEST, (0.15, 0.15, 0.08), 0.25),
        "ankle": Motion(_UP_ANKLE, (0.10, 0.06, 0.10), 0.25)}),
    Activity(17, 3.0, 0.19, {                   # ironing: the arm works, the legs do not
        "hand": Motion(_UP_HAND, (2.6, 1.1, 1.5), 0.75),
        "chest": Motion(_UP_CHEST, (0.35, 0.30, 0.20), 0.75),
        "ankle": Motion(_UP_ANKLE, (0.18, 0.12, 0.14), 0.30)}),
    Activity(16, 3.0, 0.30, {                   # vacuuming: arm sweep plus slow steps
        "hand": Motion(_UP_HAND, (3.4, 1.8, 2.2), 0.65),
        "chest": Motion(_UP_CHEST, (0.9, 0.8, 0.6), 1.30),
        "ankle": Motion(_UP_ANKLE, (1.9, 1.5, 1.4), 1.30)}),
    Activity(12, 1.5, 0.52, {                   # ascending stairs
        "hand": Motion(_UP_HAND, (2.2, 1.7, 1.9), 1.55),
        "chest": Motion(_UP_CHEST, (1.7, 2.4, 1.3), 1.55),
        "ankle": Motion(_UP_ANKLE, (5.6, 6.4, 4.1), 1.55)}),
    Activity(13, 1.5, 0.41, {                   # descending stairs: faster, harder landings
        "hand": Motion(_UP_HAND, (2.4, 1.8, 2.0), 1.80),
        "chest": Motion(_UP_CHEST, (1.9, 2.7, 1.5), 1.80),
        "ankle": Motion(_UP_ANKLE, (6.3, 7.4, 4.6), 1.80)}),
    Activity(4, 3.0, 0.34, {                    # walking
        "hand": Motion(_UP_HAND, (1.9, 1.3, 1.6), 1.90),
        "chest": Motion(_UP_CHEST, (1.4, 1.9, 1.0), 1.90),
        "ankle": Motion(_UP_ANKLE, (4.4, 5.1, 3.2), 1.90)}),
    Activity(7, 3.0, 0.45, {                    # Nordic walking: the poles load the arms
        "hand": Motion(_UP_HAND, (4.1, 2.9, 3.3), 1.75),
        "chest": Motion(_UP_CHEST, (1.6, 2.2, 1.2), 1.75),
        "ankle": Motion(_UP_ANKLE, (4.8, 5.5, 3.5), 1.75)}),
    Activity(6, 3.0, 0.47, {                    # cycling: legs spin, hands are parked
        "hand": Motion(_UP_HAND, (0.35, 0.28, 0.30), 1.40),
        "chest": Motion(_UP_CHEST, (0.75, 0.65, 0.55), 1.40),
        "ankle": Motion(_UP_ANKLE, (3.1, 2.8, 2.6), 1.40)}),
    Activity(5, 3.0, 0.72, {                    # running
        "hand": Motion(_UP_HAND, (5.2, 3.8, 4.4), 2.75),
        "chest": Motion(_UP_CHEST, (4.6, 6.1, 3.4), 2.75),
        "ankle": Motion(_UP_ANKLE, (12.5, 14.8, 9.2), 2.75)}),
    Activity(24, 1.5, 0.83, {                   # rope jumping
        "hand": Motion(_UP_HAND, (4.8, 3.2, 4.1), 2.20),
        "chest": Motion(_UP_CHEST, (5.4, 8.2, 3.9), 2.20),
        "ankle": Motion(_UP_ANKLE, (14.2, 17.5, 10.4), 2.20)}),
)

# Seated rest between protocol activities. Labelled `sitting` exactly like the
# scheduled sitting block, because that is what the subject is doing and what a
# real annotator would write down. Whether a given rest window is a body at
# baseline or a body paying off an oxygen debt is not recorded anywhere - it is
# in the heart rate, and only the clustering ever names it.
REST = Activity(2, 2.0, 0.06, {
    "hand": _still(_UP_HAND), "chest": _still(_UP_CHEST), "ankle": _still(_SIT_ANKLE)})

# First-order cardiac response. Onset is quicker than recovery, which is why an
# elevated heart rate outlives the activity that caused it.
TAU_ONSET_S = 26.0
TAU_RECOVERY_S = 78.0


def _accelerometer(motion: Motion, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One IMU's triaxial acceleration over `t`, in m/s^2, sensor frame."""
    out = G * np.asarray(motion.gravity, dtype="float64")[None, :] * np.ones((t.size, 1))

    if motion.cadence > 0:
        # Three harmonics with decaying weight: a gait waveform is periodic but
        # nothing like a pure sinusoid, and the harmonic content is what a
        # spectral feature has to work with.
        for axis in range(3):
            amplitude = motion.amplitude[axis]
            if amplitude <= 0:
                continue
            phase = rng.uniform(0, 2 * np.pi)
            wave = np.zeros_like(t)
            weights = (1.0, 0.42, 0.18)
            for harmonic, weight in enumerate(weights, start=1):
                wave += weight * np.sin(2 * np.pi * harmonic * motion.cadence * t + harmonic * phase)
            # Scale so `amplitude` is the RMS of the summed harmonics.
            wave *= np.sqrt(2.0) / np.sqrt(sum(w * w for w in weights))
            out[:, axis] += amplitude * wave

    out += rng.normal(0.0, motion.jitter, size=out.shape)
    return out


def _heart_rate(target_reserve: np.ndarray, dt: float, hr_rest: float,
                hr_max: float, rng: np.random.Generator) -> np.ndarray:
    """Integrate the first-order cardiac response against a demand trace."""
    reserve = np.empty_like(target_reserve)
    state = float(target_reserve[0])
    for i, demand in enumerate(target_reserve):
        tau = TAU_ONSET_S if demand > state else TAU_RECOVERY_S
        state += (demand - state) * (dt / tau)
        reserve[i] = state

    hr = hr_rest + reserve * (hr_max - hr_rest)
    # Beat-to-beat variability, larger at rest than at effort, as it is in life.
    hr += rng.normal(0.0, 1.0, size=hr.size) * (2.4 - 1.4 * np.clip(reserve, 0, 1))
    return hr


def build_subject(subject: str, seed: int = 0) -> np.ndarray:
    """Generate one subject's session as a (rows, 54) array in file order."""
    if subject not in schema.SUBJECTS:
        raise ValueError(f"unknown subject {subject!r}")
    info = schema.SUBJECTS[subject]
    rng = np.random.default_rng(abs(hash(subject)) % (2**32) + seed)

    dt = 1.0 / schema.IMU_HZ

    # Fitness shifts what a given activity costs: the same run is a different
    # fraction of reserve for a 54 bpm resting heart rate than a 75 bpm one.
    fitness = np.interp(info["hr_rest"], [54, 75], [0.86, 1.14])

    # Alternate activity and seated rest, so every hard bout is followed by a
    # stretch of stillness the heart has to climb down through.
    plan: list[Activity] = []
    for index, activity in enumerate(PROTOCOL):
        plan.append(activity)
        if index < len(PROTOCOL) - 1:
            plan.append(REST)

    blocks, labels, demand = [], [], []
    for activity in plan:
        minutes = activity.minutes * rng.uniform(0.9, 1.1)
        n = int(round(minutes * 60 * schema.IMU_HZ))
        t_local = np.arange(n) * dt

        row = np.empty((n, schema.N_COLUMNS - 3), dtype="float64")
        for slot, place in enumerate(schema.IMU_PLACEMENTS):
            motion = activity.motion[place]
            acc = _accelerometer(motion, t_local, rng)
            base = slot * schema.IMU_WIDTH
            row[:, base + 0] = activity.temperature + rng.normal(0, 0.05, n)
            row[:, base + 1:base + 4] = acc                       # +/-16g
            # The +/-6g sensor sees the same motion and clips, which is exactly
            # why the dataset's readme steers you to the 16g columns.
            row[:, base + 4:base + 7] = np.clip(acc, -6 * G, 6 * G)
            row[:, base + 7:base + 10] = rng.normal(0, 0.05 + 0.10 * np.mean(motion.amplitude), (n, 3))
            row[:, base + 10:base + 13] = rng.normal(0, 2.0, (n, 3)) + np.array([20.0, -8.0, 35.0])
            row[:, base + 13:base + 17] = np.nan                  # orientation: invalid, as documented

        blocks.append(row)
        labels.append(np.full(n, activity.activity_id, dtype="float64"))
        demand.append(np.full(n, np.clip(activity.hr_reserve * fitness, 0.0, 0.95)))

    body = np.vstack(blocks)
    activity_ids = np.concatenate(labels)
    reserve_demand = np.concatenate(demand)
    n_total = body.shape[0]
    timestamps = np.arange(n_total) * dt

    hr = _heart_rate(reserve_demand, dt, info["hr_rest"], info["hr_max"], rng)
    # Decimate to the strap's ~9 Hz: the rest of the column is genuinely absent
    # in the real files, and the loader has to reconstruct it either way.
    stride = int(round(schema.IMU_HZ / schema.HR_HZ))
    sampled = np.full(n_total, np.nan)
    sampled[::stride] = hr[::stride]

    out = np.empty((n_total, schema.N_COLUMNS), dtype="float64")
    out[:, 0] = timestamps
    out[:, 1] = activity_ids
    out[:, 2] = sampled
    out[:, 3:] = body
    return out


def write_cohort(root: Path, seed: int = 0) -> Path:
    """Write the whole stand-in cohort under `root/Protocol/`."""
    root = Path(root)
    protocol = root / "Protocol"
    protocol.mkdir(parents=True, exist_ok=True)

    for subject in schema.SUBJECTS:
        rows = build_subject(subject, seed=seed)
        # Match the real files: space separated, no header, "NaN" for missing.
        np.savetxt(protocol / f"{subject}.dat", rows, fmt="%.6g", delimiter=" ")
    return root
