"""The physical layout of the PAMAP2 Physical Activity Monitoring dataset.

PAMAP2 (Reiss & Stricker, 2012) is nine subjects wearing three Colibri inertial
measurement units — one on the dominant wrist, one on the chest, one on the
dominant ankle — plus a chest heart-rate monitor, through a twelve-activity
protocol. It is the dataset this project is built around because it carries all
three things a wearable strain model needs at once: a ground-truth activity
label, triaxial acceleration, and a biometric channel.

Every file in `Protocol/` and `Optional/` is whitespace-separated with 54
columns per row and no header, so the column names below *are* the schema.

    1       timestamp (s)
    2       activity id
    3       heart rate (bpm)
    4-20    IMU over the dominant wrist
    21-37   IMU on the chest
    38-54   IMU on the dominant ankle

and each 17-column IMU block is laid out

    1       temperature (C)
    2-4     acceleration, +/-16g scale (m/s^2)
    5-7     acceleration, +/-6g scale (m/s^2)
    8-10    gyroscope (rad/s)
    11-13   magnetometer (uT)
    14-17   orientation - documented as invalid in this collection

The IMUs sample at 100 Hz and the heart-rate monitor at roughly 9 Hz, so the
heart-rate column is present on about one row in eleven and missing on the
rest. That is a property of the recording, not damage to the file.
"""

from __future__ import annotations

IMU_PLACEMENTS = ("hand", "chest", "ankle")

# Within one IMU block, in file order.
_IMU_FIELDS = (
    "temp",
    "acc16_x", "acc16_y", "acc16_z",
    "acc6_x", "acc6_y", "acc6_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "orient_0", "orient_1", "orient_2", "orient_3",
)

IMU_WIDTH = len(_IMU_FIELDS)          # 17
N_COLUMNS = 3 + IMU_WIDTH * len(IMU_PLACEMENTS)   # 54


def columns() -> list[str]:
    """The 54 column names, in file order."""
    names = ["timestamp", "activity_id", "heart_rate"]
    for place in IMU_PLACEMENTS:
        names += [f"{place}_{field}" for field in _IMU_FIELDS]
    return names


# The dataset's own readme recommends the +/-16g accelerometer over the +/-6g
# one: the 6g sensor saturates during running and rope jumping, which are
# exactly the windows a strain model most needs to get right.
ACCEL_SCALE = "acc16"


def accel_columns(placement: str) -> tuple[str, str, str]:
    """The three column names carrying the recommended accelerometer axes."""
    if placement not in IMU_PLACEMENTS:
        raise ValueError(f"unknown IMU placement {placement!r}; expected one of {IMU_PLACEMENTS}")
    return tuple(f"{placement}_{ACCEL_SCALE}_{axis}" for axis in "xyz")  # type: ignore[return-value]


# Activity id 0 marks the transient periods between protocol activities — the
# subject walking to the next station. The readme says to discard it, and every
# published result on this dataset does.
TRANSIENT_ID = 0

ACTIVITIES = {
    0:  "transient",
    1:  "lying",
    2:  "sitting",
    3:  "standing",
    4:  "walking",
    5:  "running",
    6:  "cycling",
    7:  "Nordic walking",
    9:  "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

# The twelve activities every subject performs, in protocol order. The others
# in ACTIVITIES appear only in the optional sessions and only for some
# subjects, so cohort-wide comparisons are built on these.
PROTOCOL_ORDER = (1, 2, 3, 17, 16, 12, 13, 4, 7, 6, 5, 24)


# Subject characteristics, transcribed from the dataset's own
# subjectInformation.pdf. Resting and maximum heart rate are what turn a raw
# bpm trace into heart-rate reserve, which is the only form of the signal that
# is comparable between a 23-year-old and a 32-year-old.
SUBJECTS = {
    "subject101": {"sex": "M", "age": 27, "height_cm": 182, "weight_kg": 83, "hr_rest": 75, "hr_max": 193, "hand": "right"},
    "subject102": {"sex": "F", "age": 25, "height_cm": 169, "weight_kg": 78, "hr_rest": 74, "hr_max": 195, "hand": "right"},
    "subject103": {"sex": "M", "age": 31, "height_cm": 187, "weight_kg": 92, "hr_rest": 68, "hr_max": 189, "hand": "right"},
    "subject104": {"sex": "M", "age": 24, "height_cm": 194, "weight_kg": 95, "hr_rest": 58, "hr_max": 196, "hand": "right"},
    "subject105": {"sex": "M", "age": 26, "height_cm": 180, "weight_kg": 73, "hr_rest": 70, "hr_max": 194, "hand": "right"},
    "subject106": {"sex": "M", "age": 26, "height_cm": 183, "weight_kg": 69, "hr_rest": 60, "hr_max": 194, "hand": "right"},
    "subject107": {"sex": "M", "age": 23, "height_cm": 173, "weight_kg": 86, "hr_rest": 60, "hr_max": 197, "hand": "right"},
    "subject108": {"sex": "M", "age": 32, "height_cm": 179, "weight_kg": 87, "hr_rest": 66, "hr_max": 188, "hand": "left"},
    "subject109": {"sex": "M", "age": 31, "height_cm": 168, "weight_kg": 65, "hr_rest": 54, "hr_max": 189, "hand": "right"},
}

IMU_HZ = 100.0     # the inertial units
HR_HZ = 9.0        # the chest strap
