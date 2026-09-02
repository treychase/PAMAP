"""Getting PAMAP2 onto disk and into a DataFrame.

`load_subject` reads one `subjectNNN.dat` into a frame with the 54 names from
`schema.columns()`. `load_cohort` does that for every subject present.

The heart-rate column needs handling before anything downstream can use it.
The strap samples at ~9 Hz against the IMUs' 100 Hz, so the column is missing
on roughly ten rows in eleven. Those gaps are the sampling grid, not dropout,
and interpolating across them is the correct reconstruction. Genuinely long
gaps — the strap losing contact — are a different thing, so interpolation is
bounded by `HR_GAP_LIMIT_S` and anything longer stays missing.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

# The dataset's canonical home. Downloading it is a single call, but the host
# is not reachable from every environment - see `resolve` for what happens then.
UCI_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

# The strap samples at ~9 Hz, so a real gap between readings is ~0.11 s. Five
# seconds is comfortably longer than any sampling gap and short enough that a
# dropout is left as missing rather than bridged.
HR_GAP_LIMIT_S = 5.0


def download(dest: Path) -> Path:
    """Fetch and unpack the dataset. Returns the directory holding Protocol/."""
    import urllib.request

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL) as response:
        payload = response.read()

    # The published archive nests a second zip inside the first.
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        inner_name = next((n for n in outer.namelist() if n.endswith(".zip")), None)
        if inner_name is None:
            outer.extractall(dest)
        else:
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                inner.extractall(dest)

    protocol = next(dest.rglob("Protocol"), None)
    if protocol is None:
        raise FileNotFoundError(f"no Protocol/ directory under {dest}")
    return protocol.parent


def interpolate_heart_rate(frame: pd.DataFrame) -> pd.Series:
    """Put the ~9 Hz heart rate onto the 100 Hz grid the IMUs are sampled on."""
    hr = frame["heart_rate"].astype("float64")
    if hr.notna().sum() < 2:
        return hr

    t = frame["timestamp"].to_numpy(dtype="float64")
    known = hr.notna().to_numpy()
    filled = np.interp(t, t[known], hr.to_numpy()[known])

    # np.interp holds the end values flat beyond the observed range and bridges
    # every interior gap however long, so both are undone here: outside the
    # observed span, and across any gap longer than a plausible sampling
    # interval, the value goes back to missing.
    seen = t[known]
    gap_index = np.searchsorted(seen, t)
    left = seen[np.clip(gap_index - 1, 0, seen.size - 1)]
    right = seen[np.clip(gap_index, 0, seen.size - 1)]
    too_wide = (right - left) > HR_GAP_LIMIT_S
    outside = (t < seen[0]) | (t > seen[-1])

    filled[too_wide | outside] = np.nan
    return pd.Series(filled, index=frame.index, name="heart_rate")


def load_subject(path: Path, drop_transient: bool = True) -> pd.DataFrame:
    """Read one subjectNNN.dat into a named, typed frame."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=schema.columns(),
        na_values=["NaN"],
        # The default C parser handles the whitespace separator and is an order
        # of magnitude faster than the Python one on files this shape - a
        # session is ~320k rows of 54 floats.
        engine="c",
    )
    frame["activity_id"] = frame["activity_id"].astype("int64")
    frame["heart_rate"] = interpolate_heart_rate(frame)

    if drop_transient:
        frame = frame[frame["activity_id"] != schema.TRANSIENT_ID]

    frame["subject"] = path.stem
    frame["activity"] = frame["activity_id"].map(schema.ACTIVITIES)
    return frame.reset_index(drop=True)


def load_cohort(root: Path, split: str = "Protocol", drop_transient: bool = True):
    """Yield (subject id, frame) for every .dat file in `root/split`."""
    directory = Path(root) / split
    for path in sorted(directory.glob("subject*.dat")):
        yield path.stem, load_subject(path, drop_transient=drop_transient)


class DatasetUnavailable(RuntimeError):
    """Raised when the real dataset is neither on disk nor reachable."""


def resolve(root: Path | None, allow_download: bool = True) -> Path:
    """Find an unpacked PAMAP2 on disk, downloading it if permitted.

    Raises DatasetUnavailable rather than silently substituting anything. A
    stand-in cohort is a deliberate choice the caller makes by reaching for
    `pamap.synth`, never something this function does on the caller's behalf.
    """
    if root is not None:
        root = Path(root)
        if (root / "Protocol").is_dir():
            return root
        found = next(root.rglob("Protocol"), None)
        if found is not None:
            return found.parent
        if not allow_download:
            raise DatasetUnavailable(f"no Protocol/ directory under {root}")

    if not allow_download:
        raise DatasetUnavailable("no dataset root given and downloading is disabled")

    target = Path(root) if root is not None else Path.home() / ".cache" / "pamap2"
    try:
        return download(target)
    except Exception as exc:  # network, policy, or a moved URL
        raise DatasetUnavailable(
            f"could not reach {UCI_URL} ({exc.__class__.__name__}: {exc}). "
            "Download the dataset by hand and pass --data <dir>, or build "
            "against the stand-in cohort with --synthetic."
        ) from exc
