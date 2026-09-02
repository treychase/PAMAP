"""One call from raw files to the window table every other module consumes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import features, ingest, schema, synth


def build_windows(root: Path, split: str = "Protocol") -> pd.DataFrame:
    """Load every subject under `root/split` and window them.

    Subjects are loaded and reduced one at a time: a session is ~300k rows of
    54 float columns, so holding nine of them at once costs about a gigabyte
    for no reason. Windowed, the whole cohort is a few thousand rows.
    """
    tables = []
    for subject, frame in ingest.load_cohort(root, split=split):
        if subject not in schema.SUBJECTS:
            continue
        tables.append(features.window_subject(frame, subject))
        del frame

    if not tables:
        raise FileNotFoundError(f"no subject files found under {Path(root) / split}")

    windows = pd.concat(tables, ignore_index=True)
    return features.add_posture_tilt(windows)


def ensure_dataset(root: Path | None, synthetic: bool, cache: Path) -> tuple[Path, str]:
    """Resolve the real dataset, or write the stand-in cohort. Returns (root, provenance)."""
    if not synthetic:
        found = ingest.resolve(root)
        return found, "pamap2"

    cache = Path(cache)
    if not (cache / "Protocol").is_dir():
        synth.write_cohort(cache)
    return cache, "synthetic"
