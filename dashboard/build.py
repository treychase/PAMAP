"""Build the wearable strain dashboard as one self-contained HTML file.

    python dashboard/build.py --out wearable-strain.html                # stand-in cohort
    python dashboard/build.py --data ~/PAMAP2_Dataset --out page.html   # the real thing

Runs the whole pipeline — load, window, cluster, fit strain — and writes the
result into `dashboard/template.html` in place of its `__DATA__` token. The
page has no dependencies and fetches nothing at view time.

Every predicted number written into the payload is the leave-one-subject-out
prediction: the value the model produced for that person while fitted only on
the other eight. A dashboard showing in-sample predictions would look better
and mean nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamap import cluster, pipeline, schema, strain  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "template.html"

PROVENANCE_NOTE = {
    "pamap2": (
        "Built on the PAMAP2 Physical Activity Monitoring dataset (Reiss and "
        "Stricker, 2012) &mdash; nine subjects, three inertial units at 100&nbsp;Hz and a "
        "chest heart-rate monitor, through the twelve-activity protocol."
    ),
    "synthetic": (
        "<b>Built on a stand-in cohort, not on recorded people.</b> This page was "
        "generated in an environment with no route to archive.ics.uci.edu, where "
        "PAMAP2 is published, so the numbers on it come from a simulator that writes "
        "files in the dataset&rsquo;s exact schema &mdash; the same 54 columns, the same "
        "100&nbsp;Hz and 9&nbsp;Hz sampling rates, the same nine subject profiles and "
        "activity ids. Acceleration is synthesised in the sensor frame from a posture "
        "and a cadence, and heart rate is integrated as a first-order response with a "
        "slower recovery constant than onset, which is what creates the recovery state "
        "at all. The pipeline behind this page is the real one and runs on the real "
        "files unchanged: point it at a downloaded copy with <code>--data</code> and "
        "every number here is recomputed from recorded data. Treat the values as a "
        "demonstration that the method works, not as measurements of anybody."
    ),
}


def _subject_payload(subject: str, block: pd.DataFrame, rate: np.ndarray) -> dict:
    """Everything the page needs for one person."""
    info = schema.SUBJECTS[subject]
    block = block.sort_values("t_start").reset_index(drop=True)
    minutes = block["stride_s"].to_numpy() / 60.0

    observed_cum = strain.to_scale(np.cumsum(block["trimp"].to_numpy()))
    predicted_cum = strain.to_scale(np.cumsum(np.clip(rate, 0, None) * minutes))

    # The activity names repeat across hundreds of windows, so they are stored
    # once and referenced by index rather than written out per window.
    names = sorted(block["activity"].unique())
    index = {name: i for i, name in enumerate(names)}

    by_state = block.groupby("state")["stride_s"].sum() / 60.0
    total_trimp = float(block["trimp"].sum())

    activities = []
    for name, rows in block.groupby("activity"):
        activities.append({
            "name": name,
            "minutes": round(float(rows["stride_s"].sum() / 60.0), 2),
            "hr": round(float(rows["hr_mean"].mean()), 1),
            "reserve": round(float(rows["hr_reserve"].mean()), 4),
            "trimp": round(float(rows["trimp"].sum()), 2),
            "share": round(float(rows["trimp"].sum() / total_trimp) if total_trimp else 0.0, 4),
            "state": rows["state"].value_counts().idxmax(),
        })
    activities.sort(key=lambda row: -row["trimp"])

    return {
        "id": subject,
        "label": "Subject " + subject.replace("subject", ""),
        "sex": info["sex"], "age": info["age"],
        "height": info["height_cm"], "weight": info["weight_kg"],
        "hrRest": info["hr_rest"], "hrMax": info["hr_max"],
        "window": float(block["duration_s"].iloc[0]),
        # The ribbon tiles at the stride so consecutive blocks abut instead of
        # overlapping; the window span is still what a tooltip describes.
        "stride": float(block["stride_s"].iloc[0]),
        "durationMin": round(float(block["stride_s"].sum() / 60.0), 1),
        "trimp": round(total_trimp, 2),
        "observed": round(float(observed_cum[-1]), 2),
        "predicted": round(float(predicted_cum[-1]), 2),
        "peakHr": int(round(float(block["hr_mean"].max()))),
        "minutes": {state: round(float(by_state.get(state, 0.0)), 2) for state in cluster.STATES},
        "activities": activities,
        "acts": names,
        "t": [int(round(v)) for v in block["t_start"]],
        "a": [index[name] for name in block["activity"]],
        "s": [list(cluster.STATES).index(state) for state in block["state"]],
        "hr": [int(round(v)) for v in block["hr_mean"]],
        "res": [round(float(v), 3) for v in block["hr_reserve"]],
        "cum": [round(float(v), 2) for v in observed_cum],
        "pcum": [round(float(v), 2) for v in predicted_cum],
    }


def build(windows: pd.DataFrame, provenance: str) -> dict:
    fitted = cluster.fit(windows)
    windows = windows.assign(state=fitted.labels)
    windows = strain.add_observed_strain(windows)

    model = strain.fit(windows)
    windows = windows.assign(predicted_rate=model.held_out_rate)

    table = cluster.contingency(windows, fitted.labels)
    recorded = windows.groupby("activity")["stride_s"].sum() / 60.0
    contingency = [
        {"activity": activity,
         "minutes": round(float(recorded[activity]), 1),
         **{state: round(float(table.loc[activity, state]), 4) for state in cluster.STATES}}
        for activity in table.index
    ]
    # Heaviest activities at the top, so the row that matters is not buried.
    contingency.sort(key=lambda row: -row["minutes"])

    centroids = [
        {"state": state,
         "share": round(float(fitted.centroids.loc[state, "share"]), 4),
         "sma": round(float(fitted.centroids.loc[state, "sma_ankle"]), 3),
         "hr": round(float(fitted.centroids.loc[state, "hr_mean"]), 1),
         "reserve": round(float(fitted.centroids.loc[state, "hr_reserve"]), 4),
         "slope": round(float(fitted.centroids.loc[state, "hr_slope"]), 2),
         "tilt": round(float(fitted.centroids.loc[state, "chest_tilt"]), 1)}
        for state in cluster.STATES
    ]

    subjects = [
        _subject_payload(subject, block, block["predicted_rate"].to_numpy())
        for subject, block in windows.groupby("subject")
    ]
    subjects.sort(key=lambda row: row["id"])

    diagnostics = fitted.diagnostics
    sitting = windows[windows["activity"] == "sitting"]
    sitting_split = (sitting["state"].value_counts(normalize=True).to_dict()
                     if len(sitting) else {})

    return {
        "source": "PAMAP2 protocol" if provenance == "pamap2" else "stand-in cohort",
        "provenance": provenance,
        "provenanceNote": PROVENANCE_NOTE[provenance],
        "activities": int(windows["activity"].nunique()),
        "states": list(cluster.STATES),
        "cohort": {
            "nWindows": diagnostics["n_windows"],
            "stage1": diagnostics["stage1"],
            "stage2": diagnostics["stage2"],
            "ariActivity": round(diagnostics.get("ari_vs_activity", float("nan")), 4),
            "shares": {k: round(v, 4) for k, v in diagnostics["shares"].items()},
            "selection": [{"k": row["k"],
                           "silhouette": round(row["silhouette"], 4),
                           "bic": round(row["bic"], 1)} for row in diagnostics["selection"]],
            "flat": {"stillInOneCluster": round(diagnostics["flat"]["still_in_one_cluster"], 4),
                     "centres": diagnostics["flat"]["centres"]},
            "contingency": contingency,
            "centroids": centroids,
            "sittingSplit": {k: round(float(v), 4) for k, v in sitting_split.items()},
            "model": {
                "chosen": model.scores["chosen"],
                "r2": round(model.scores["candidates"][model.scores["chosen"]]["r2"], 4),
                "maeRate": round(model.scores["candidates"][model.scores["chosen"]]["mae_rate"], 4),
                "sessionMae": round(model.scores["session_mae_strain"], 3),
                "sessionMax": round(model.scores["session_max_error"], 3),
            },
        },
        "subjects": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="an unpacked PAMAP2_Dataset directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="build against the stand-in cohort instead of the real files")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "pamap-synthetic",
                        help="where the stand-in cohort is written")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root, provenance = pipeline.ensure_dataset(args.data, args.synthetic, args.cache)
    print(f"reading {provenance} cohort from {root}")

    windows = pipeline.build_windows(root)
    print(f"{len(windows)} windows over {windows['subject'].nunique()} subjects")

    payload = build(windows, provenance)
    print(f"strain model: {payload['cohort']['model']}")

    html = TEMPLATE.read_text()
    if "__DATA__" not in html:
        raise RuntimeError("template has no __DATA__ token")
    # separators keeps the payload compact; the page is committed to a repo.
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
