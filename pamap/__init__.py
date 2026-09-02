"""Wearable activity states and cardiovascular strain, on PAMAP2.

    from pamap import pipeline, cluster, strain

    root, provenance = pipeline.ensure_dataset(path, synthetic=False, cache=cache)
    windows = pipeline.build_windows(root)          # one row per 10 s window
    states = cluster.fit(windows)                   # active / static / recovery
    windows = strain.add_observed_strain(windows.assign(state=states.labels))
    model = strain.fit(windows)                     # strain from movement alone
"""

__all__ = ["cluster", "features", "ingest", "pipeline", "schema", "strain", "synth"]
