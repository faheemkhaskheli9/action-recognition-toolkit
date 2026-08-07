"""Person detection + tracking: turns a raw multi-person video into
per-person track clips, so the rest of the pipeline (labeling, training,
inference) never has to deal with more than one subject per clip.

See `detectors/` and `trackers/` for the pluggable backends (each with its
own registry, mirroring `models/registry.py`), and `extract.py` for the
pipeline that ties them together.
"""
