"""Standalone NWB packaging for passive SweepStim sessions."""

from __future__ import annotations

import datetime
import json
import logging
import pickle
import re
from pathlib import Path
from uuid import uuid4

import numpy as np

from hdmf.common import VectorData
from ndx_events import NdxEventsNWBFile
from ndx_hed import HedLabMetaData, HedTags
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from .classify import classify_sweepstim_session
from .running import add_running_speed
from .timestamp_alignment import compute_sweepstim_timestamp_alignment


logger = logging.getLogger(__name__)
HED_SCHEMA_VERSION = "8.3.0"


def _to_datetime(value) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.datetime.fromisoformat(value)
    elif isinstance(value, (int, float)):
        # camstim stores start_time as a Unix epoch timestamp.
        dt = datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc)
    else:
        dt = datetime.datetime.now(datetime.timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _hed_safe_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def _clip_name(stim_obj: dict, default_index: int) -> str:
    raw = stim_obj.get("movie_path") or stim_obj.get("stim_path")
    if not raw:
        return f"stim_{default_index:03d}"
    # camstim paths are Windows-style (backslash UNC paths); split on both
    # separators so this resolves the basename on Linux too, then drop the ext.
    base = re.split(r"[\\/]", str(raw))[-1]
    return base.rsplit(".", 1)[0] or base


def _as_sequence(value):
    """Normalize list/tuple/numpy-array payloads to a plain Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _subject_id_from_pkl(pkl: dict) -> str:
    items = pkl.get("items") or {}
    for key in ("behavior", "foraging"):
        params = ((items.get(key) or {}).get("params") or {})
        if params.get("mouse_id") is not None:
            return str(params.get("mouse_id"))
    if pkl.get("mouseid") is not None:
        return str(pkl.get("mouseid"))
    return "unknown"


def build_sweepstim_nwbfile(pkl: dict, metadata: dict) -> NdxEventsNWBFile:
    session_desc = metadata.get("session_description", pkl.get("stage", "sweepstim_passive"))
    nwb = NdxEventsNWBFile(
        session_description=session_desc,
        identifier=metadata.get("identifier", str(uuid4())),
        session_start_time=_to_datetime(pkl.get("startdatetime") or pkl.get("start_time")),
        experimenter=metadata.get("experimenter"),
        lab=metadata.get("lab"),
        institution=metadata.get("institution"),
        notes=metadata.get("notes"),
    )
    nwb.subject = Subject(
        subject_id=_subject_id_from_pkl(pkl),
        species=metadata.get("species", "Mus musculus"),
        age=metadata.get("age"),
        sex=metadata.get("sex", "U"),
        genotype=metadata.get("genotype"),
        description=metadata.get("subject_description"),
    )
    return nwb


def _iter_sweep_rows(stimuli: list[dict], stim_ts_visual: np.ndarray):
    n_frames = len(stim_ts_visual)
    rows = []

    for block_idx, stim_obj in enumerate(stimuli):
        clip = _clip_name(stim_obj, block_idx)
        clip_label = _hed_safe_label(clip)
        sweeps = _as_sequence(stim_obj.get("sweep_order"))
        sweep_frames = _as_sequence(stim_obj.get("sweep_frames"))
        n_sweeps = min(len(sweeps), len(sweep_frames))
        runs = int(stim_obj.get("runs") or 1)
        sweeps_per_run = max(1, n_sweeps // runs) if n_sweeps else 1
        hed = f"Sensory-event, Visual-presentation, (Movie, Label/{clip_label})"

        for k in range(n_sweeps):
            sf, ef = sweep_frames[k]
            sf = int(sf)
            ef = int(ef)
            if sf >= n_frames:
                continue
            if ef <= sf:
                ef = sf + 1
            stop_frame = min(ef, n_frames - 1)
            start_time = float(stim_ts_visual[sf])
            stop_time = float(stim_ts_visual[stop_frame])
            if stop_time <= start_time:
                stop_time = start_time + (1.0 / 60.0)

            rows.append({
                "start_time": start_time,
                "stop_time": stop_time,
                "start_frame": sf,
                "stop_frame": stop_frame,
                "movie_name": clip,
                "movie_frame_index": int(sweeps[k]) if sweeps[k] is not None else -1,
                "movie_repeat": int(k // sweeps_per_run),
                "stim_block": int(block_idx),
                "epoch_name": "passive_viewing",
                "HED": hed,
            })

    rows.sort(key=lambda r: r["start_time"])
    return rows


def _build_epoch_list(stimuli: list[dict], rows: list[dict], pkl: dict) -> list[dict]:
    epochs = []

    for block_idx, stim_obj in enumerate(stimuli):
        clip = _clip_name(stim_obj, block_idx)
        seq = _as_sequence(stim_obj.get("display_sequence"))
        if not seq:
            continue
        for window in seq:
            if isinstance(window, np.ndarray):
                window = window.tolist()
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start, stop = float(window[0]), float(window[1])
            if stop <= start:
                continue
            epochs.append({
                "name": clip,
                "start": start,
                "stop": stop,
                "HED": (
                    "Experimental-procedure, "
                    "(Task, Label/passive_viewing), "
                    f"(Movie, Label/{_hed_safe_label(clip)})"
                ),
            })

    if not epochs:
        if rows:
            epochs.append({
                "name": "passive_viewing",
                "start": float(rows[0]["start_time"]),
                "stop": float(rows[-1]["stop_time"]),
                "HED": "Experimental-procedure, (Task, Label/passive_viewing)",
            })
        else:
            return []

    epochs.sort(key=lambda e: e["start"])

    session_end = max(
        float(pkl.get("session_duration", 0.0) or 0.0),
        float(rows[-1]["stop_time"]) if rows else 0.0,
        max((float(e["stop"]) for e in epochs), default=0.0),
    )

    with_spont = []
    prev = 0.0
    for ep in epochs:
        if ep["start"] > prev:
            with_spont.append({
                "name": "spontaneous",
                "start": prev,
                "stop": ep["start"],
                "HED": "Experimental-procedure, (Task, Label/spontaneous)",
            })
        with_spont.append(ep)
        prev = max(prev, ep["stop"])

    if session_end > prev:
        with_spont.append({
            "name": "spontaneous",
            "start": prev,
            "stop": session_end,
            "HED": "Experimental-procedure, (Task, Label/spontaneous)",
        })

    return with_spont


def build_stimulus_presentations_sweepstim(rows: list[dict]) -> TimeIntervals:
    return TimeIntervals(
        name="stimulus_presentations",
        description="Per-frame SweepStim movie presentations for passive sessions.",
        columns=[
            VectorData(name="start_time", description="Frame onset (s).",
                       data=[r["start_time"] for r in rows]),
            VectorData(name="stop_time", description="Frame offset (s).",
                       data=[r["stop_time"] for r in rows]),
            VectorData(name="movie_name", description="Movie clip label.",
                       data=[r["movie_name"] for r in rows]),
            VectorData(name="movie_frame_index",
                       description="Frame index within movie clip.",
                       data=[r["movie_frame_index"] for r in rows]),
            VectorData(name="movie_repeat",
                       description="Repeat index for this clip (0-based).",
                       data=[r["movie_repeat"] for r in rows]),
            VectorData(name="stim_block",
                       description="Stimulus block index from pkl top-level stimuli list.",
                       data=[r["stim_block"] for r in rows]),
            VectorData(name="start_frame",
                       description="Vsync frame index at onset.",
                       data=[r["start_frame"] for r in rows]),
            VectorData(name="stop_frame",
                       description="Vsync frame index at offset.",
                       data=[r["stop_frame"] for r in rows]),
            VectorData(name="epoch_name",
                       description="Canonical epoch label.",
                       data=[r["epoch_name"] for r in rows]),
            HedTags(name="HED",
                    description="HED tag string for this movie frame.",
                    data=[r["HED"] for r in rows]),
        ],
        id=list(range(len(rows))),
    )


def build_intervals_table_sweepstim(epoch_list: list[dict], rows: list[dict]) -> TimeIntervals:
    flat_rows = []

    for ep in epoch_list:
        flat_rows.append({
            "start_time": ep["start"],
            "stop_time": ep["stop"],
            "interval_type": "epoch",
            "label": ep["name"],
            "stimulus_presentations_id": -1,
            "HED": ep["HED"],
        })

    for sid, row in enumerate(rows):
        flat_rows.append({
            "start_time": row["start_time"],
            "stop_time": row["stop_time"],
            "interval_type": "stimulus_presentation",
            "label": row["movie_name"],
            "stimulus_presentations_id": sid,
            "HED": row["HED"],
        })

    flat_rows.sort(key=lambda r: r["start_time"])

    return TimeIntervals(
        name="intervals",
        description="Flat intervals table for SweepStim sessions (epochs + stimulus frames).",
        columns=[
            VectorData(name="start_time", description="Interval start (s).",
                       data=[r["start_time"] for r in flat_rows]),
            VectorData(name="stop_time", description="Interval stop (s).",
                       data=[r["stop_time"] for r in flat_rows]),
            VectorData(name="interval_type",
                       description="epoch or stimulus_presentation.",
                       data=[r["interval_type"] for r in flat_rows]),
            VectorData(name="label",
                       description="Epoch label or movie clip label.",
                       data=[r["label"] for r in flat_rows]),
            VectorData(name="stimulus_presentations_id",
                       description="Foreign key into stimulus_presentations (-1 if N/A).",
                       data=[r["stimulus_presentations_id"] for r in flat_rows]),
            HedTags(name="HED",
                    description="HED tag string for this interval.",
                    data=[r["HED"] for r in flat_rows]),
        ],
        id=list(range(len(flat_rows))),
    )


def build_sweepstim_sidecar() -> dict:
    """Build compact sidecar for SweepStim-specific columns."""
    return {
        "start_time": {"Description": "Frame or interval start time (s).", "HED": "Time-value/# s"},
        "stop_time": {"Description": "Frame or interval stop time (s).", "HED": "Time-value/# s"},
        "movie_name": {"Description": "Movie clip label."},
        "movie_frame_index": {"Description": "Frame index within movie clip.", "HED": "Label/movie_frame_index-#"},
        "movie_repeat": {"Description": "Repeat index within clip.", "HED": "Label/movie_repeat-#"},
        "stim_block": {"Description": "Stimulus block index from pkl top-level stimuli list.", "HED": "Label/stim_block-#"},
        "start_frame": {"Description": "Vsync frame at onset.", "HED": "Label/frame-#"},
        "stop_frame": {"Description": "Vsync frame at offset.", "HED": "Label/frame-#"},
        "interval_type": {
            "Description": "Type of interval row.",
            "Levels": {
                "epoch": "Session-level epoch row.",
                "stimulus_presentation": "Per-frame movie presentation.",
            },
        },
        "epoch_name": {"Description": "Canonical epoch label."},
        "HED": {"Description": "Hierarchical Event Descriptor tags for each row."},
        "hed_defs": {"HED": {"alldefs": ""}},
    }


def package_sweepstim_to_nwb(
    pkl_path: str | Path,
    sync_path: str | Path,
    output_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Package a passive SweepStim pkl+sync pair into an NWB file."""
    metadata = metadata or {}
    output_path = Path(output_path)

    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f, encoding="latin1")

    is_sweepstim, detail = classify_sweepstim_session(pkl)
    if not is_sweepstim:
        raise ValueError(f"Not a SweepStim session: {detail}")

    logger.info("SweepStim session detected: %s", detail)
    ts = compute_sweepstim_timestamp_alignment(pkl, sync_path)
    stimuli = pkl.get("stimuli")
    if isinstance(stimuli, np.ndarray):
        stimuli = stimuli.tolist()
    if stimuli is None:
        stimuli = []
    if not isinstance(stimuli, list) or not stimuli:
        raise ValueError("SweepStim packaging requires a non-empty top-level stimuli list")

    rows = _iter_sweep_rows(stimuli, ts["stim_ts_visual"])
    epoch_list = _build_epoch_list(stimuli, rows, pkl)

    nwb = build_sweepstim_nwbfile(pkl, metadata)
    nwb.add_lab_meta_data(HedLabMetaData(hed_schema_version=HED_SCHEMA_VERSION))

    logger.info("Adding SweepStim stimulus_presentations (%d rows)", len(rows))
    nwb.add_time_intervals(build_stimulus_presentations_sweepstim(rows))

    logger.info("Adding SweepStim flat intervals table (%d epochs)", len(epoch_list))
    nwb.add_time_intervals(build_intervals_table_sweepstim(epoch_list, rows))

    logger.info("Adding running speed for SweepStim")
    add_running_speed(nwb, pkl, ts["stim_vsync_fall"])

    logger.info("Writing SweepStim NWB to %s", output_path)
    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwb)

    sidecar_path = output_path.with_suffix(".events.json")
    with open(sidecar_path, "w") as f:
        json.dump(build_sweepstim_sidecar(), f, indent=2, ensure_ascii=False)

    logger.info("Wrote SweepStim sidecar JSON to %s", sidecar_path)
    return output_path
