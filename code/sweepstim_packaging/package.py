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
from hdmf_zarr.nwb import NWBZarrIO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from .classify import classify_sweepstim_session
from .running import add_running_speed
from .timestamp_alignment import compute_sweepstim_timestamp_alignment


logger = logging.getLogger(__name__)
HED_SCHEMA_VERSION = "8.3.0"

# camstim sweep-dimension name -> snake_case NWB column name. Parametric-sweep
# stimuli (gratings) store per-condition parameters in ``sweep_table``, with the
# column order given by ``dimnames``; we surface each dimension as its own column.
_DIMNAME_COLUMN = {
    "Contrast": "contrast",
    "TF": "temporal_frequency",
    "SF": "spatial_frequency",
    "Ori": "orientation",
    "Phase": "phase",
    "Pos": "position",
    "Size": "size",
}

# A movie block's sole "dimension" is which image/frame is shown.
_MOVIE_DIMNAMES = ["ReplaceImage"]

# Preferred left-to-right order for the stim-type-specific columns, which are
# only emitted when the session actually uses them (session-driven schema).
_EXTRA_COLUMN_ORDER = [
    "orientation", "spatial_frequency", "temporal_frequency", "contrast",
    "phase", "position", "size",
    "condition_index", "condition_repeat",
    "movie_frame_index", "movie_repeat",
]

_COLUMN_DESCRIPTIONS = {
    "orientation": "Grating orientation (degrees).",
    "spatial_frequency": "Grating spatial frequency (cycles/degree).",
    "temporal_frequency": "Grating temporal frequency (Hz).",
    "contrast": "Grating contrast (0-1).",
    "phase": "Grating phase.",
    "position": "Stimulus position.",
    "size": "Stimulus size.",
    "condition_index": "Index into the block's sweep_table (grating condition id).",
    "condition_repeat": "0-based count of prior presentations of this condition.",
    "movie_frame_index": "Frame index within movie clip.",
    "movie_repeat": "Repeat index for this clip (0-based).",
}

# BIDS-sidecar HED value templates for the optional columns; '#' is the per-row
# value placeholder. Kept in the safe Label/ extension form used elsewhere.
_COLUMN_HED_TEMPLATE = {
    "orientation": "Label/orientation-#",
    "spatial_frequency": "Label/spatial_frequency-#",
    "temporal_frequency": "Label/temporal_frequency-#",
    "contrast": "Label/contrast-#",
    "phase": "Label/phase-#",
    "position": "Label/position-#",
    "size": "Label/size-#",
    "condition_index": "Label/condition_index-#",
    "condition_repeat": "Label/condition_repeat-#",
    "movie_frame_index": "Label/movie_frame_index-#",
    "movie_repeat": "Label/movie_repeat-#",
}

# Generic columns present on every presentation row, in output order.
_BASE_PRESENTATION_SPECS = [
    ("start_time", "Frame or presentation onset (s)."),
    ("stop_time", "Frame or presentation offset (s)."),
    ("epoch_name", "Epoch label: movie clip or grating stimulus name."),
    ("stim_type", "Stimulus class (e.g. GratingStim, ImageStimNumpyuByte)."),
    ("stim_block", "Stimulus block index from pkl top-level stimuli list."),
    ("start_frame", "Vsync frame index at onset."),
    ("stop_frame", "Vsync frame index at offset."),
]


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


def _dimname_to_column(dimname) -> str:
    """Map a camstim sweep dimension name to a snake_case column name."""
    if dimname in _DIMNAME_COLUMN:
        return _DIMNAME_COLUMN[dimname]
    norm = re.sub(r"[^a-z0-9]+", "_", str(dimname).lower()).strip("_")
    return norm or "param"


def _is_grating_block(stim_obj: dict) -> bool:
    """True for a parametric-sweep stimulus (e.g. drifting gratings).

    Such blocks carry a ``sweep_table`` of per-condition parameter tuples and
    real parameter ``dimnames`` (Contrast/TF/SF/Ori/...), unlike a movie block
    whose only dimension is ``ReplaceImage`` and which has no sweep_table. The
    ``frame_list`` value for these blocks is a condition index into sweep_table,
    not a movie frame index.
    """
    st = stim_obj.get("sweep_table")
    if st is None or len(st) == 0:
        return False
    return list(_as_sequence(stim_obj.get("dimnames"))) != _MOVIE_DIMNAMES


def _stim_type(stim_obj: dict) -> str:
    """Best-effort stimulus class from the pickled ``stim`` repr string, e.g.
    'GratingStim' or 'ImageStimNumpyuByte'."""
    rep = stim_obj.get("stim")
    if isinstance(rep, str):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", rep)
        if m:
            return m.group(1)
    return "GratingStim" if _is_grating_block(stim_obj) else "MovieStim"


def _grating_params(sweep_table: list, dim_columns: list, value: int) -> dict:
    """Resolve a condition index into ``{column: float value}`` via sweep_table."""
    if not 0 <= value < len(sweep_table):
        return {}
    out = {}
    for col, pv in zip(dim_columns, _as_sequence(sweep_table[value])):
        try:
            out[col] = float(pv)
        except (TypeError, ValueError):
            out[col] = np.nan
    return out


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
    # AIND subject metadata threaded in from the primary subject.json (see
    # aind_metadata.subject_fields_for_nwb): also populate strain + dob. ``age``
    # is an ISO-8601 duration ("P142D"); ``date_of_birth`` is "YYYY-MM-DD".
    dob = metadata.get("date_of_birth")
    if isinstance(dob, str):
        try:
            dob = datetime.date.fromisoformat(dob)
        except ValueError:
            dob = None
    # pynwb Subject.date_of_birth requires a (tz-aware) datetime, not a date.
    if isinstance(dob, datetime.date) and not isinstance(dob, datetime.datetime):
        dob = datetime.datetime(dob.year, dob.month, dob.day,
                                tzinfo=datetime.timezone.utc)
    nwb.subject = Subject(
        subject_id=_subject_id_from_pkl(pkl),
        species=metadata.get("species", "Mus musculus"),
        age=metadata.get("age"),
        sex=metadata.get("sex", "U"),
        genotype=metadata.get("genotype"),
        strain=metadata.get("strain"),
        date_of_birth=dob,
        description=metadata.get("subject_description"),
    )
    return nwb


def _block_rows_from_frame_list(stim_obj: dict, block_idx: int,
                                stim_ts_visual: np.ndarray) -> list[dict]:
    """Build per-presentation rows for one block from camstim ``frame_list``.

    ``frame_list`` is indexed by *global* display frame (60 Hz from session
    start, so it aligns 1:1 with the vsync timebase in ``stim_ts_visual``) and
    its value is the on-screen identifier — a grating condition index, or a
    movie frame index — with ``-1`` marking frames where this block is not on
    screen. This is the authoritative record of what was displayed *when*, and
    it already respects each block's ``display_sequence`` windows (the gaps show
    up as ``-1``), so no local→global frame conversion is needed.

    Each maximal run of a constant value ``>= 0`` is one presentation: gratings
    hold a condition for the whole sweep (e.g. 120 frames = 2 s, separated by
    blank ``-1`` gaps); movies show each frame for 2 display frames (30 Hz).

    For a grating block the run's value is a *condition index* into
    ``sweep_table``; we resolve it through ``dimnames`` into the stimulus
    parameters (orientation, contrast, spatial/temporal frequency, ...) and emit
    them as columns. For a movie block the value is the movie frame index.
    """
    n_frames = len(stim_ts_visual)
    clip = _clip_name(stim_obj, block_idx)
    clip_label = _hed_safe_label(clip)
    stim_type = _stim_type(stim_obj)
    grating = _is_grating_block(stim_obj)
    if grating:
        dim_columns = [_dimname_to_column(d)
                       for d in _as_sequence(stim_obj.get("dimnames"))]
        sweep_table = list(stim_obj.get("sweep_table") or [])
        hed = f"Sensory-event, Visual-presentation, (Image, Label/{clip_label})"
    else:
        dim_columns, sweep_table = [], []
        hed = f"Sensory-event, Visual-presentation, (Movie, Label/{clip_label})"

    fl = np.asarray(_as_sequence(stim_obj.get("frame_list")))
    if fl.size == 0:
        return []

    # Run boundaries: a run spans global frames [start, stop) with fl constant.
    change = np.flatnonzero(np.diff(fl)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [fl.size]))

    rows = []
    repeat_counter: dict[int, int] = {}
    for start_frame, stop_frame in zip(starts.tolist(), stops.tolist()):
        value = int(fl[start_frame])
        if value < 0:            # blank / block not on screen
            continue
        if start_frame >= n_frames:
            continue
        # Offset = onset of the next display frame after the run (contiguous).
        stop_frame = min(stop_frame, n_frames - 1)
        start_time = float(stim_ts_visual[start_frame])
        stop_time = float(stim_ts_visual[stop_frame])
        if stop_time <= start_time:
            stop_time = start_time + (1.0 / 60.0)

        repeat = repeat_counter.get(value, 0)
        repeat_counter[value] = repeat + 1
        row = {
            "start_time": start_time,
            "stop_time": stop_time,
            "start_frame": int(start_frame),
            "stop_frame": int(stop_frame),
            "stim_type": stim_type,
            "stim_block": int(block_idx),
            "epoch_name": clip,
            "HED": hed,
        }
        if grating:
            row["condition_index"] = value
            row["condition_repeat"] = repeat
            row.update(_grating_params(sweep_table, dim_columns, value))
        else:
            row["movie_frame_index"] = value
            row["movie_repeat"] = repeat
        rows.append(row)
    return rows


def _block_rows_from_sweep_frames(stim_obj: dict, block_idx: int,
                                  stim_ts_visual: np.ndarray) -> list[dict]:
    """Fallback for blocks with no ``frame_list``.

    Treats ``sweep_frames`` as global vsync indices. This is only correct for a
    single-block session that starts at frame 0; for multi-block/interleaved
    sessions it mis-places frames, so it is used only when ``frame_list`` is
    absent and a warning is logged by the caller.
    """
    n_frames = len(stim_ts_visual)
    clip = _clip_name(stim_obj, block_idx)
    clip_label = _hed_safe_label(clip)
    stim_type = _stim_type(stim_obj)
    grating = _is_grating_block(stim_obj)
    if grating:
        dim_columns = [_dimname_to_column(d)
                       for d in _as_sequence(stim_obj.get("dimnames"))]
        sweep_table = list(stim_obj.get("sweep_table") or [])
        hed = f"Sensory-event, Visual-presentation, (Image, Label/{clip_label})"
    else:
        dim_columns, sweep_table = [], []
        hed = f"Sensory-event, Visual-presentation, (Movie, Label/{clip_label})"
    sweeps = _as_sequence(stim_obj.get("sweep_order"))
    sweep_frames = _as_sequence(stim_obj.get("sweep_frames"))
    n_sweeps = min(len(sweeps), len(sweep_frames))
    runs = int(stim_obj.get("runs") or 1)
    sweeps_per_run = max(1, n_sweeps // runs) if n_sweeps else 1

    rows = []
    repeat_counter: dict[int, int] = {}
    for k in range(n_sweeps):
        value = int(sweeps[k]) if sweeps[k] is not None else -1
        if value < 0:            # blank sweep
            continue
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
        repeat = repeat_counter.get(value, 0)
        repeat_counter[value] = repeat + 1
        row = {
            "start_time": start_time,
            "stop_time": stop_time,
            "start_frame": sf,
            "stop_frame": stop_frame,
            "stim_type": stim_type,
            "stim_block": int(block_idx),
            "epoch_name": clip,
            "HED": hed,
        }
        if grating:
            row["condition_index"] = value
            row["condition_repeat"] = repeat
            row.update(_grating_params(sweep_table, dim_columns, value))
        else:
            row["movie_frame_index"] = value
            row["movie_repeat"] = int(k // sweeps_per_run)
        rows.append(row)
    return rows


def _iter_sweep_rows(stimuli: list[dict], stim_ts_visual: np.ndarray):
    rows = []
    for block_idx, stim_obj in enumerate(stimuli):
        frame_list = stim_obj.get("frame_list")
        if frame_list is not None and len(frame_list) > 0:
            rows.extend(_block_rows_from_frame_list(stim_obj, block_idx, stim_ts_visual))
        else:
            logger.warning(
                "Block %d (%s) has no frame_list; falling back to raw "
                "sweep_frames — timing may be wrong for multi-block sessions.",
                block_idx, _clip_name(stim_obj, block_idx))
            rows.extend(_block_rows_from_sweep_frames(stim_obj, block_idx, stim_ts_visual))

    rows.sort(key=lambda r: r["start_time"])
    return rows


def _build_epoch_list(stimuli: list[dict], rows: list[dict], pkl: dict,
                      stim_ts_visual: np.ndarray, fps: float) -> list[dict]:
    # display_sequence windows are in seconds on the stimulus clock (session
    # start = 0). Convert them to the sync/vsync timebase used by the
    # presentations (start = stim_ts_visual[0]) so epochs and frames line up:
    # seconds -> global display frame (* fps) -> vsync time.
    n_frames = len(stim_ts_visual)
    session_start = float(stim_ts_visual[0]) if n_frames else 0.0

    def sec_to_time(sec) -> float:
        frame = int(round(float(sec) * fps))
        frame = max(0, min(frame, n_frames - 1))
        return float(stim_ts_visual[frame])

    epochs = []
    for block_idx, stim_obj in enumerate(stimuli):
        clip = _clip_name(stim_obj, block_idx)
        media = "Image" if _is_grating_block(stim_obj) else "Movie"
        seq = _as_sequence(stim_obj.get("display_sequence"))
        if not seq:
            continue
        for window in seq:
            if isinstance(window, np.ndarray):
                window = window.tolist()
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start, stop = sec_to_time(window[0]), sec_to_time(window[1])
            if stop <= start:
                continue
            epochs.append({
                "name": clip,
                "start": start,
                "stop": stop,
                "HED": (
                    "Experimental-procedure, "
                    "(Task, Label/passive_viewing), "
                    f"({media}, Label/{_hed_safe_label(clip)})"
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
        float(rows[-1]["stop_time"]) if rows else 0.0,
        max((float(e["stop"]) for e in epochs), default=0.0),
    )

    with_spont = []
    prev = session_start
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


def _presentation_extra_columns(rows: list[dict]) -> list[str]:
    """Session-driven optional columns: those actually populated by some row,
    ordered by preference then alphabetically. Movie sessions get
    movie_frame_index/movie_repeat; grating sessions get the parameter columns;
    a mixed session gets the union (NaN-filled where a row doesn't use one)."""
    base = {name for name, _ in _BASE_PRESENTATION_SPECS} | {"HED"}
    present: set[str] = set()
    for r in rows:
        present.update(r.keys())
    present -= base
    ordered = [c for c in _EXTRA_COLUMN_ORDER if c in present]
    ordered += sorted(present - set(_EXTRA_COLUMN_ORDER))
    return ordered


def build_stimulus_presentations_sweepstim(rows: list[dict]) -> TimeIntervals:
    columns = [
        VectorData(name=name, description=desc, data=[r[name] for r in rows])
        for name, desc in _BASE_PRESENTATION_SPECS
    ]
    for name in _presentation_extra_columns(rows):
        columns.append(VectorData(
            name=name,
            description=_COLUMN_DESCRIPTIONS.get(name, f"{name} stimulus parameter."),
            data=[r.get(name, np.nan) for r in rows]))
    columns.append(HedTags(name="HED",
                           description="HED tag string for this presentation.",
                           data=[r["HED"] for r in rows]))
    return TimeIntervals(
        name="stimulus_presentations",
        description=("Per-presentation SweepStim stimulus table "
                     "(movie frames and/or parametric gratings)."),
        columns=columns,
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
            "label": row["epoch_name"],
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


def build_sweepstim_sidecar(rows: list[dict]) -> dict:
    """Build compact BIDS-style sidecar describing the columns actually written.

    Optional (stim-type-specific) columns are included only when the session
    uses them, matching the session-driven presentations table.
    """
    sidecar = {
        "start_time": {"Description": "Frame or interval start time (s).", "HED": "Time-value/# s"},
        "stop_time": {"Description": "Frame or interval stop time (s).", "HED": "Time-value/# s"},
        "epoch_name": {"Description": "Epoch label: movie clip or grating stimulus name."},
        "stim_type": {"Description": "Stimulus class (e.g. GratingStim, ImageStimNumpyuByte)."},
        "stim_block": {"Description": "Stimulus block index from pkl top-level stimuli list.", "HED": "Label/stim_block-#"},
        "start_frame": {"Description": "Vsync frame at onset.", "HED": "Label/frame-#"},
        "stop_frame": {"Description": "Vsync frame at offset.", "HED": "Label/frame-#"},
    }
    for col in _presentation_extra_columns(rows):
        entry = {"Description": _COLUMN_DESCRIPTIONS.get(col, f"{col} stimulus parameter.")}
        if col in _COLUMN_HED_TEMPLATE:
            entry["HED"] = _COLUMN_HED_TEMPLATE[col]
        sidecar[col] = entry
    sidecar["interval_type"] = {
        "Description": "Type of interval row.",
        "Levels": {
            "epoch": "Session-level epoch row.",
            "stimulus_presentation": "Per-presentation stimulus row.",
        },
    }
    sidecar["HED"] = {"Description": "Hierarchical Event Descriptor tags for each row."}
    sidecar["hed_defs"] = {"HED": {"alldefs": ""}}
    return sidecar


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

    fps = float(pkl.get("fps") or (stimuli[0].get("fps") if stimuli else None) or 60.0)
    rows = _iter_sweep_rows(stimuli, ts["stim_ts_visual"])
    epoch_list = _build_epoch_list(stimuli, rows, pkl, ts["stim_ts_visual"], fps)

    nwb = build_sweepstim_nwbfile(pkl, metadata)
    nwb.add_lab_meta_data(HedLabMetaData(hed_schema_version=HED_SCHEMA_VERSION))

    logger.info("Adding SweepStim stimulus_presentations (%d rows)", len(rows))
    nwb.add_time_intervals(build_stimulus_presentations_sweepstim(rows))

    logger.info("Adding SweepStim flat intervals table (%d epochs)", len(epoch_list))
    nwb.add_time_intervals(build_intervals_table_sweepstim(epoch_list, rows))

    logger.info("Adding running speed for SweepStim")
    add_running_speed(nwb, pkl, ts["stim_vsync_fall"])

    logger.info("Writing SweepStim NWB-Zarr to %s", output_path)
    with NWBZarrIO(str(output_path), "w") as io:
        io.write(nwb)

    # output_path may be "<...>/behavior.nwb.zarr" (double extension); strip all
    # suffixes off the stem rather than with_suffix (which only replaces .zarr).
    sidecar_stem = output_path.name.split(".")[0]
    sidecar_path = output_path.parent / f"{sidecar_stem}.events.json"
    with open(sidecar_path, "w") as f:
        json.dump(build_sweepstim_sidecar(rows), f, indent=2, ensure_ascii=False)

    logger.info("Wrote SweepStim sidecar JSON to %s", sidecar_path)
    # monitor_delay_sec is None: passive SweepStim packaging does not measure a
    # photodiode monitor delay (unlike change-detection).
    return {
        "output_path": output_path,
        "monitor_delay_sec": None,
        "hed_schema_version": HED_SCHEMA_VERSION,
    }
