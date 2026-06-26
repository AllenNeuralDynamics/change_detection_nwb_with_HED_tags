"""Summarize packaged change-detection NWB files into two tidy tables.

Loops over every ``*.nwb`` file under a results directory and builds:

1. ``session_metrics.csv`` — one row per session with identifying metadata
   (subject, acquisition date, session type, rig metadata...) plus behavioral
   summary metrics: lick counts by classification, reward counts/volume,
   numbers of changes/omissions, trial-outcome counts (hit/miss/FA/CR/...),
   detection performance (hit/FA rate, d-prime), lick-bout count, and running
   speed statistics.

2. ``session_task_parameters.csv`` — one row per session with the same
   identifying metadata plus every field of the ``ChangeDetectionTaskParameters``
   container stored on each NWB.

Usage
-----
    python summarize_sessions.py [RESULTS_DIR] [OUT_DIR]

RESULTS_DIR (where the NWBs are) defaults to ``/results`` (env ``RESULTS_DIR``
honored). OUT_DIR (where the CSVs are written) defaults to the ``summaries/``
folder in the capsule workspace -- a tracked, non-gitignored location so the
tables actually show up in the VS Code file explorer (``/results`` and
``/scratch`` are both gitignored and therefore hidden).
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Extension types must be imported so pynwb can resolve them on read.
import ndx_events  # noqa: F401
import ndx_change_detection_task  # noqa: F401
from ndx_events import EventsTable
from pynwb import NWBHDF5IO

logger = logging.getLogger("summarize_sessions")

# Lick classifications we expect (from build_events_and_intervals); any extra
# value found in a file is added as its own column automatically.
LICK_CLASSES = [
    "hit", "false_alarm", "abort", "early", "late", "consumption", "spontaneous",
]


def build_asset_map(data_dir: Path) -> dict[str, str]:
    """Map session id -> originating data-asset folder name under ``data_dir``.

    Session ids are derived from the stim pkl exactly as ``run_capsule.py``
    discovers them (``<id>_stim.pkl`` or bare ``<id>.pkl``); the asset name is
    the top-level folder beneath ``data_dir`` that contains the pkl.
    """
    asset_of: dict[str, str] = {}
    if not data_dir.exists():
        return asset_of
    for pkl in sorted(data_dir.rglob("*.pkl")):
        if pkl.name.endswith("_stim.pkl"):
            sid = pkl.name[: -len("_stim.pkl")]
        elif pkl.stem.isdigit():
            sid = pkl.stem
        else:
            continue
        try:
            asset = pkl.relative_to(data_dir).parts[0]
        except (ValueError, IndexError):
            continue
        # Prefer a *_stim.pkl over a bare <id>.pkl for the same id (matches
        # run_capsule's discovery precedence).
        if sid in asset_of and not pkl.name.endswith("_stim.pkl"):
            continue
        asset_of[sid] = asset
    return asset_of


def _get_events_table(nwb) -> EventsTable | None:
    """The ndx-events EventsTable is not a first-class NWBFile attribute."""
    for obj in nwb.objects.values():
        if isinstance(obj, EventsTable):
            return obj
    return None


def _safe_rate(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else np.nan


def _dprime(hit_rate: float, fa_rate: float, n_go: int, n_catch: int) -> float:
    """Signal-detection d-prime with a log-linear correction for 0/1 rates."""
    from scipy.stats import norm  # local import; scipy is in the env

    if not n_go or not n_catch:
        return np.nan
    # Hautus log-linear correction avoids infinities at perfect/zero performance.
    h = (hit_rate * n_go + 0.5) / (n_go + 1)
    f = (fa_rate * n_catch + 0.5) / (n_catch + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def _identity(nwb, session_id: str, params, asset_name: str | None) -> dict:
    """Identifying metadata shared by both output tables."""
    subj = nwb.subject
    start = nwb.session_start_time
    return {
        "session_id": session_id,
        # asset_name kept near the front so it lines up against subject_id /
        # acquisition_date for cross-checking against the data-asset folder.
        "asset_name": asset_name,
        "subject_id": getattr(subj, "subject_id", None) if subj else None,
        "acquisition_date": start.isoformat() if start is not None else None,
        # session_type from the task params is the canonical camstim "stage";
        # session_description is the human-readable mirror of it.
        "session_type": getattr(params, "session_type", None) if params else None,
        "session_description": nwb.session_description,
        "stimulus_class": getattr(params, "stimulus_class", None) if params else None,
        "image_set_name": getattr(params, "image_set_name", None) if params else None,
        "species": getattr(subj, "species", None) if subj else None,
        "sex": getattr(subj, "sex", None) if subj else None,
        "age": getattr(subj, "age", None) if subj else None,
        "genotype": getattr(subj, "genotype", None) if subj else None,
        "institution": nwb.institution,
        "lab": nwb.lab,
        "experimenter": (", ".join(nwb.experimenter)
                         if nwb.experimenter else None),
        "identifier": nwb.identifier,
    }


def _running_stats(nwb) -> dict:
    """Mean/std/median/max of running speed (cm/s), NaN-aware."""
    out = {k: np.nan for k in (
        "running_speed_mean", "running_speed_std",
        "running_speed_median", "running_speed_max")}
    mod = nwb.processing.get("running")
    if mod is None or "speed" not in mod.data_interfaces:
        return out
    speed = np.asarray(mod["speed"].data[:], dtype=float)
    if speed.size:
        out["running_speed_mean"] = float(np.nanmean(speed))
        out["running_speed_std"] = float(np.nanstd(speed))
        out["running_speed_median"] = float(np.nanmedian(speed))
        out["running_speed_max"] = float(np.nanmax(speed))
    return out


def _event_metrics(nwb) -> dict:
    """Counts derived from the discrete EventsTable (licks/rewards/changes)."""
    out: dict = {}
    et = _get_events_table(nwb)
    if et is None:
        return out
    df = et.to_dataframe()
    etype = df["event_type"]

    is_lick = etype == "lick"
    out["n_licks"] = int(is_lick.sum())
    out["n_rewards"] = int((etype == "reward").sum())
    out["n_image_changes"] = int((etype == "image_change").sum())
    out["n_image_omissions"] = int((etype == "image_omission").sum())

    # Reward volume + earned/auto split.
    rew = df[etype == "reward"]
    out["total_reward_volume_ml"] = (
        float(np.nansum(rew["reward_volume"].astype(float))) if len(rew) else 0.0)
    if "reward_type" in df.columns and len(rew):
        out["n_earned_rewards"] = int((rew["reward_type"] == "earned").sum())
        out["n_auto_rewards"] = int((rew["reward_type"] == "auto_reward").sum())

    # Licks by classification.
    licks = df[is_lick]
    seen = set()
    if "lick_classification" in df.columns and len(licks):
        counts = licks["lick_classification"].value_counts()
        for cls in LICK_CLASSES + [c for c in counts.index if c not in LICK_CLASSES]:
            if cls in ("n/a",):
                continue
            out[f"n_lick_{cls}"] = int(counts.get(cls, 0))
            seen.add(cls)
    # Ensure canonical columns always present even if zero.
    for cls in LICK_CLASSES:
        out.setdefault(f"n_lick_{cls}", 0)

    # Lick bouts.
    if "lick_bouts" in df.columns and len(licks):
        out["n_lick_bouts"] = int((licks["lick_bouts"] == "bout_start").sum())
    else:
        out["n_lick_bouts"] = 0
    return out


def _trial_metrics(nwb) -> dict:
    """Trial-outcome counts and detection performance from nwb.trials."""
    out: dict = {}
    if nwb.trials is None:
        return out
    t = nwb.trials.to_dataframe()
    out["n_trials"] = int(len(t))

    bool_cols = ["go", "catch", "auto_rewarded", "aborted", "hit", "miss",
                 "false_alarm", "correct_reject", "warm_up"]
    for c in bool_cols:
        if c in t.columns:
            out[f"n_{c}"] = int(t[c].astype(bool).sum())

    n_hit = out.get("n_hit", 0)
    n_miss = out.get("n_miss", 0)
    n_fa = out.get("n_false_alarm", 0)
    n_cr = out.get("n_correct_reject", 0)
    n_go_resp = n_hit + n_miss
    n_catch_resp = n_fa + n_cr
    hit_rate = _safe_rate(n_hit, n_go_resp)
    fa_rate = _safe_rate(n_fa, n_catch_resp)
    out["hit_rate"] = hit_rate
    out["false_alarm_rate"] = fa_rate
    out["dprime"] = _dprime(hit_rate, fa_rate, n_go_resp, n_catch_resp)

    # Mean response latency on hit trials (change -> first lick).
    if "response_latency" in t.columns and "hit" in t.columns:
        lat = t.loc[t["hit"].astype(bool), "response_latency"].astype(float)
        out["mean_hit_response_latency_sec"] = (
            float(np.nanmean(lat)) if len(lat) else np.nan)

    # Session duration from trial span.
    if {"start_time", "stop_time"}.issubset(t.columns) and len(t):
        dur = float(t["stop_time"].max() - t["start_time"].min())
        out["session_duration_min"] = dur / 60.0
    return out


def _stimulus_metrics(nwb) -> dict:
    """Counts from the stimulus_presentations interval table."""
    out: dict = {}
    sp = nwb.intervals.get("stimulus_presentations")
    if sp is None:
        return out
    df = sp.to_dataframe()
    out["n_stimulus_presentations"] = int(len(df))
    if "omitted" in df.columns:
        out["n_omitted_flashes"] = int(df["omitted"].astype(bool).sum())
    if "is_change" in df.columns:
        out["n_change_flashes"] = int(df["is_change"].astype(bool).sum())
    return out


def _task_parameter_fields(params) -> dict:
    """Every scalar field stored on the ChangeDetectionTaskParameters."""
    if params is None:
        return {}
    out: dict = {}
    for field in params.fields:
        if field in ("name",):
            continue
        val = getattr(params, field, None)
        # response_window_sec is a 2-element array → split for flat CSV.
        if field == "response_window_sec" and val is not None and len(val) == 2:
            out["response_window_start_sec"] = float(val[0])
            out["response_window_stop_sec"] = float(val[1])
            continue
        if isinstance(val, (list, tuple, np.ndarray)):
            val = ", ".join(str(v) for v in val)
        out[field] = val
    return out


def summarize_file(path: Path, asset_map: dict[str, str] | None = None) -> tuple[dict, dict]:
    """Return (metrics_row, task_params_row) for one NWB file."""
    session_id = path.stem
    asset_name = (asset_map or {}).get(session_id)
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        params = nwb.lab_meta_data.get("task_parameters")
        ident = _identity(nwb, session_id, params, asset_name)

        metrics = dict(ident)
        metrics.update(_event_metrics(nwb))
        metrics.update(_trial_metrics(nwb))
        metrics.update(_stimulus_metrics(nwb))
        metrics.update(_running_stats(nwb))

        task_row = dict(ident)
        task_row.update(_task_parameter_fields(params))
    return metrics, task_row


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")

    results_dir = Path(argv[1]) if len(argv) > 1 else Path(
        os.environ.get("RESULTS_DIR", "/results"))
    # Default to the workspace `summaries/` dir: tracked + not gitignored, so
    # the CSVs are visible in the file explorer (unlike /results and /scratch).
    default_out = Path(__file__).resolve().parent.parent / "summaries"
    out_dir = Path(argv[2]) if len(argv) > 2 else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    nwb_files = sorted(results_dir.rglob("*.nwb"))
    if not nwb_files:
        raise SystemExit(f"ERROR: no *.nwb files found under {results_dir}")
    logger.info("Found %d NWB file(s) under %s", len(nwb_files), results_dir)

    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    asset_map = build_asset_map(data_dir)
    logger.info("Mapped %d session(s) to data assets under %s",
                len(asset_map), data_dir)

    metric_rows, task_rows = [], []
    for path in nwb_files:
        try:
            m, t = summarize_file(path, asset_map)
            metric_rows.append(m)
            task_rows.append(t)
            logger.info("Summarized %s", path.name)
        except Exception:
            logger.exception("FAILED to summarize %s — skipping.", path.name)

    if not metric_rows:
        raise SystemExit("ERROR: no NWB files could be summarized.")

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["subject_id", "acquisition_date"]).reset_index(drop=True)
    task_df = pd.DataFrame(task_rows).sort_values(
        ["subject_id", "acquisition_date"]).reset_index(drop=True)

    metrics_path = out_dir / "session_metrics.csv"
    task_path = out_dir / "session_task_parameters.csv"
    metrics_df.to_csv(metrics_path, index=False)
    task_df.to_csv(task_path, index=False)

    logger.info("Wrote %s  (%d sessions x %d columns)",
                metrics_path, *metrics_df.shape)
    logger.info("Wrote %s  (%d sessions x %d columns)",
                task_path, *task_df.shape)


if __name__ == "__main__":
    main(sys.argv)
