"""SweepStim timestamp alignment utilities.

These are intentionally separate from change-detection builders.
"""

from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np


def _get_edges(bits, counters, line_labels, line_name, edge_type, sample_rate):
    """Extract edge times from sync data for a given line."""
    aliases = {
        "stim_vsync": ("stim_vsync", "vsync_stim"),
        "vsync_stim": ("stim_vsync", "vsync_stim"),
        "2p_vsync": ("2p_vsync", "vsync_2p"),
        "vsync_2p": ("2p_vsync", "vsync_2p"),
        "acq_trigger": ("acq_trigger", "stim_running"),
        "stim_running": ("acq_trigger", "stim_running"),
        "stim_photodiode": ("stim_photodiode", "photodiode"),
    }
    candidates = aliases.get(line_name, (line_name,))

    bit_idx = None
    for cand in candidates:
        if cand in line_labels:
            bit_idx = line_labels.index(cand)
            break
    if bit_idx is None:
        raise ValueError(
            f"No alias for line {line_name!r} found in sync labels: {line_labels}")

    line_state = (bits >> bit_idx) & 1
    changes = np.diff(line_state.astype(np.int8))
    if edge_type == "falling":
        edge_indices = np.where(changes == -1)[0] + 1
    elif edge_type == "rising":
        edge_indices = np.where(changes == 1)[0] + 1
    else:
        edge_indices = np.where(changes != 0)[0] + 1

    return counters[edge_indices].astype(np.float64) / float(sample_rate)


def resolve_frame_count(pkl: dict) -> int:
    """Resolve expected frame count from SweepStim/session structure."""
    items = pkl.get("items") or {}
    behavior = items.get("behavior") or {}
    foraging = items.get("foraging") or {}

    if isinstance(behavior.get("intervalsms"), (list, tuple, np.ndarray)):
        return len(behavior["intervalsms"]) + 1
    if isinstance(foraging.get("intervalsms"), (list, tuple, np.ndarray)):
        return len(foraging["intervalsms"]) + 1
    if isinstance(pkl.get("intervalsms"), (list, tuple, np.ndarray)):
        return len(pkl["intervalsms"]) + 1

    # Last-resort fallbacks in historical camstim files.
    if pkl.get("vsynccount"):
        return int(pkl["vsynccount"])
    if pkl.get("total_frames"):
        return int(pkl["total_frames"])

    raise KeyError("Could not resolve frame count from pkl")


def compute_sweepstim_timestamp_alignment(pkl: dict, sync_path: str | Path) -> dict:
    """Compute visual/behavioral frame timestamps for SweepStim sessions."""
    with h5py.File(sync_path, "r") as sync_file:
        meta = ast.literal_eval(sync_file["meta"][()].decode("utf-8"))
        sample_rate = meta["ni_daq"].get("sample_rate", meta["ni_daq"].get("counter_output_freq"))
        line_labels = meta["line_labels"]
        sync_data = sync_file["data"][:]

    counters = sync_data[:, 0]
    bits = sync_data[:, 1]

    stim_vsync_fall = _get_edges(bits, counters, line_labels, "stim_vsync", "falling", sample_rate)
    n_pkl_frames = resolve_frame_count(pkl)
    stim_vsync_fall = stim_vsync_fall[:n_pkl_frames]

    all_pd_edges = _get_edges(bits, counters, line_labels, "stim_photodiode", "both", sample_rate)
    all_pd_edges = np.sort(all_pd_edges)

    monitor_delay = 0.0356
    pd_diffs = np.diff(all_pd_edges)
    regular_mask = (pd_diffs > 0.8) & (pd_diffs < 1.2)
    regular_indices = np.where(regular_mask)[0]

    if len(regular_indices) > 10:
        first_regular = regular_indices[0]
        last_regular = regular_indices[-1] + 1
        clean_pd = all_pd_edges[first_regular:last_regular + 1].copy()
        while True:
            diffs = np.diff(clean_pd)
            anomalies = np.where(diffs < 0.5)[0]
            if len(anomalies) == 0:
                break
            clean_pd = np.delete(clean_pd, anomalies[-1] + 1)

        transitions = stim_vsync_fall[::60]
        if len(clean_pd) and len(transitions):
            nearest_idx = int(np.argmin(np.abs(transitions - clean_pd[0])))
            n_match = min(len(clean_pd), len(transitions) - nearest_idx)
            if n_match > 0:
                delays = clean_pd[:n_match] - transitions[nearest_idx:nearest_idx + n_match]
                valid = (delays > 0) & (delays < 0.07)
                if np.sum(valid) > 10:
                    monitor_delay = float(np.mean(delays[valid]))
                else:
                    monitor_delay = float(np.median(delays))
                    if not (0 < monitor_delay < 0.07):
                        monitor_delay = 0.0356

    return {
        "stim_ts_visual": stim_vsync_fall + monitor_delay,
        "stim_ts_behavioral": stim_vsync_fall,
        "monitor_delay": monitor_delay,
        "stim_vsync_fall": stim_vsync_fall,
    }
