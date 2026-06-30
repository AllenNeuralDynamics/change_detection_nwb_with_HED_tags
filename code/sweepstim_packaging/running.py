"""Running-speed extraction for SweepStim packaging.

Standalone implementation so SweepStim packaging does not depend on
change-detection modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.signal as _signal
from scipy.stats import zscore as _zscore

from pynwb import ProcessingModule
from pynwb.base import TimeSeries


_WHEEL_DIAM_CM = 6.5 * 2.54
_WHEEL_RUNNING_RADIUS_CM = 0.5 * (2.0 * _WHEEL_DIAM_CM / 3.0)


def _shift(arr, periods=1, fill_value=np.nan):
    shifted = np.roll(arr, periods).astype(float)
    shifted[:periods] = fill_value
    return shifted


def _identify_wraps(vsig, min_threshold=1.5, max_threshold=3.5):
    shifted = _shift(np.asarray(vsig))
    vsig = np.asarray(vsig)
    with np.errstate(invalid="ignore"):
        pos = np.nonzero((vsig < min_threshold) & (shifted > max_threshold))[0]
        neg = np.nonzero((vsig > max_threshold) & (shifted < min_threshold))[0]
    return pos, neg


def _unwrap_voltage_signal(vsig, pos_wraps, neg_wraps,
                           max_threshold=5.1, max_diff=1.0):
    vsig = np.asarray(vsig)
    vmax = vsig[vsig < max_threshold].max()
    diff = np.zeros(vsig.shape)
    vsig_prev = _shift(vsig)
    if len(pos_wraps):
        diff[pos_wraps] = (vsig[pos_wraps] + vmax) - vsig_prev[pos_wraps]
    if len(neg_wraps):
        diff[neg_wraps] = vsig[neg_wraps] - (vsig_prev[neg_wraps] + vmax)
    wrap_ix = np.concatenate((pos_wraps, neg_wraps))
    other_ix = np.array(sorted(set(range(len(vsig))) - set(wrap_ix.tolist())))
    diff[other_ix] = vsig[other_ix] - vsig_prev[other_ix]
    with np.errstate(invalid="ignore"):
        diff = np.where(np.abs(diff) <= max_diff, diff, np.nan)
    nan_ix = np.isnan(diff)
    summed = np.nancumsum(diff) + vsig[0]
    summed[nan_ix] = np.nan
    return summed


def _local_boundaries(time, index, span=0.25):
    t_val = time[index]
    eligible = np.nonzero(
        (time <= t_val + abs(span)) & (time >= t_val - abs(span)))[0]
    return eligible.min(), eligible.max()


def _clip_speed_wraps(speed, time, wrap_indices, t_span=0.25):
    out = speed.copy()
    for w in wrap_indices:
        lo, hi = _local_boundaries(time, w, t_span)
        local = np.concatenate((speed[lo:w], speed[w + 1:hi + 1]))
        out[w] = np.clip(speed[w], np.nanmin(local), np.nanmax(local))
    return out


def _zscore_threshold_1d(data, threshold=10.0):
    out = data.copy().astype(float)
    scores = _zscore(data, nan_policy="omit")
    with np.errstate(invalid="ignore"):
        out[np.abs(scores) > threshold] = np.nan
    return out


def _get_encoder(pkl: dict) -> dict:
    items = pkl.get("items") or {}
    for key in ("foraging", "behavior"):
        encoders = ((items.get(key) or {}).get("encoders") or [])
        if encoders:
            return encoders[0]
    raise KeyError("No encoder found under items.foraging.encoders or items.behavior.encoders")


def compute_running_speed(pkl: dict, time: np.ndarray,
                          lowpass: bool = True,
                          zscore_threshold: float = 10.0) -> pd.DataFrame:
    """Compute linear running speed (cm/s) from encoder + sync timebase."""
    enc = _get_encoder(pkl)
    v_sig = np.asarray(enc["vsig"])
    v_in = np.asarray(enc["vin"])
    dx_raw = np.asarray(enc["dx"])

    if len(v_in) == len(time) + 1:
        v_in = v_in[:-1]
        v_sig = v_sig[:-1]
    elif len(v_in) > len(time):
        v_in = v_in[:len(time)]
        v_sig = v_sig[:len(time)]
    elif len(v_in) < len(time):
        time = time[:len(v_in)]

    pos, neg = _identify_wraps(v_sig)
    unwrapped = _unwrap_voltage_signal(v_sig, pos, neg)
    delta_theta = np.diff(unwrapped, prepend=np.nan) / v_in * 2 * np.pi
    theta = np.nancumsum(delta_theta)
    theta[np.isnan(delta_theta)] = np.nan

    dt = np.diff(time, prepend=np.nan)
    angular_speed = np.diff(theta, prepend=np.nan) / dt
    linear_speed = angular_speed * _WHEEL_RUNNING_RADIUS_CM

    linear_speed = _clip_speed_wraps(
        linear_speed, time, np.concatenate([pos, neg]), t_span=0.25)
    linear_speed = _zscore_threshold_1d(linear_speed, threshold=zscore_threshold)

    if lowpass:
        b, a = _signal.butter(3, Wn=4, fs=60, btype="lowpass")
        linear_speed = _signal.filtfilt(b, a, np.nan_to_num(linear_speed))

    n = len(time)
    return pd.DataFrame(
        {
            "speed": linear_speed[:n],
            "dx": dx_raw[:n],
            "v_sig": v_sig[:n],
            "v_in": v_in[:n],
        },
        index=pd.Index(time, name="timestamps"),
    )


def add_running_speed(nwb, pkl: dict, stim_vsync_fall: np.ndarray) -> None:
    """Add processed running speed and raw encoder traces to NWB."""
    df = compute_running_speed(pkl, stim_vsync_fall)
    timestamps = df.index.values

    speed_ts = TimeSeries(
        name="speed", data=df["speed"].values, timestamps=timestamps,
        unit="cm/s",
        description="Running speed from wheel encoder (SweepStim path).",
    )
    dx_ts = TimeSeries(
        name="dx", data=df["dx"].values, timestamps=timestamps, unit="cm",
        description="Running-wheel angular change (raw pkl encoder dx).",
    )
    v_sig_ts = TimeSeries(
        name="v_sig", data=df["v_sig"].values, timestamps=timestamps, unit="V",
        description="Raw voltage signal from the running-wheel encoder.",
    )
    v_in_ts = TimeSeries(
        name="v_in", data=df["v_in"].values, timestamps=timestamps, unit="V",
        description="Theoretical max encoder voltage for normalisation.",
    )

    if "running" in nwb.processing:
        mod = nwb.processing["running"]
    else:
        mod = ProcessingModule(
            name="running",
            description="Running speed processing module",
        )
        nwb.add_processing_module(mod)
    mod.add_data_interface(speed_ts)
    mod.add_data_interface(dx_ts)
    nwb.add_acquisition(v_sig_ts)
    nwb.add_acquisition(v_in_ts)
