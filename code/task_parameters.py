"""Extract change-detection task parameters from a stim pkl.

Two entry points:
- `extract_task_parameters(pkl)` → dict of cleaned field values
- `build_task_parameters(pkl, name='task_parameters')` →
  ChangeDetectionTaskParameters container, ready for `nwbfile.add_lab_meta_data`
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path

from ndx_change_detection_task import ChangeDetectionTaskParameters


# HED descriptor for the task at the protocol level.
# Uses only HED v8.3.0 base-schema tags.
_TASK_HED = 'Task, Experiment-procedure, Label/change_detection_task'


def _image_set_name(image_set_path: str | None) -> str | None:
    """Strip directory and .pkl extension from camstim image_set path."""
    if not image_set_path:
        return None
    base = os.path.basename(image_set_path)
    if base.endswith('.pkl'):
        base = base[:-4]
    return base


def extract_task_parameters(pkl) -> dict:
    """Pull task parameters from a loaded stim pkl into a flat dict.

    Parameters
    ----------
    pkl : dict or str/Path
        Already-loaded pkl dict, or a path to one.

    Returns
    -------
    dict
        Field names matching `ChangeDetectionTaskParameters` schema.
        Missing/None values are omitted (the schema allows it).
    """
    if isinstance(pkl, (str, Path)):
        with open(pkl, 'rb') as f:
            pkl = pickle.load(f, encoding='latin1')

    beh = pkl['items']['behavior']
    params = beh.get('params', {})

    out: dict = {}

    # Trial logic
    _set(out, 'change_flashes_min', params.get('change_flashes_min'), int)
    _set(out, 'change_flashes_max', params.get('change_flashes_max'), int)
    _set(out, 'change_time_distribution', params.get('change_time_dist'), str)
    _set(out, 'change_time_scale', params.get('change_time_scale'), float)
    _set(out, 'pre_change_time_sec', params.get('pre_change_time'), float)
    _set(out, 'min_no_lick_time_sec', params.get('min_no_lick_time'), float)
    _set(out, 'timeout_duration_sec', params.get('timeout_duration'), float)
    _set(out, 'end_after_response_sec', params.get('end_after_response_sec'), float)
    _set(out, 'failure_repeats', params.get('failure_repeats'), int)
    _set(out, 'catch_frequency', params.get('catch_frequency'), float)
    out['catch_mode'] = (
        'explicit' if params.get('catch_frequency') is not None else 'implicit'
    )

    # Response window: stored as [start, stop] array
    rw = params.get('response_window')
    if rw is not None and len(rw) == 2:
        out['response_window_sec'] = [float(rw[0]), float(rw[1])]

    # Stimulus
    stim = params.get('stimulus', {})
    _set(out, 'stimulus_class', stim.get('class'), str)
    _set(out, 'image_set_name',
         _image_set_name(stim.get('params', {}).get('image_set')), str)

    # periodic_flash is [flash_duration, blank_duration]
    pf = params.get('periodic_flash')
    if pf is not None and len(pf) == 2:
        out['stimulus_duration_sec'] = float(pf[0])
        out['blank_duration_sec'] = float(pf[1])

    _set(out, 'flash_omit_probability', params.get('flash_omit_probability'), float)

    # Reward
    _set(out, 'reward_volume_ml', params.get('reward_volume'), float)
    _set(out, 'auto_reward_volume_ml', params.get('auto_reward_vol'), float)
    _set(out, 'auto_reward_delay_sec', params.get('auto_reward_delay'), float)
    _set(out, 'volume_limit_ml', params.get('volume_limit'), float)
    _set(out, 'free_reward_trials', params.get('free_reward_trials'), int)
    _set(out, 'warm_up_trials', params.get('warm_up_trials'), int)

    # Session
    _set(out, 'session_type', params.get('stage'), str)
    _set(out, 'task_id', params.get('task_id'), str)
    out['task_name'] = 'change detection'  # canonical, not in pkl
    _set(out, 'max_task_duration_min', params.get('max_task_duration_min'), float)

    # Epilogue (fingerprint movie)
    ep = params.get('epilogue')
    if isinstance(ep, dict):
        _set(out, 'epilogue_name', ep.get('name'), str)
        ep_params = ep.get('params', {})
        _set(out, 'epilogue_movie_path', ep_params.get('movie_path'), str)
        _set(out, 'epilogue_runs', ep_params.get('runs'), int)
        _set(out, 'epilogue_frame_length_sec', ep_params.get('frame_length'), float)

    out['task_hed_tags'] = _TASK_HED
    return out


def build_task_parameters(pkl, name: str = 'task_parameters') -> ChangeDetectionTaskParameters:
    """Build a ChangeDetectionTaskParameters container from a pkl."""
    fields = extract_task_parameters(pkl)
    return ChangeDetectionTaskParameters(name=name, **fields)


def _set(d: dict, key: str, value, caster) -> None:
    """Cast and store value if it's not None."""
    if value is None:
        return
    d[key] = caster(value)
