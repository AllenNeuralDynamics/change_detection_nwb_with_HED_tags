"""Local NWB extension: ndx-change-detection-task.

Defines a `ChangeDetectionTaskParameters` neurodata type extending
`LabMetaData`. The full schema is declared in-module; YAML spec files
are written to `_specs/` on first import and reused thereafter.

Usage
-----
    from ndx_change_detection_task import ChangeDetectionTaskParameters
    tp = ChangeDetectionTaskParameters(
        name='task_parameters',
        task_name='change detection',
        response_window_sec=[0.15, 0.75],
        reward_volume_ml=0.007,
        ...
    )
    nwbfile.add_lab_meta_data(tp)

Regenerating the spec
---------------------
Edit `SCHEMA` below and delete `_specs/` to force regeneration on next
import, or call `_write_spec()` directly.
"""
from __future__ import annotations
from pathlib import Path

from pynwb import load_namespaces, get_class
from pynwb.spec import (
    NWBNamespaceBuilder,
    NWBGroupSpec,
    NWBAttributeSpec,
    NWBDatasetSpec,
)

NAMESPACE = 'ndx-change-detection-task'
NEURODATA_TYPE = 'ChangeDetectionTaskParameters'
VERSION = '0.1.0'

_HERE = Path(__file__).parent
_SPEC_DIR = _HERE / '_specs'
_NS_PATH = _SPEC_DIR / f'{NAMESPACE}.namespace.yaml'
_EXT_PATH = _SPEC_DIR / f'{NAMESPACE}.extensions.yaml'


# ── Schema declaration ────────────────────────────────────────────────
# Scalar fields → NWBAttributeSpec. Array fields → NWBDatasetSpec.
# All non-required so individual sessions can omit fields they don't use.

_SCALAR_FIELDS = [
    # Trial logic
    ('change_flashes_min', 'int',
     'Minimum number of stimulus flashes before a change can occur.'),
    ('change_flashes_max', 'int',
     'Maximum number of stimulus flashes before a change can occur '
     '(if no change has occurred, the trial closes).'),
    ('change_time_distribution', 'text',
     'Distribution used to draw change times. Typical values: '
     '"geometric", "exponential".'),
    ('change_time_scale', 'float',
     'Scale parameter of the change-time distribution (seconds).'),
    ('pre_change_time_sec', 'float',
     'Required no-change time at the start of each trial (seconds).'),
    ('min_no_lick_time_sec', 'float',
     'Required no-lick period before a trial advances (seconds).'),
    ('timeout_duration_sec', 'float',
     'Timeout penalty after a false alarm or abort (seconds).'),
    ('end_after_response_sec', 'float',
     'Trial duration after a response is detected (seconds).'),
    ('failure_repeats', 'int',
     'Number of times a failed trial is repeated.'),
    ('catch_frequency', 'float',
     'Fraction of trials that are catch trials (no change). Only set '
     'when catch_mode is "explicit"; omitted when catch_mode is "implicit".'),
    ('catch_mode', 'text',
     'How catch trials are produced. "explicit": each trial has a fixed '
     'catch_frequency probability of being a catch trial. "implicit": '
     'sham changes are drawn from the change-time distribution and '
     'occur on trials that end before the drawn change time.'),

    # Stimulus
    ('stimulus_class', 'text',
     'Class of stimulus presented. Typical values: "images", "gratings".'),
    ('image_set_name', 'text',
     'Name of the image set used (file name without path).'),
    ('stimulus_duration_sec', 'float',
     'Duration of each stimulus flash (seconds).'),
    ('blank_duration_sec', 'float',
     'Duration of the gray-screen gap between flashes (seconds).'),
    ('flash_omit_probability', 'float',
     'Fraction of stimulus flashes that are intentionally omitted.'),

    # Reward
    ('reward_volume_ml', 'float',
     'Volume of each earned water reward (mL).'),
    ('auto_reward_volume_ml', 'float',
     'Volume of each auto-delivered water reward (mL).'),
    ('auto_reward_delay_sec', 'float',
     'Delay between change and auto-reward delivery (seconds).'),
    ('volume_limit_ml', 'float',
     'Maximum total reward volume per session (mL).'),
    ('free_reward_trials', 'int',
     'Number of warm-up trials at the start that are auto-rewarded.'),
    ('warm_up_trials', 'int',
     'Number of warm-up trials with reduced difficulty at session start.'),

    # Session
    ('session_type', 'text',
     'Session-type identifier (e.g. "OPHYS_3_images_A").'),
    ('task_id', 'text',
     'Task-protocol identifier (e.g. "DoC" for detection-of-change).'),
    ('task_name', 'text',
     'Human-readable task name (e.g. "change detection").'),
    ('max_task_duration_min', 'float',
     'Maximum session duration in minutes before forced stop.'),

    # Epilogue (fingerprint movie shown after task)
    ('epilogue_name', 'text',
     'Name of the post-task stimulus epilogue (e.g. "fingerprint").'),
    ('epilogue_movie_path', 'text',
     'Path to the movie file used in the epilogue.'),
    ('epilogue_runs', 'int',
     'Number of repetitions of the epilogue movie.'),
    ('epilogue_frame_length_sec', 'float',
     'Frame length of the epilogue movie (seconds per frame).'),

    # HED descriptor for the task itself
    ('task_hed_tags', 'text',
     'HED tag string describing the task at the protocol level '
     '(e.g. "Task, Visual-detection-task").'),
]

_ARRAY_FIELDS = [
    ('response_window_sec', 'float', (2,),
     '[start, stop] of the response window relative to change (seconds).'),
]


def _write_spec() -> None:
    """Write namespace + extensions YAML to `_specs/`."""
    _SPEC_DIR.mkdir(exist_ok=True)

    ns_builder = NWBNamespaceBuilder(
        doc='NWB extension for change detection task parameters.',
        name=NAMESPACE,
        version=VERSION,
        author=['Change-detection NWB+HED project'],
        contact=['n/a'],
    )
    ns_builder.include_type('LabMetaData', namespace='core')

    attrs = [
        NWBAttributeSpec(name=n, dtype=dt, doc=doc, required=False)
        for n, dt, doc in _SCALAR_FIELDS
    ]
    datasets = [
        NWBDatasetSpec(name=n, dtype=dt, shape=sh, doc=doc, quantity='?')
        for n, dt, sh, doc in _ARRAY_FIELDS
    ]

    cdtp = NWBGroupSpec(
        neurodata_type_def=NEURODATA_TYPE,
        neurodata_type_inc='LabMetaData',
        doc=('Task parameters for the visual change-detection task '
             '(images / gratings, response-window detection paradigm).'),
        attributes=attrs,
        datasets=datasets,
    )

    # add_spec wants the bare filename; export wants the full path
    ns_builder.add_spec(_EXT_PATH.name, cdtp)
    ns_builder.export(_NS_PATH.name, outdir=str(_SPEC_DIR))


# Build + load on import. Always regenerate so the YAML can't drift from
# the in-code SCHEMA. Cheap (a few ms) and avoids stale-spec footguns.
_write_spec()
load_namespaces(str(_NS_PATH))

ChangeDetectionTaskParameters = get_class(NEURODATA_TYPE, NAMESPACE)

__all__ = ['ChangeDetectionTaskParameters', 'NAMESPACE', 'VERSION']
