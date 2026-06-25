"""Package change-detection behavior data into a HED-annotated NWB file.

Entry point: `package_to_nwb(pkl_path, sync_path, output_path, metadata=None)`.

Layout produced
---------------
NWBFile
├── lab_meta_data
│   ├── task_parameters (ChangeDetectionTaskParameters)
│   └── hed_schema (HedLabMetaData)
├── subject (subject_id = mouse_id from pkl)
├── trials (compositional view: timing + go/catch/hit/miss/etc.; HedTags)
├── intervals
│   ├── intervals (canonical flat table — every interval timing + type + HED)
│   ├── stimulus_presentations (compositional view: timing + image_name,
│   │   is_change, omitted; HedTags)
│   └── natural_movie_one_presentations (compositional view: timing +
│       frame_index, repeat; HedTags)
└── events
    └── events (ndx-events EventsTable — *discrete* events only:
        lick, reward, miss, image_change, image_omission; HedTags)

Notes
-----
The flat ``intervals`` table is the canonical source of timing for every
session interval (epochs, trials, change/response windows, stimulus
presentations, movie frames). The ``trials``, ``stimulus_presentations``,
and ``natural_movie_one_presentations`` tables are compositional —
they duplicate timing for ergonomics and add the task-specific annotation
columns. ``nwb.epochs`` is intentionally not set; epochs live as
``interval_type='epoch'`` rows on the flat intervals table.
"""
from __future__ import annotations
import argparse
import datetime
import json
import logging
import pickle
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy.signal as _signal
from scipy.stats import zscore as _zscore

from pynwb import NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from hdmf.common import VectorData

from ndx_events import (
    EventsTable,
    MeaningsTable,
    CategoricalVectorData,
    NdxEventsNWBFile,
)
from ndx_hed import HedLabMetaData, HedTags

from build_events_and_intervals import build_all
from task_parameters import build_task_parameters

logger = logging.getLogger(__name__)

HED_SCHEMA_VERSION = '8.3.0'


# ── DataFrame helper for readers ──────────────────────────────────────
# NWB's schema mandates the row-id dataset live at `id` in every
# DynamicTable, so we can't rename it on disk. But once a table is read
# back, this helper gives the dataframe a typed index name like
# 'trials_id' / 'stimulus_presentations_id', which is much clearer when
# joining tables across the file.
def to_df(table, index_name: str | None = None) -> pd.DataFrame:
    """Convert an NWB DynamicTable to a DataFrame with a typed index name.

    Parameters
    ----------
    table : DynamicTable
        Any NWB DynamicTable (trials, epochs, TimeIntervals, EventsTable...).
    index_name : str, optional
        Name to give the dataframe index. Defaults to ``f'{table.name}_id'``.
    """
    df = table.to_dataframe()
    df.index.name = index_name or f'{table.name}_id'
    return df


def nwb_to_dfs(nwb) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct (events_df, intervals_df) in the original CSV format
    from a packaged NWB.

    The NWB events table holds only discrete events; image_onset/offset
    and movie_onset/offset rows are synthesized from the
    stimulus_presentations / natural_movie_one_presentations tables.

    The NWB flat intervals table holds only timing; trial annotations
    (go/catch/hit/miss/etc.) are merged in from ``nwb.trials``, and the
    new ``stimulus_presentation`` / ``movie_frame`` rows are dropped to
    match the original CSV intervals schema.
    """
    # ── events_df: discrete events + synthesized image/movie onsets/offsets ──
    et_df = nwb.get_events_table('events').to_dataframe()
    discrete = pd.DataFrame({
        'timestamp': et_df['timestamp'].astype(float),
        'event_type': et_df['event_type'].values,
        'trials_id': et_df['trials_id'].astype(int).values,
        'stimulus_presentations_id': et_df['stimulus_presentations_id'].astype(int).values,
        'image_name': et_df['image_name'].replace('', np.nan).values,
        'frame': et_df['frame'].replace(-1, np.nan).values,
        'lick_latency': np.nan,  # recomputed below for lick rows
        'reward_volume': et_df['reward_volume'].values,
        'movie_frame_index': np.nan,
        'movie_repeat': np.nan,
        'lick_classification': et_df['lick_classification'].replace('n/a', np.nan).values,
        'bout_start': (et_df['lick_bouts'] == 'bout_start').values,
        'reward_type': et_df['reward_type'].replace('n/a', np.nan).values,
    })

    sp_df = nwb.intervals['stimulus_presentations'].to_dataframe()
    flashes = sp_df[~sp_df['omitted'].astype(bool)]
    image_onsets = pd.DataFrame({
        'timestamp': flashes['start_time'].astype(float).values,
        'event_type': 'image_onset',
        'trials_id': flashes['trials_id'].astype(int).values,
        'stimulus_presentations_id': flashes['stimulus_presentations_id'].astype(int).values,
        'image_name': flashes['image_name'].values,
        'frame': flashes['start_frame'].astype(float).values,
        'lick_latency': np.nan, 'reward_volume': np.nan,
        'movie_frame_index': np.nan, 'movie_repeat': np.nan,
        'lick_classification': np.nan, 'bout_start': False, 'reward_type': np.nan,
    })
    image_offsets = pd.DataFrame({
        'timestamp': flashes['stop_time'].astype(float).values,
        'event_type': 'image_offset',
        'trials_id': flashes['trials_id'].astype(int).values,
        'stimulus_presentations_id': flashes['stimulus_presentations_id'].astype(int).values,
        'image_name': flashes['image_name'].values,
        'frame': flashes['stop_frame'].astype(float).values,
        'lick_latency': np.nan, 'reward_volume': np.nan,
        'movie_frame_index': np.nan, 'movie_repeat': np.nan,
        'lick_classification': np.nan, 'bout_start': False, 'reward_type': np.nan,
    })

    parts = [discrete, image_onsets, image_offsets]
    if 'natural_movie_one_presentations' in nwb.intervals:
        nm_df = nwb.intervals['natural_movie_one_presentations'].to_dataframe()
        movie_onsets = pd.DataFrame({
            'timestamp': nm_df['start_time'].astype(float).values,
            'event_type': 'movie_onset',
            'trials_id': -1, 'stimulus_presentations_id': -1,
            'image_name': 'natural_movie_one',
            'frame': nm_df['start_frame'].astype(float).values,
            'lick_latency': np.nan, 'reward_volume': np.nan,
            'movie_frame_index': nm_df['movie_frame_index'].astype(float).values,
            'movie_repeat': nm_df['movie_repeat'].astype(float).values,
            'lick_classification': np.nan, 'bout_start': False, 'reward_type': np.nan,
        })
        movie_offsets = pd.DataFrame({
            'timestamp': nm_df['stop_time'].astype(float).values,
            'event_type': 'movie_offset',
            'trials_id': -1, 'stimulus_presentations_id': -1,
            'image_name': np.nan,
            'frame': nm_df['stop_frame'].astype(float).values,
            'lick_latency': np.nan, 'reward_volume': np.nan,
            'movie_frame_index': nm_df['movie_frame_index'].astype(float).values,
            'movie_repeat': nm_df['movie_repeat'].astype(float).values,
            'lick_classification': np.nan, 'bout_start': False, 'reward_type': np.nan,
        })
        parts.extend([movie_onsets, movie_offsets])

    events_df = pd.concat(parts, ignore_index=True)
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)

    # Recompute lick_latency for lick events to match the original CSV format
    # ("time since most recent image_onset"). The events table no longer stores
    # this; we derive it from stimulus_presentations start times.
    onset_times = (sp_df.loc[~sp_df['omitted'].astype(bool), 'start_time']
                   .astype(float).sort_values().values)
    lick_mask = events_df['event_type'] == 'lick'
    if len(onset_times) and lick_mask.any():
        lt = events_df.loc[lick_mask, 'timestamp'].values
        idx = np.searchsorted(onset_times, lt, side='right') - 1
        latencies = np.where(idx >= 0, lt - onset_times[idx.clip(min=0)], np.nan)
        events_df.loc[lick_mask, 'lick_latency'] = latencies

    # ── intervals_df: flat intervals (epochs/trials/windows only) +
    #    trial annotations merged in from nwb.trials ──
    iv_df = nwb.intervals['intervals'].to_dataframe()
    csv_types = {'epoch', 'trial', 'change_window', 'response_window'}
    iv_df = iv_df[iv_df['interval_type'].isin(csv_types)].copy()
    iv_df = iv_df.drop(columns=['stimulus_presentations_id',
                                 'natural_movie_one_presentations_id'],
                        errors='ignore')

    trials_df = nwb.trials.to_dataframe()
    annot_cols = ['go', 'catch', 'auto_rewarded', 'aborted', 'hit', 'miss',
                  'false_alarm', 'correct_reject', 'change_time', 'change_frame',
                  'initial_image_name', 'change_image_name', 'reward_time',
                  'reward_volume', 'response_time', 'response_latency']
    trials_annot = (trials_df[annot_cols].reset_index()
                    .rename(columns={'id': 'trials_id'}))
    iv_df = iv_df.merge(trials_annot, on='trials_id', how='left')
    not_trial = iv_df['interval_type'] != 'trial'
    iv_df.loc[not_trial, annot_cols] = np.nan
    # Replace sentinel -1 in change_frame with NaN for CSV compatibility
    iv_df['change_frame'] = iv_df['change_frame'].replace(-1, np.nan)
    # Empty strings → NaN for image-name columns
    for c in ('initial_image_name', 'change_image_name'):
        iv_df[c] = iv_df[c].replace('', np.nan)

    intervals_df = iv_df.rename(columns={'HED': 'hed_string'}).copy()
    intervals_df = intervals_df.sort_values('start_time').reset_index(drop=True)

    return events_df, intervals_df

# ── HED fragments ──────────────────────────────────────────────────────
# All tags below are validated against the HED v8.3.0 base schema.
# Orthogonal design: base event_type HED + independent classification/
# reward_type fragments that compose into the full tag string.

# Events table holds discrete (point) events only. Visual onsets/offsets
# (image_onset/image_offset/movie_onset/movie_offset) live in the intervals
# table as interval_type='stimulus_presentation' / 'movie_frame'. Miss is a
# trial-level outcome (annotated on nwb.trials), not a discrete event.
_EVENT_TYPE_HED = {
    'lick':
        'Agent-action, (Animal-agent, Move-face), Participant-response, Label/lick',
    'reward':
        'Sensory-event, Gustatory-presentation, (Ingestible-object, Reward), Label/water',
    'image_change':
        'Sensory-event, Visual-presentation, Target, Label/image_change',
    'image_omission':
        'Sensory-event, Unexpected, Label/omitted_flash',
}

# Event types from build_events_and_intervals.py that don't belong on the
# events table: intervals belong on the intervals table; 'miss' is a trial
# outcome (already on nwb.trials).
_DROP_EVENT_TYPES = {
    'image_onset', 'image_offset', 'movie_onset', 'movie_offset', 'miss',
}

_LICK_CLASSIFICATION_HED = {
    'hit': 'Correct-action, Label/hit',
    'false_alarm': 'Incorrect-action, Label/false_alarm',
    'abort': 'Incorrect-action, Label/abort',
    'early': 'Incorrect-action, Label/early',
    'late': 'Incorrect-action, Label/late',
    'consumption': 'Label/consumption',
    'spontaneous': 'Label/spontaneous',
    'n/a': '',
}

_REWARD_TYPE_HED = {
    'earned': 'Label/earned',
    'auto_reward': 'Label/auto_reward',
    'n/a': '',
}

_LICK_BOUTS_HED = {
    'bout_start': '(Temporal-marker, Label/bout_start)',
    'within_bout': '',
    'n/a': '',
}

# Per-trial HED — composed from outcome flags.
_TRIAL_OUTCOME_HED = {
    'hit':
        'Experimental-trial, Target, Correct-action, Label/hit_trial',
    'miss':
        'Experimental-trial, Target, Miss, Label/miss_trial',
    'false_alarm':
        'Experimental-trial, Non-target, Incorrect-action, Label/false_alarm_trial',
    'correct_reject':
        'Experimental-trial, Non-target, Correct-action, Label/correct_reject_trial',
    'aborted':
        'Experimental-trial, Incorrect-action, Label/aborted_trial',
    'auto_rewarded':
        'Experimental-trial, Reward, Label/auto_rewarded_trial',
    'no_outcome':
        'Experimental-trial',
}

_EPOCH_HED = {
    'change_detection':
        'Time-block, Experiment-procedure, Label/change_detection_task',
    'natural_movie_one':
        'Time-block, Sensory-event, Visual-presentation, '
        '(Movie, Label/natural_movie_one), Label/fingerprint_epoch',
    'spontaneous':
        'Time-block, Pause, Label/spontaneous',
}


# ── Semantic descriptions ──────────────────────────────────────────────
# Plain-English meanings for each categorical value. These get inserted
# into the MeaningsTables as a `description` column and exported as the
# `Levels` block of the sidecar JSON. Keys must match the *_HED dicts above.
_EVENT_TYPE_DESC = {
    'lick':
        'Time of a lick contact on the lick spout, detected by the lick '
        'sensor.',
    'reward':
        'Time of a water-reward delivery to the lick spout.',
    'image_change':
        'Time of a stimulus presentation where the identity of the presented '
        'image is distinct from the previously presented image; demarcates '
        'Go trials where the mouse can earn rewards for licks within the '
        'post-change reward window.',
    'image_omission':
        'Time of a scheduled image flash that was withheld (no image shown) '
        'in slots where a flash would otherwise have occurred.',
}

_LICK_CLASSIFICATION_DESC = {
    'hit':
        'First lick within the response window after an image change on '
        'a go trial.',
    'false_alarm':
        'Incorrect lick occurring during the window of time where the image '
        'could have changed (based on the change trial distribution) but '
        'did not change; triggers reset of trial.',
    'abort':
        'Incorrect lick occurring in the 4 flash period after the start of '
        'a trial, prior to the change window onset; triggers reset of trial.',
    'early':
        'Unrewarded lick in the 150 ms window after the image change onset '
        'but prior to the reward window onset; considered as too early to '
        'be a valid response to the image change.',
    'late':
        'Lick after the response window has closed on a trial with a change '
        'but no reward (i.e., a missed change).',
    'consumption':
        'Lick after a reward delivery on the same trial when the mouse is '
        'consuming the water reward.',
    'spontaneous':
        'Lick outside of any task-defined window (e.g., between trials, '
        'during warm-up, during inter-trial intervals).',
    'n/a':
        'Not a lick event; this column does not apply.',
}

_REWARD_TYPE_DESC = {
    'earned':
        'Reward delivered as a consequence of a correct lick (hit) on a go '
        'trial.',
    'auto_reward':
        'Reward delivered automatically by the task (e.g., during warm-up '
        'or auto-rewarded trials) regardless of the animal\'s response.',
    'n/a':
        'Not a reward event; this column does not apply.',
}

_LICK_BOUTS_DESC = {
    'bout_start':
        'First lick of a bout: the inter-lick interval from the previous '
        'lick exceeds 500 ms (or this is the first lick of the session).',
    'within_bout':
        'Lick that occurs within an ongoing bout (inter-lick interval from '
        'previous lick ≤ 500 ms).',
    'n/a':
        'Not a lick event; this column does not apply.',
}

_TRIAL_OUTCOME_DESC = {
    'hit':
        'Go trial in which the animal licked within the response window '
        'after the image change — a correct detection.',
    'miss':
        'Go trial in which the animal failed to lick within the response '
        'window after the image change.',
    'false_alarm':
        'Catch trial in which the animal licked within the response window '
        'despite no image change occurring.',
    'correct_reject':
        'Catch trial in which the animal correctly withheld licking during '
        'the response window.',
    'aborted':
        'Trial in which the animal licked before the change window opened, '
        'aborting the trial.',
    'auto_rewarded':
        'Trial on which the task automatically delivered a reward regardless '
        'of the animal\'s response (e.g., warm-up trials, reminder trials).',
    'no_outcome':
        'Trial that did not produce any of the standard outcome categories '
        '(rare; typically a configuration/edge case).',
}

_EPOCH_DESC = {
    'change_detection':
        'Active change-detection task period during which mice can earn '
        'water rewards for licking after changes in image identity. Includes '
        'warm-up trials at the start of session, contingent reward trials '
        'throughout, and auto-reward reminders during disengaged periods.',
    'natural_movie_one':
        'Passive viewing of the "natural_movie_one" fingerprint movie clip, '
        'shown after the active task to identify recorded cells across '
        'sessions.',
    'spontaneous':
        'Any time outside of the named task epochs — gray-screen periods '
        'before, between, or after the change_detection and '
        'natural_movie_one epochs.',
}

# Interval types in the canonical flat intervals table.
_INTERVAL_TYPE_DESC = {
    'epoch':
        'A session-level epoch (e.g., change_detection, natural_movie_one, '
        'spontaneous). The epoch name is carried in the `label` column.',
    'trial':
        'A single behavioral trial in the change-detection task, bounded by '
        'trial_start and trial_end, defined by the change time distribution '
        'parameter (typically geometric between 4-12 flashes after trial '
        'start). Trials contain multiple stimulus presentations. Trial start '
        'time is un-cued. Annotations live on `nwb.trials`.',
    'change_window':
        'Window of time after trial start during which an image change may '
        'occur (as defined by the change time distribution), starting at '
        'the change_flashes_min-th flash and ending at the change (or, for '
        'catch/aborted trials, at the change_flashes_max-th flash or trial '
        'end).',
    'response_window':
        'Window after the image change during which a lick counts as a hit '
        '(go trial) or false alarm (catch trial). Defined by the '
        'response_window task parameter relative to change_time.',
    'stimulus_presentation':
        'A single image flash presentation (or omitted slot) during the '
        'active task. The image name is carried in the `label` column.',
    'movie_frame':
        'A single frame presentation of the natural_movie_one fingerprint '
        'movie.',
}

# HED-tag templates for descriptive value columns (continuous / id columns).
# These mirror the VRF sidecar's per-column HED templates with # placeholders.
_VALUE_COLUMN_HED = {
    'timestamp': 'Time-value/# s',
    'start_time': 'Time-value/# s',
    'stop_time': 'Time-value/# s',
    'reward_volume': 'Volume/# mL',
    'frame': 'Label/frame-#',
    'trials_id': 'Label/trial-#',
    'stimulus_presentations_id': 'Label/stimulus_presentation-#',
    'natural_movie_one_presentations_id': 'Label/movie_frame-#',
    'movie_frame_index': 'Label/movie_frame_index-#',
    'movie_repeat': 'Label/movie_repeat-#',
    'change_frame': 'Label/change_frame-#',
    'change_time': 'Time-value/# s',
    'reward_time': 'Time-value/# s',
    'response_time': 'Time-value/# s',
    'response_latency': 'Time-value/# s',
    'lick_latency': 'Time-value/# s',
    'change_window_start_time': 'Time-value/# s',
    'change_window_stop_time': 'Time-value/# s',
    'response_window_start_time': 'Time-value/# s',
    'response_window_stop_time': 'Time-value/# s',
    'start_frame': 'Label/frame-#',
    'stop_frame': 'Label/frame-#',
}

# Descriptions for descriptive value / id / metadata columns that don't
# have a Levels-style enumeration.
_VALUE_COLUMN_DESC = {
    'timestamp':
        'Event time, in seconds from session start, aligned to the sync-file '
        'hardware clock. Visual events include the measured monitor delay; '
        'behavioral events (licks, rewards) do not.',
    'start_time':
        'Interval start time, in seconds from session start.',
    'stop_time':
        'Interval stop time, in seconds from session start.',
    'reward_volume':
        'Volume of water delivered for this reward, in millilitres.',
    'frame':
        'Vsync falling-edge frame index (into the sync-file frame array) '
        'corresponding to this event\'s timestamp. -1 if not applicable.',
    'start_frame':
        'Vsync falling-edge frame index for the interval start time.',
    'stop_frame':
        'Vsync falling-edge frame index for the interval stop time.',
    'trials_id':
        'Foreign key into the trials table — index of the trial this row '
        'belongs to. -1 if the row falls outside any trial.',
    'stimulus_presentations_id':
        'Foreign key into the stimulus_presentations table. -1 if not '
        'applicable.',
    'natural_movie_one_presentations_id':
        'Foreign key into the natural_movie_one_presentations table. -1 if '
        'not applicable.',
    'movie_frame_index':
        'Frame index within a single playback of natural_movie_one (0-based, '
        '0–899 for a 30-second clip at 30 Hz).',
    'movie_repeat':
        'Repetition number of natural_movie_one (0-based) for this frame.',
    'image_name':
        'Identifier of the image presented (e.g., "im065"), or '
        '"natural_movie_one" for movie frames, or "omitted" for withheld '
        'flashes.',
    'label':
        'Descriptive label for this row. Carries the epoch name for '
        'interval_type=epoch rows and the image_name for '
        'interval_type=stimulus_presentation rows; empty otherwise.',
    'change_time':
        'Time of the image change on this trial, in seconds. NaN if no '
        'change occurred (catch / aborted trials).',
    'change_frame':
        'Vsync falling-edge frame index of the image change. -1 if no '
        'change occurred.',
    'reward_time':
        'Time of reward delivery on this trial, in seconds. NaN if no '
        'reward was delivered.',
    'response_time':
        'Time of the first lick after the image change on this trial. NaN '
        'if the animal did not lick.',
    'response_latency':
        'Latency from change_time to response_time on this trial, in '
        'seconds. NaN if no response.',
    'lick_latency':
        'Time, in seconds, from the most recent image_onset to this lick '
        '(events table) or from this stimulus onset to the next lick before '
        'the following presentation (stimulus_presentations table).',
    'initial_image_name':
        'Image identity shown at the start of the trial (before any change).',
    'change_image_name':
        'Image identity shown after the change. Equal to initial_image_name '
        'for catch / aborted trials where no change occurred.',
    'change_window_start_time':
        'Start time of the trial\'s change window, in seconds. NaN if '
        'undefined.',
    'change_window_stop_time':
        'Stop time of the trial\'s change window, in seconds. NaN if '
        'undefined.',
    'response_window_start_time':
        'Start time of the trial\'s response window, in seconds. NaN if '
        'no change occurred.',
    'response_window_stop_time':
        'Stop time of the trial\'s response window, in seconds. NaN if '
        'no change occurred.',
    'epoch_name':
        'Canonical name of the session epoch containing this row. One of '
        'change_detection, natural_movie_one, spontaneous.',
    'is_change':
        'True if this image flash differs in identity from the previous '
        '(non-omitted) flash — i.e., this is the target stimulus on a go '
        'trial.',
    'omitted':
        'True if this row is a withheld (omitted) flash slot rather than an '
        'actually-displayed image.',
    'go':
        'True if this trial was a go trial (image change scheduled, not '
        'auto-rewarded, not aborted).',
    'catch':
        'True if this trial was a catch trial (no image change scheduled).',
    'auto_rewarded':
        'True if this trial automatically delivered a reward (e.g., warm-up '
        'trials).',
    'aborted':
        'True if this trial was aborted by a lick before the change window '
        'opened.',
    'hit':
        'True if the animal licked within the response window on a go trial.',
    'miss':
        'True if the animal failed to lick within the response window on a '
        'go trial.',
    'false_alarm':
        'True if the animal licked in the change or response window on a '
        'catch trial.',
    'correct_reject':
        'True if the animal correctly withheld licking on a catch trial.',
    'warm_up':
        'True for trials in the initial warm-up block (the first N '
        'auto-rewarded trials).',
    'interval_type':
        'Discriminator for the kind of interval this row represents '
        '(epoch, trial, change_window, response_window, '
        'stimulus_presentation, movie_frame).',
    'event_type':
        'Discriminator for the kind of point event this row represents '
        '(lick, reward, image_change, image_omission).',
    'lick_classification':
        'Task-context classification of a lick event. "n/a" for non-lick '
        'events.',
    'reward_type':
        'Type of reward (earned vs auto_reward). "n/a" for non-reward '
        'events.',
    'lick_bouts':
        'Bout-structure marker for lick events (bout_start vs within_bout). '
        '"n/a" for non-lick events.',
    'HED':
        'Hierarchical Event Descriptor (HED) tag string for this row, '
        'composed from the base event/interval tag and any orthogonal '
        'category fragments.',
}


# ── Helpers ────────────────────────────────────────────────────────────
def _stim_presentation_hed(image_name: str, is_change: bool, omitted: bool) -> str:
    if omitted:
        return 'Sensory-event, Unexpected, Label/omitted_flash'
    base = (f'Sensory-event, Visual-presentation, '
            f'(Image, Label/{image_name})')
    if is_change:
        base += ', Target'
    return base


def _trial_hed(row: pd.Series) -> str:
    if row.get('hit'):
        key = 'hit'
    elif row.get('miss'):
        key = 'miss'
    elif row.get('false_alarm'):
        key = 'false_alarm'
    elif row.get('correct_reject'):
        key = 'correct_reject'
    elif row.get('aborted'):
        key = 'aborted'
    elif row.get('auto_rewarded'):
        key = 'auto_rewarded'
    else:
        key = 'no_outcome'
    return _TRIAL_OUTCOME_HED[key]


# ── Running speed processing (ported from AllenSDK) ───────────────────
# Port of allensdk.brain_observatory.behavior.data_objects.running_speed.
# running_processing.get_running_df. Converts the rotary-encoder voltage
# signal stored in the pkl into linear running speed (cm/s), aligned to
# the sync-file vsync falling-edge timestamps (no monitor delay — running
# is a behavioral signal, not visual).
_WHEEL_DIAM_CM = 6.5 * 2.54           # 6.5" wheel
_WHEEL_RUNNING_RADIUS_CM = 0.5 * (2.0 * _WHEEL_DIAM_CM / 3.0)  # mouse at 2/3 R


def _shift(arr, periods=1, fill_value=np.nan):
    shifted = np.roll(arr, periods).astype(float)
    shifted[:periods] = fill_value
    return shifted


def _identify_wraps(vsig, min_threshold=1.5, max_threshold=3.5):
    """Find 0V↔5V wrap indices in the encoder voltage signal."""
    shifted = _shift(np.asarray(vsig))
    vsig = np.asarray(vsig)
    with np.errstate(invalid='ignore'):
        pos = np.nonzero((vsig < min_threshold) & (shifted > max_threshold))[0]
        neg = np.nonzero((vsig > max_threshold) & (shifted < min_threshold))[0]
    return pos, neg


def _unwrap_voltage_signal(vsig, pos_wraps, neg_wraps,
                           max_threshold=5.1, max_diff=1.0):
    """Cumulatively unwrap the encoder voltage across 0V↔5V wraps."""
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
    with np.errstate(invalid='ignore'):
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
    """Clip transient spikes at voltage wraps to the local min/max."""
    out = speed.copy()
    for w in wrap_indices:
        lo, hi = _local_boundaries(time, w, t_span)
        local = np.concatenate((speed[lo:w], speed[w + 1:hi + 1]))
        out[w] = np.clip(speed[w], np.nanmin(local), np.nanmax(local))
    return out


def _zscore_threshold_1d(data, threshold=10.0):
    out = data.copy().astype(float)
    scores = _zscore(data, nan_policy='omit')
    with np.errstate(invalid='ignore'):
        out[np.abs(scores) > threshold] = np.nan
    return out


def compute_running_speed(pkl: dict, time: np.ndarray,
                          lowpass: bool = True,
                          zscore_threshold: float = 10.0) -> pd.DataFrame:
    """Compute linear running speed (cm/s) from the pkl encoder + sync times.

    Mirrors AllenSDK's ``get_running_df``: identifies voltage wraps,
    recomputes angular change from unwrapped voltage (more reliable than
    the pkl ``dx``), converts to linear speed via wheel geometry, clips
    wrap artifacts, removes z-score outliers, and (optionally) low-pass
    filters with a 3rd-order Butterworth at 4 Hz (60 Hz fs).

    Parameters
    ----------
    pkl : loaded behavior pickle.
    time : 1d sync-file vsync falling-edge times (s), one per frame.
    lowpass : whether to apply the 4 Hz Butterworth filter.
    zscore_threshold : outlier rejection threshold in SDs.

    Returns
    -------
    DataFrame indexed by ``timestamps`` with columns ``speed``, ``dx``,
    ``v_sig``, ``v_in`` (length-matched to ``time``).
    """
    enc = pkl['items']['behavior']['encoders'][0]
    v_sig = np.asarray(enc['vsig'])
    v_in = np.asarray(enc['vin'])
    dx_raw = np.asarray(enc['dx'])

    # AllenSDK guard: encoder array can be 1 longer than the frame array.
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
    angular_speed = np.diff(theta, prepend=np.nan) / dt  # rad/s
    linear_speed = angular_speed * _WHEEL_RUNNING_RADIUS_CM  # cm/s

    linear_speed = _clip_speed_wraps(
        linear_speed, time, np.concatenate([pos, neg]), t_span=0.25)
    linear_speed = _zscore_threshold_1d(linear_speed, threshold=zscore_threshold)

    if lowpass:
        b, a = _signal.butter(3, Wn=4, fs=60, btype='lowpass')
        linear_speed = _signal.filtfilt(b, a, np.nan_to_num(linear_speed))

    n = len(time)
    return pd.DataFrame(
        {'speed': linear_speed[:n], 'dx': dx_raw[:n],
         'v_sig': v_sig[:n], 'v_in': v_in[:n]},
        index=pd.Index(time, name='timestamps'),
    )


def add_running_speed(nwb: NdxEventsNWBFile, pkl: dict,
                      stim_vsync_fall: np.ndarray) -> None:
    """Add processed running speed + raw encoder traces to the NWB.

    Layout (matches AllenSDK):
      - ``nwb.processing['running']['speed']``: low-pass-filtered cm/s
      - ``nwb.processing['running']['dx']``: raw angular change from pkl
      - ``nwb.acquisition['v_sig']``, ``nwb.acquisition['v_in']``: raw V
    """
    df = compute_running_speed(pkl, stim_vsync_fall)
    timestamps = df.index.values

    speed_ts = TimeSeries(
        name='speed', data=df['speed'].values, timestamps=timestamps,
        unit='cm/s',
        description='Mouse running speed on the wheel, computed from the '
                    'rotary-encoder voltage signal (AllenSDK pipeline: '
                    'unwrap → angular change → linear speed via wheel '
                    'geometry → wrap-artifact clip → 10σ z-score outlier '
                    'rejection → 3rd-order 4 Hz Butterworth low-pass). '
                    'Timestamps are sync-file vsync falling edges (no '
                    'monitor delay; running is a behavioral signal).')
    dx_ts = TimeSeries(
        name='dx', data=df['dx'].values, timestamps=timestamps, unit='cm',
        description='Running-wheel angular change (raw pkl encoder dx).')
    v_sig_ts = TimeSeries(
        name='v_sig', data=df['v_sig'].values, timestamps=timestamps, unit='V',
        description='Raw voltage signal from the running-wheel encoder.')
    v_in_ts = TimeSeries(
        name='v_in', data=df['v_in'].values, timestamps=timestamps, unit='V',
        description='Theoretical max encoder voltage (nominally 5 V, varies '
                    'in practice; used to normalise the wrap).')

    if 'running' in nwb.processing:
        mod = nwb.processing['running']
    else:
        mod = ProcessingModule(name='running',
                               description='Running speed processing module')
        nwb.add_processing_module(mod)
    mod.add_data_interface(speed_ts)
    mod.add_data_interface(dx_ts)
    nwb.add_acquisition(v_sig_ts)
    nwb.add_acquisition(v_in_ts)


# ── Builders ───────────────────────────────────────────────────────────
def build_subject(pkl: dict, metadata: dict) -> Subject:
    """Build a Subject from pkl mouse_id, with optional metadata overrides."""
    mouse_id = str(pkl['items']['behavior']['params'].get('mouse_id', 'unknown'))
    return Subject(
        subject_id=mouse_id,
        species=metadata.get('species', 'Mus musculus'),
        age=metadata.get('age'),
        sex=metadata.get('sex', 'U'),  # U = unknown
        genotype=metadata.get('genotype'),
        description=metadata.get('subject_description'),
    )


def build_nwbfile(pkl: dict, metadata: dict) -> NdxEventsNWBFile:
    """Create an empty NdxEventsNWBFile with task-level metadata."""
    params = pkl['items']['behavior']['params']
    start_time = pkl.get('start_time')
    if start_time is None:
        start_time = datetime.datetime.now()
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)

    nwb = NdxEventsNWBFile(
        session_description=metadata.get(
            'session_description', params.get('stage', 'change detection')),
        identifier=metadata.get('identifier', str(uuid4())),
        session_start_time=start_time,
        experimenter=metadata.get('experimenter'),
        lab=metadata.get('lab'),
        institution=metadata.get('institution'),
        notes=metadata.get('notes'),
    )
    nwb.subject = build_subject(pkl, metadata)
    return nwb


def build_stimulus_presentations(
    events_df: pd.DataFrame,
    epoch_list: list[dict] | None = None,
) -> TimeIntervals:
    """One row per active-task image flash + omitted slot. Matches reference NWB.

    Built via bulk column construction (much faster than add_row in a loop).
    """
    onsets = events_df[events_df['event_type'] == 'image_onset']
    offsets = events_df[events_df['event_type'] == 'image_offset']
    omitted = events_df[events_df['event_type'] == 'image_omission']
    change_sids = set(
        events_df.loc[events_df['event_type'] == 'image_change',
                       'stimulus_presentations_id'].astype(int).values)
    offsets_by_sid = dict(zip(
        offsets['stimulus_presentations_id'].astype(int),
        offsets['timestamp']))
    offset_frames_by_sid = dict(zip(
        offsets['stimulus_presentations_id'].astype(int),
        offsets['frame'].astype(int)))

    # Build per-row tuples, then sort by start_time, then unzip into columns.
    rows = []
    for _, row in onsets.iterrows():
        sid = int(row['stimulus_presentations_id'])
        on_t = float(row['timestamp'])
        off_t = float(offsets_by_sid.get(sid, on_t + 0.25))
        on_f = int(row['frame'])
        off_f = int(offset_frames_by_sid.get(sid, on_f + 15))
        is_change = sid in change_sids
        tid = int(row['trials_id']) if pd.notna(row['trials_id']) else -1
        rows.append((on_t, off_t, str(row['image_name']), is_change, False,
                     sid, tid, on_f, off_f))
    for _, row in omitted.iterrows():
        sid = (int(row['stimulus_presentations_id'])
               if pd.notna(row['stimulus_presentations_id']) else -1)
        on_t = float(row['timestamp'])
        on_f = int(row['frame'])
        tid = int(row['trials_id']) if pd.notna(row['trials_id']) else -1
        # Omitted slot: synthetic 250ms / 15-frame duration
        rows.append((on_t, on_t + 0.25, 'omitted', False, True, sid, tid,
                     on_f, on_f + 15))
    rows.sort(key=lambda r: r[0])

    start_time = [r[0] for r in rows]
    stop_time = [r[1] for r in rows]
    image_name = [r[2] for r in rows]
    is_change = [r[3] for r in rows]
    omitted_flag = [r[4] for r in rows]
    sids = [r[5] for r in rows]
    tids = [r[6] for r in rows]
    start_frames = [r[7] for r in rows]
    stop_frames = [r[8] for r in rows]
    hed = [_stim_presentation_hed(n, c, o)
           for n, c, o in zip(image_name, is_change, omitted_flag)]
    epoch_names = ([epoch_name_at(t, epoch_list) for t in start_time]
                   if epoch_list is not None else ['change_detection'] * len(rows))

    # Per-presentation lick latency: time from this stim's onset to the first
    # lick that falls before the next presentation. NaN if no lick in that
    # window. Was previously stored on the events table for lick events; lives
    # on stimulus_presentations now since it's intrinsically stim-relative.
    import numpy as _np  # local alias to keep the surgical edit obvious
    lick_times = _np.asarray(
        events_df.loc[events_df['event_type'] == 'lick', 'timestamp'].values,
        dtype=float,
    )
    starts = _np.asarray(start_time, dtype=float)
    # next_start[i] = start_time of the (i+1)th presentation; last row gets +inf
    next_start = _np.concatenate([starts[1:], [_np.inf]])
    lick_latency = []
    for s, n in zip(starts, next_start):
        idx = _np.searchsorted(lick_times, s, side='left')
        if idx < len(lick_times) and lick_times[idx] < n:
            lick_latency.append(float(lick_times[idx] - s))
        else:
            lick_latency.append(_np.nan)

    return TimeIntervals(
        name='stimulus_presentations',
        description='Per-flash stimulus presentations during the active '
                    'change-detection task. Omitted flashes are represented '
                    'with image_name="omitted" and omitted=True.',
        columns=[
            VectorData(name='start_time', description='Flash onset (s).',
                       data=start_time),
            VectorData(name='stop_time', description='Flash offset (s).',
                       data=stop_time),
            VectorData(name='image_name', description='Image identity, or "omitted".',
                       data=image_name),
            VectorData(name='is_change',
                       description='True if image identity differs from the '
                                   'previous (non-omitted) flash.',
                       data=is_change),
            VectorData(name='omitted',
                       description='True for withheld flashes (no image shown).',
                       data=omitted_flag),
            VectorData(name='stimulus_presentations_id',
                       description='Sequential id matching the events table.',
                       data=sids),
            VectorData(name='trials_id',
                       description='Id of the trial this presentation belongs '
                                   'to (-1 if outside any trial).',
                       data=tids),
            VectorData(name='start_frame',
                       description='Vsync falling-edge frame index for the flash onset.',
                       data=start_frames),
            VectorData(name='stop_frame',
                       description='Vsync falling-edge frame index for the flash offset.',
                       data=stop_frames),
            VectorData(name='lick_latency',
                       description='Time from this stim onset to the first lick '
                                   'occurring before the next stim onset, in '
                                   'seconds. NaN if no lick in that window.',
                       data=lick_latency),
            VectorData(name='epoch_name',
                       description='Canonical epoch containing this presentation.',
                       data=epoch_names),
            HedTags(name='HED',
                    description='HED tag string for this presentation.',
                    data=hed),
        ],
        id=list(range(len(rows))),
    )


def build_natural_movie_one_presentations(events_df: pd.DataFrame) -> TimeIntervals | None:
    """One row per movie frame. Returns None if no movie events present.

    Built via bulk column construction.
    """
    onsets = events_df[events_df['event_type'] == 'movie_onset'].sort_values('timestamp')
    offsets = events_df[events_df['event_type'] == 'movie_offset'].sort_values('timestamp')
    if len(onsets) == 0:
        return None

    n = len(onsets)
    hed_str = ('Sensory-event, Visual-presentation, '
               '(Movie, Label/natural_movie_one)')

    start_time = onsets['timestamp'].astype(float).tolist()
    stop_time = offsets['timestamp'].astype(float).tolist()
    start_frames = onsets['frame'].astype(int).tolist()
    stop_frames = offsets['frame'].astype(int).tolist()
    frame_idx = onsets['movie_frame_index'].fillna(-1).astype(int).tolist()
    repeat = onsets['movie_repeat'].fillna(-1).astype(int).tolist()
    epoch_names = ['natural_movie_one'] * n  # all movie frames are in this epoch

    return TimeIntervals(
        name='natural_movie_one_presentations',
        description='Per-frame presentations of the natural_movie_one '
                    'fingerprint clip shown after the active task.',
        columns=[
            VectorData(name='start_time', description='Frame onset (s).',
                       data=start_time),
            VectorData(name='stop_time', description='Frame offset (s).',
                       data=stop_time),
            VectorData(name='movie_frame_index',
                       description='Frame index within a single movie playback.',
                       data=frame_idx),
            VectorData(name='movie_repeat',
                       description='Repetition of the movie this frame belongs to.',
                       data=repeat),
            VectorData(name='start_frame',
                       description='Vsync falling-edge frame index for the movie-frame onset.',
                       data=start_frames),
            VectorData(name='stop_frame',
                       description='Vsync falling-edge frame index for the movie-frame offset.',
                       data=stop_frames),
            VectorData(name='epoch_name',
                       description='Canonical epoch containing this frame.',
                       data=epoch_names),
            HedTags(name='HED',
                    description='HED tag string for this movie frame.',
                    data=[hed_str] * n),
        ],
        id=list(range(n)),
    )


def add_trials(
    nwb: NdxEventsNWBFile,
    intervals_df: pd.DataFrame,
    warm_up_n: int,
    epoch_list: list[dict] | None = None,
) -> None:
    """Populate nwb.trials from intervals_df."""
    trials = intervals_df[intervals_df['interval_type'] == 'trial'].copy()
    cw = intervals_df[intervals_df['interval_type'] == 'change_window']
    rw = intervals_df[intervals_df['interval_type'] == 'response_window']

    # Map trial_id -> change/response window times
    cw_by_tid = {int(r['trials_id']): (r['start_time'], r['stop_time'])
                 for _, r in cw.iterrows()}
    rw_by_tid = {int(r['trials_id']): (r['start_time'], r['stop_time'])
                 for _, r in rw.iterrows()}

    columns = [
        ('go', 'Go trial (change presented).'),
        ('catch', 'Catch trial (no change presented).'),
        ('auto_rewarded', 'Trial with automatic reward delivery.'),
        ('aborted', 'Trial aborted by early lick.'),
        ('hit', 'Lick during response window after a change.'),
        ('miss', 'No lick during response window after a change.'),
        ('false_alarm', 'Lick during catch trial response window or change window.'),
        ('correct_reject', 'No lick on a catch trial.'),
        ('warm_up', 'True for the first N warm-up trials.'),
        ('change_time', 'Time of image change on this trial (NaN if no change).'),
        ('change_frame', 'Frame index of change (-1 if no change).'),
        ('initial_image_name', 'Image shown at trial start.'),
        ('change_image_name', 'Image shown after change (same as initial if catch/abort).'),
        ('reward_time', 'Time of reward delivery (NaN if none).'),
        ('reward_volume', 'Volume of reward delivered (mL).'),
        ('response_time', 'Time of first lick after change (NaN if none).'),
        ('response_latency', 'Latency from change to first lick (NaN if none).'),
        ('change_window_start_time', 'Start of change window (NaN if none).'),
        ('change_window_stop_time', 'Stop of change window (NaN if none).'),
        ('response_window_start_time', 'Start of response window (NaN if none).'),
        ('response_window_stop_time', 'Stop of response window (NaN if none).'),
        ('epoch_name', 'Canonical epoch containing this trial.'),
    ]
    for name, desc in columns:
        nwb.add_trial_column(name=name, description=desc)
    nwb.add_trial_column(name='HED',
                         description='HED tag string for the trial outcome.',
                         col_cls=HedTags)

    for i, (_, row) in enumerate(trials.iterrows()):
        tid = int(row['trials_id'])
        cw_t = cw_by_tid.get(tid, (np.nan, np.nan))
        rw_t = rw_by_tid.get(tid, (np.nan, np.nan))
        nwb.add_trial(
            start_time=float(row['start_time']),
            stop_time=float(row['stop_time']),
            go=bool(row['go']),
            catch=bool(row['catch']),
            auto_rewarded=bool(row['auto_rewarded']),
            aborted=bool(row['aborted']),
            hit=bool(row['hit']),
            miss=bool(row['miss']),
            false_alarm=bool(row['false_alarm']),
            correct_reject=bool(row['correct_reject']),
            warm_up=(i < warm_up_n),
            change_time=float(row['change_time']) if pd.notna(row['change_time']) else np.nan,
            change_frame=int(row['change_frame']) if pd.notna(row['change_frame']) else -1,
            initial_image_name=str(row['initial_image_name']) if pd.notna(row['initial_image_name']) else '',
            change_image_name=str(row['change_image_name']) if pd.notna(row['change_image_name']) else '',
            reward_time=float(row['reward_time']) if pd.notna(row['reward_time']) else np.nan,
            reward_volume=float(row['reward_volume']) if pd.notna(row['reward_volume']) else 0.0,
            response_time=float(row['response_time']) if pd.notna(row['response_time']) else np.nan,
            response_latency=float(row['response_latency']) if pd.notna(row['response_latency']) else np.nan,
            change_window_start_time=float(cw_t[0]) if pd.notna(cw_t[0]) else np.nan,
            change_window_stop_time=float(cw_t[1]) if pd.notna(cw_t[1]) else np.nan,
            response_window_start_time=float(rw_t[0]) if pd.notna(rw_t[0]) else np.nan,
            response_window_stop_time=float(rw_t[1]) if pd.notna(rw_t[1]) else np.nan,
            epoch_name=(epoch_name_at(float(row['start_time']), epoch_list)
                        if epoch_list is not None else 'change_detection'),
            HED=_trial_hed(row),
        )


def build_epoch_lookup(
    intervals_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> list[dict]:
    """Build the canonical session epoch list.

    - Folds 'warm_up' rows into 'change_detection' (start = warm_up.start).
    - Inserts 'spontaneous' rows to fill any gap before the first epoch,
      between epochs, or after the last epoch (up to the latest timestamp
      in the events or intervals dataframes).

    Returns
    -------
    list of dicts with keys ``name``, ``start``, ``stop``, sorted by start.
    """
    raw = intervals_df[intervals_df['interval_type'] == 'epoch'].sort_values('start_time')

    named: list[dict] = []
    if 'change_detection' in raw['label'].values:
        cd = raw[raw['label'] == 'change_detection'].iloc[0]
        cd_start = float(cd['start_time'])
        cd_stop = float(cd['stop_time'])
        # Fold warm_up in if present (warm_up always precedes change_detection)
        if 'warm_up' in raw['label'].values:
            wu = raw[raw['label'] == 'warm_up'].iloc[0]
            cd_start = float(wu['start_time'])
        named.append({'name': 'change_detection', 'start': cd_start, 'stop': cd_stop})
    if 'natural_movie_one' in raw['label'].values:
        nm = raw[raw['label'] == 'natural_movie_one'].iloc[0]
        named.append({'name': 'natural_movie_one',
                      'start': float(nm['start_time']),
                      'stop': float(nm['stop_time'])})
    named.sort(key=lambda e: e['start'])

    # Session end = max timestamp across events + intervals
    session_end = max(
        float(events_df['timestamp'].max()),
        float(intervals_df['stop_time'].max()),
    )

    out: list[dict] = []
    prev = 0.0
    for ep in named:
        if ep['start'] - prev > 1e-6:
            out.append({'name': 'spontaneous', 'start': prev, 'stop': ep['start']})
        out.append(ep)
        prev = ep['stop']
    if session_end - prev > 1e-6:
        out.append({'name': 'spontaneous', 'start': prev, 'stop': session_end})
    return out


def epoch_name_at(t: float, epoch_list: list[dict]) -> str:
    """Return the epoch_name containing time t (or 'spontaneous' if none)."""
    for ep in epoch_list:
        if ep['start'] <= t < ep['stop']:
            return ep['name']
    return 'spontaneous'


def add_epochs(nwb: NdxEventsNWBFile, epoch_list: list[dict]) -> None:
    """Populate nwb.epochs from the canonical epoch list.

    Epoch names are 'change_detection', 'natural_movie_one', 'spontaneous'.
    """
    nwb.add_epoch_column(
        name='epoch_name',
        description='Canonical epoch name. One of: change_detection, '
                    'natural_movie_one, spontaneous.')
    nwb.add_epoch_column(
        name='HED', description='HED tag for the epoch.', col_cls=HedTags)
    for ep in epoch_list:
        nwb.add_epoch(
            start_time=ep['start'], stop_time=ep['stop'],
            tags=[ep['name']],
            epoch_name=ep['name'],
            HED=_EPOCH_HED.get(ep['name'], ''),
        )


def build_events_table(events_df: pd.DataFrame) -> tuple[EventsTable, list[MeaningsTable]]:
    """Build the ndx-events EventsTable of discrete (point) events.

    Includes lick, reward, image_change, and image_omission. Visual
    onsets/offsets (image_onset/image_offset/movie_onset/movie_offset)
    are intervals and live on the intervals/stimulus_presentations/
    natural_movie_one_presentations tables instead. Miss is a trial-
    level outcome and lives on nwb.trials, not here.

    Orthogonal categorical columns follow HED's PASS design:
      - event_type: base action
      - lick_classification: task context of lick (hit, abort, etc.;
        'n/a' for non-licks)
      - reward_type: earned vs auto_reward ('n/a' for non-rewards)
      - lick_bouts: temporal bout structure (bout_start vs within_bout
        for lick events; 'n/a' for non-lick events)

    Each categorical column has its own MeaningsTable with HED fragments
    that compose into the full tag string.
    """
    df = events_df[~events_df['event_type'].isin(_DROP_EVENT_TYPES)]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── MeaningsTables (one per categorical column) ──
    # Each table carries a `value_description` column with a plain-English
    # meaning of every value, alongside the HED-tag fragment in `meaning`.
    # NOTE: the column is named `value_description` rather than `description`
    # because MeaningsTable already has a table-level `description` attribute
    # (the table's own description) — using `description` as a column name
    # collides at HDF5 write time.
    def _build_meanings(name, table_desc, hed_dict, desc_dict):
        mt = MeaningsTable(name=name, description=table_desc)
        mt.add_column(
            name='value_description',
            description='Plain-English semantic description of the value.',
        )
        for value, hed in hed_dict.items():
            mt.add_row(
                value=value,
                meaning=hed,
                value_description=desc_dict.get(value, ''),
            )
        return mt

    event_type_meanings = _build_meanings(
        'event_type_meanings',
        'Base HED tag string and semantic description for each event type.',
        _EVENT_TYPE_HED, _EVENT_TYPE_DESC,
    )
    lick_classification_meanings = _build_meanings(
        'lick_classification_meanings',
        'HED fragment and semantic description for the task context of a '
        'lick. Composes with the base lick tag from event_type_meanings.',
        _LICK_CLASSIFICATION_HED, _LICK_CLASSIFICATION_DESC,
    )
    reward_type_meanings = _build_meanings(
        'reward_type_meanings',
        'HED fragment and semantic description for reward type (earned vs '
        'auto_reward). Composes with the base reward tag from '
        'event_type_meanings.',
        _REWARD_TYPE_HED, _REWARD_TYPE_DESC,
    )
    lick_bouts_meanings = _build_meanings(
        'lick_bouts_meanings',
        'HED fragment and semantic description indicating whether a lick '
        'starts a bout (within-bout licks have no marker; bout-start licks '
        'carry a Temporal-marker tag). "n/a" for non-lick events.',
        _LICK_BOUTS_HED, _LICK_BOUTS_DESC,
    )

    all_meanings = [event_type_meanings, lick_classification_meanings,
                    reward_type_meanings, lick_bouts_meanings]

    # ── Build column data vectors ──
    n = len(df)
    is_lick = (df['event_type'] == 'lick').values
    is_reward = (df['event_type'] == 'reward').values

    lick_cls = ['n/a'] * n
    reward_tp = ['n/a'] * n
    bout = ['n/a'] * n
    for i, (_, row) in enumerate(df.iterrows()):
        if is_lick[i]:
            if pd.notna(row.get('lick_classification')):
                lick_cls[i] = str(row['lick_classification'])
            bout[i] = 'bout_start' if bool(row.get('bout_start')) else 'within_bout'
        elif is_reward[i] and pd.notna(row.get('reward_type')):
            reward_tp[i] = str(row['reward_type'])

    # Set up schema with add_column (no data), then populate data lists
    # directly. This avoids both the circular-ref issue from the bulk
    # columns= constructor (broken HDF5 references) and the slowness of
    # per-row add_row() for 30k+ events.
    et = EventsTable(
        name='events',
        description='All behavior and task events with orthogonal categorical '
                    'columns following HED PASS design. event_type gives the '
                    'base action; lick_classification, reward_type, and '
                    'lick_bouts provide independent dimensions that compose '
                    'into the full HED string.',
        meanings_tables=all_meanings,
    )
    et.add_column(name='event_type',
                  description='Event type: lick, reward, image_change, '
                              'image_omission.',
                  col_cls=CategoricalVectorData,
                  meanings=event_type_meanings)
    et.add_column(name='lick_classification',
                  description='Task context of a lick event (hit, false_alarm, '
                              'abort, early, late, consumption, spontaneous). '
                              '"n/a" for non-lick events.',
                  col_cls=CategoricalVectorData,
                  meanings=lick_classification_meanings)
    et.add_column(name='reward_type',
                  description='Reward type (earned or auto_reward). '
                              '"n/a" for non-reward events.',
                  col_cls=CategoricalVectorData,
                  meanings=reward_type_meanings)
    et.add_column(name='lick_bouts',
                  description='For lick events: bout_start vs within_bout. '
                              'For non-lick events: n/a.',
                  col_cls=CategoricalVectorData,
                  meanings=lick_bouts_meanings)
    et.add_column(name='trials_id',
                  description='Id of trial this event belongs to (-1 if none).')
    et.add_column(name='stimulus_presentations_id',
                  description='Id of stimulus presentation (-1 if none).')
    et.add_column(name='image_name',
                  description='Image name (for image-related events; empty otherwise).')
    et.add_column(name='frame',
                  description='Vsync falling-edge frame index (-1 if N/A).')
    et.add_column(name='reward_volume',
                  description='Volume of reward in mL (NaN if not a reward).')

    # Populate all column data directly (fast bulk approach)
    et.timestamp.data.extend(df['timestamp'].astype(float).tolist())
    et.id.data.extend(list(range(n)))
    et['event_type'].data.extend(df['event_type'].tolist())
    et['lick_classification'].data.extend(lick_cls)
    et['reward_type'].data.extend(reward_tp)
    et['lick_bouts'].data.extend(bout)
    et['trials_id'].data.extend(
        [int(v) if pd.notna(v) else -1 for v in df['trials_id']])
    et['stimulus_presentations_id'].data.extend(
        [int(v) if pd.notna(v) else -1 for v in df['stimulus_presentations_id']])
    et['image_name'].data.extend(
        [str(v) if pd.notna(v) else '' for v in df['image_name']])
    et['frame'].data.extend(
        [int(v) if pd.notna(v) else -1 for v in df['frame']])
    et['reward_volume'].data.extend(
        [float(v) if pd.notna(v) else np.nan for v in df['reward_volume']])

    return et, all_meanings


def build_intervals_table(
    intervals_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> TimeIntervals:
    """Build the canonical flat TimeIntervals table — timing only.

    Contains all interval types in a single table with start/stop times,
    a discriminator (``interval_type``), a label, foreign-key columns
    pointing to the per-type annotation tables (``trials_id``,
    ``stimulus_presentations_id``, ``natural_movie_one_presentations_id``),
    and a HED tag. Task-specific annotation columns (go/catch/hit/miss/
    image_name/etc.) live on the per-type physical tables, not here.

    Interval types included:
      - ``epoch``: session epochs (change_detection, natural_movie_one,
        spontaneous; warm_up is folded into change_detection upstream).
      - ``trial``: behavioral trials.
      - ``change_window`` / ``response_window``: task-defined windows.
      - ``stimulus_presentation``: per-flash image presentations
        (image_name carried as ``label``).
      - ``movie_frame``: per-frame natural_movie_one presentations.
    """
    epoch_list = build_epoch_lookup(intervals_df, events_df)

    rows: list[dict] = []

    # ── Epochs (timing-only, from canonical epoch list) ──
    for ep in epoch_list:
        rows.append({
            'start_time': float(ep['start']),
            'stop_time': float(ep['stop']),
            'interval_type': 'epoch',
            'label': ep['name'],
            'trials_id': -1,
            'stimulus_presentations_id': -1,
            'natural_movie_one_presentations_id': -1,
            'HED': _EPOCH_HED.get(ep['name'], ''),
        })

    # ── Trials / change_window / response_window (from intervals_df) ──
    iv = intervals_df.sort_values('start_time').reset_index(drop=True)
    for _, r in iv.iterrows():
        itype = r['interval_type']
        if itype == 'epoch':
            continue  # already handled via canonical list
        if itype == 'trial':
            hed = _trial_hed(r)
        else:
            hed = str(r['hed_string']) if pd.notna(r.get('hed_string')) else ''
        rows.append({
            'start_time': float(r['start_time']),
            'stop_time': float(r['stop_time']),
            'interval_type': itype,
            'label': '',
            'trials_id': int(r['trials_id']) if pd.notna(r.get('trials_id')) else -1,
            'stimulus_presentations_id': -1,
            'natural_movie_one_presentations_id': -1,
            'HED': hed,
        })

    # ── Stimulus presentations ──
    onsets = events_df[events_df['event_type'] == 'image_onset']
    offsets = events_df[events_df['event_type'] == 'image_offset']
    omitted = events_df[events_df['event_type'] == 'image_omission']
    change_sids = set(
        events_df.loc[events_df['event_type'] == 'image_change',
                       'stimulus_presentations_id'].astype(int).values)
    offsets_by_sid = dict(zip(
        offsets['stimulus_presentations_id'].astype(int),
        offsets['timestamp']))

    stim_rows = []
    for _, row in onsets.iterrows():
        sid = int(row['stimulus_presentations_id'])
        on_t = float(row['timestamp'])
        off_t = float(offsets_by_sid.get(sid, on_t + 0.25))
        image_name = str(row['image_name'])
        is_change = sid in change_sids
        stim_rows.append((on_t, off_t, sid, image_name, False, is_change))
    for _, row in omitted.iterrows():
        sid = (int(row['stimulus_presentations_id'])
               if pd.notna(row['stimulus_presentations_id']) else -1)
        on_t = float(row['timestamp'])
        stim_rows.append((on_t, on_t + 0.25, sid, 'omitted', True, False))
    stim_rows.sort(key=lambda r: r[0])
    # Reassign sequential sids in chronological order so they match the
    # stimulus_presentations physical table row ids.
    for new_sid, (on_t, off_t, _, image_name, is_omitted, is_change) in enumerate(stim_rows):
        rows.append({
            'start_time': on_t,
            'stop_time': off_t,
            'interval_type': 'stimulus_presentation',
            'label': image_name,
            'trials_id': -1,
            'stimulus_presentations_id': new_sid,
            'natural_movie_one_presentations_id': -1,
            'HED': _stim_presentation_hed(image_name, is_change, is_omitted),
        })

    # ── Movie frames ──
    movie_onsets = events_df[events_df['event_type'] == 'movie_onset'].sort_values('timestamp')
    movie_offsets = events_df[events_df['event_type'] == 'movie_offset'].sort_values('timestamp')
    movie_hed = ('Sensory-event, Visual-presentation, '
                 '(Movie, Label/natural_movie_one)')
    for mid, ((_, on), (_, off)) in enumerate(zip(movie_onsets.iterrows(),
                                                   movie_offsets.iterrows())):
        rows.append({
            'start_time': float(on['timestamp']),
            'stop_time': float(off['timestamp']),
            'interval_type': 'movie_frame',
            'label': '',
            'trials_id': -1,
            'stimulus_presentations_id': -1,
            'natural_movie_one_presentations_id': mid,
            'HED': movie_hed,
        })

    rows.sort(key=lambda r: r['start_time'])

    return TimeIntervals(
        name='intervals',
        description='Canonical flat table of every session interval — '
                    'epochs, trials, change_windows, response_windows, '
                    'stimulus_presentations, and movie_frames. Contains '
                    'only timing, type, label, foreign keys, and HED. '
                    'Task-specific annotations live on the per-type physical '
                    'tables (trials, stimulus_presentations, '
                    'natural_movie_one_presentations).',
        columns=[
            VectorData(name='start_time', description='Interval start (s).',
                       data=[r['start_time'] for r in rows]),
            VectorData(name='stop_time', description='Interval stop (s).',
                       data=[r['stop_time'] for r in rows]),
            VectorData(name='interval_type',
                       description='One of: epoch, trial, change_window, '
                                   'response_window, stimulus_presentation, '
                                   'movie_frame.',
                       data=[r['interval_type'] for r in rows]),
            VectorData(name='label',
                       description='Descriptive label (epoch name; image_name '
                                   'for stimulus_presentation rows; empty otherwise).',
                       data=[r['label'] for r in rows]),
            VectorData(name='trials_id',
                       description='Foreign key into the trials table '
                                   '(-1 if N/A).',
                       data=[r['trials_id'] for r in rows]),
            VectorData(name='stimulus_presentations_id',
                       description='Foreign key into the stimulus_presentations '
                                   'table (-1 if N/A).',
                       data=[r['stimulus_presentations_id'] for r in rows]),
            VectorData(name='natural_movie_one_presentations_id',
                       description='Foreign key into the '
                                   'natural_movie_one_presentations table '
                                   '(-1 if N/A).',
                       data=[r['natural_movie_one_presentations_id'] for r in rows]),
            HedTags(name='HED', description='HED tag string for this interval.',
                    data=[r['HED'] for r in rows]),
        ],
        id=list(range(len(rows))),
    )


# ── Sidecar JSON ──────────────────────────────────────────────────────
def build_events_sidecar() -> dict:
    """Build the BIDS-style sidecar JSON describing every column used in
    the events, intervals, trials, stimulus_presentations, and
    natural_movie_one_presentations tables.

    Structure follows the VRF example sidecar:
      - Categorical columns have ``Description``, ``Levels`` (value →
        semantic description) and ``HED`` (value → HED fragment).
      - Descriptive value / id columns have ``Description`` and (optionally)
        a ``HED`` template with a ``#`` placeholder.
      - A trailing ``hed_defs`` block reserves space for HED definitions
        used by the templates above.

    The returned dict is JSON-serialisable; ``package_to_nwb()`` writes it
    to ``<output>.events.json`` alongside the NWB.
    """
    sidecar: dict = {}

    # ── Categorical columns (Description + Levels + HED) ──
    def _categorical(desc_dict, hed_dict, column_description):
        return {
            'Description': column_description,
            'Levels': dict(desc_dict),
            'HED': {k: v for k, v in hed_dict.items() if v},
        }

    sidecar['event_type'] = _categorical(
        _EVENT_TYPE_DESC, _EVENT_TYPE_HED,
        _VALUE_COLUMN_DESC['event_type'],
    )
    sidecar['lick_classification'] = _categorical(
        _LICK_CLASSIFICATION_DESC, _LICK_CLASSIFICATION_HED,
        _VALUE_COLUMN_DESC['lick_classification'],
    )
    sidecar['reward_type'] = _categorical(
        _REWARD_TYPE_DESC, _REWARD_TYPE_HED,
        _VALUE_COLUMN_DESC['reward_type'],
    )
    sidecar['lick_bouts'] = _categorical(
        _LICK_BOUTS_DESC, _LICK_BOUTS_HED,
        _VALUE_COLUMN_DESC['lick_bouts'],
    )
    sidecar['interval_type'] = {
        'Description': _VALUE_COLUMN_DESC['interval_type'],
        'Levels': dict(_INTERVAL_TYPE_DESC),
        # No per-interval-type base HED — HED on intervals is composed from
        # the interval contents (epoch name, trial outcome, image_name, ...).
    }
    sidecar['epoch_name'] = _categorical(
        _EPOCH_DESC, _EPOCH_HED,
        _VALUE_COLUMN_DESC['epoch_name'],
    )
    # Trial outcome is not a stored column but a derived discriminator;
    # we include it so the per-outcome HED on `nwb.trials` HED is documented.
    sidecar['trial_outcome'] = _categorical(
        _TRIAL_OUTCOME_DESC, _TRIAL_OUTCOME_HED,
        'Composite outcome of a trial, derived from the boolean go/catch/'
        'hit/miss/false_alarm/correct_reject/aborted/auto_rewarded columns. '
        'Used to compose the per-trial HED tag.',
    )

    # ── Descriptive value / id / metadata columns ──
    # Anything in _VALUE_COLUMN_DESC that hasn't already been added.
    for col, desc in _VALUE_COLUMN_DESC.items():
        if col in sidecar:
            continue
        entry: dict = {'Description': desc}
        if col in _VALUE_COLUMN_HED:
            entry['HED'] = _VALUE_COLUMN_HED[col]
        sidecar[col] = entry

    # ── HED Definitions block ──
    # Reserved for any (Definition/...) macros referenced by the HED tags
    # above. None are currently needed because all tags use the base
    # HED v8.3.0 schema, but the block matches the VRF sidecar format.
    sidecar['hed_defs'] = {
        'HED': {
            'alldefs': '',
        },
    }

    return sidecar


# ── Main entry point ──────────────────────────────────────────────────
def package_to_nwb(
    pkl_path: str | Path,
    sync_path: str | Path,
    output_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Run the full pipeline and write a HED-annotated NWB file.

    Parameters
    ----------
    pkl_path : path to camstim _stim.pkl
    sync_path : path to camstim _sync.h5
    output_path : where to write the .nwb
    metadata : optional dict to override defaults — keys: session_description,
        identifier, experimenter, lab, institution, notes, species, age, sex,
        genotype, subject_description.

    Returns
    -------
    Path to the written file.
    """
    metadata = metadata or {}
    output_path = Path(output_path)

    logger.info('Building events and intervals tables...')
    res = build_all(str(pkl_path), str(sync_path))
    events_df = res['events_df']
    intervals_df = res['intervals_df']
    task_parameters = res['task_parameters']
    stim_vsync_fall = res['timestamp_data']['stim_vsync_fall']

    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f, encoding='latin1')
    warm_up_n = int(pkl['items']['behavior']['params'].get('warm_up_trials', 0))

    logger.info('Creating NWBFile...')
    nwb = build_nwbfile(pkl, metadata)
    nwb.add_lab_meta_data(task_parameters)
    nwb.add_lab_meta_data(HedLabMetaData(hed_schema_version=HED_SCHEMA_VERSION))

    logger.info('Building canonical epoch list (with spontaneous gaps)...')
    epoch_list = build_epoch_lookup(intervals_df, events_df)
    for ep in epoch_list:
        logger.info(f"  {ep['name']:18s} {ep['start']:9.2f} → {ep['stop']:9.2f}")

    logger.info('Adding stimulus_presentations TimeIntervals...')
    nwb.add_time_intervals(build_stimulus_presentations(events_df, epoch_list))

    movie_ti = build_natural_movie_one_presentations(events_df)
    if movie_ti is not None:
        logger.info('Adding natural_movie_one_presentations TimeIntervals...')
        nwb.add_time_intervals(movie_ti)

    logger.info('Adding trials table...')
    add_trials(nwb, intervals_df, warm_up_n, epoch_list)

    # Note: nwb.epochs is intentionally NOT set. Session epochs live as rows
    # in the flat intervals table (interval_type='epoch'). Setting both
    # creates two views of the same data in the file layout.

    logger.info('Adding events table...')
    et, _ = build_events_table(events_df)
    nwb.add_events_table(et)

    logger.info('Adding flat intervals table...')
    nwb.add_time_intervals(build_intervals_table(intervals_df, events_df))

    logger.info('Adding running speed (processing/running) + raw encoder traces...')
    add_running_speed(nwb, pkl, stim_vsync_fall)

    logger.info(f'Writing NWB to {output_path}')
    with NWBHDF5IO(str(output_path), 'w') as io:
        io.write(nwb)
    logger.info(f'Done. File size: {output_path.stat().st_size / 1e6:.1f} MB')

    # Sidecar JSON: column-level Description / Levels / HED, BIDS-style.
    sidecar_path = output_path.with_suffix('.events.json')
    with open(sidecar_path, 'w') as f:
        json.dump(build_events_sidecar(), f, indent=2, ensure_ascii=False)
    logger.info(f'Wrote sidecar JSON to {sidecar_path}')

    return output_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path')
    parser.add_argument('sync_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    package_to_nwb(args.pkl_path, args.sync_path, args.output_path)
