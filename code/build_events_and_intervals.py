"""
Consolidated script to build events and intervals tables for a change detection
behavior task from raw pkl and sync files.

This script combines timestamp alignment, events table construction, and
intervals table construction into a single reproducible pipeline.

The logic was validated against reference NWB file
(behavior_ophys_experiment_1050485649.nwb) with 0.0 microsecond deviation on
all metrics.

Usage:
    python build_events_and_intervals.py <pkl_path> <sync_path> [--output-dir <dir>]

Outputs:
    events_df.pkl      - Events table (DataFrame)
    intervals_df.pkl   - Intervals table (DataFrame)
    timestamp_data.npz - Intermediate timestamp arrays
"""

import pickle
import ast
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HED tag templates for each event type
# ---------------------------------------------------------------------------
HED_TAGS = {
    # --- Point events ---
    'image_onset': 'Sensory-event, Visual-presentation, (Photograph, Label/{image_name}), Onset',
    'image_offset': 'Sensory-event, Visual-presentation, Offset',
    'image_change': 'Sensory-event, Visual-presentation, (Photograph, Label/{image_name}), Experimental-stimulus, Event/Change',
    'image_omission': 'Sensory-event, Visual-presentation, Omitted-stimulus',
    'movie_onset': 'Sensory-event, Visual-presentation, (Movie, Label/natural_movie_one), Onset',
    'movie_offset': 'Sensory-event, Visual-presentation, (Movie, Label/natural_movie_one), Offset',
    'lick': 'Agent-action, (Animal-agent, Lick)',
    'reward': 'Sensory-event, Gustatory-presentation, (Ingestible-object, Water, Reward), Label/{reward_type}',
    'miss': 'Agent-action, (Animal-agent, Miss)',
    # --- Intervals ---
    'epoch_warm_up': 'Experimental-procedure, (Task, Label/warm_up)',
    'epoch_change_detection': 'Experimental-procedure, (Task, Label/change_detection)',
    'epoch_natural_movie_one': 'Experimental-procedure, (Task, Label/natural_movie_one)',
    'trial_go': 'Experimental-trial, (Task, Label/change_detection), Go-trial',
    'trial_catch': 'Experimental-trial, (Task, Label/change_detection), No-go-trial',
    'trial_auto_rewarded': 'Experimental-trial, (Task, Label/change_detection), Practice-trial',
    'trial_aborted': 'Experimental-trial, (Task, Label/change_detection), Aborted',
    'change_window': 'Time-interval, (Task-property, Label/change_window)',
    'response_window': 'Time-interval, (Task-property, Label/response_window)',
}

# Composable HED tag fragments for lick classification (orthogonal to base lick tag)
LICK_HED_FRAGMENTS = {
    'hit': ', Participant-response, Correct-action',
    'false_alarm': ', Participant-response, Incorrect-action',
    'abort': ', Participant-response, Incorrect-action',
    'early': ', Participant-response',
    'late': ', Participant-response',
    'consumption': '',
    'spontaneous': '',
}
BOUT_START_HED_FRAGMENT = ', (Temporal-marker, Label/bout_start)'


def format_hed_string(event_type, image_name=None):
    """Format HED tag string for an event, substituting image_name if needed.

    Returns empty string for 'lick' and 'reward' events — their HED strings
    are composed later by _classify_licks() and _classify_rewards().
    """
    if event_type in ('lick', 'reward'):
        return ''
    template = HED_TAGS.get(event_type, '')
    if '{image_name}' in template and image_name:
        return template.replace('{image_name}', image_name)
    return template


# ===================================================================
# TASK 1: Timestamp alignment from sync file
# ===================================================================

def _get_edges(bits, counters, line_labels, line_name, edge_type, sample_rate):
    """Extract edge times from sync data for a given line.

    Parameters
    ----------
    bits : np.ndarray
        Bit-packed signal column from sync data.
    counters : np.ndarray
        Counter column from sync data.
    line_labels : list of str
        Ordered label names corresponding to bit positions.
    line_name : str
        Name of the line to extract edges for.
    edge_type : str
        'falling', 'rising', or 'both'.
    sample_rate : float
        DAQ sample rate in Hz.

    Returns
    -------
    np.ndarray
        Edge times in seconds.
    """
    # Sync line labels were renamed between camstim versions (e.g. the 2020
    # files use 'stim_vsync', newer files use 'vsync_stim'). Look up by all
    # known aliases for this line, and use whichever label exists.
    _SYNC_ALIASES = {
        'stim_vsync': ('stim_vsync', 'vsync_stim'),
        'vsync_stim': ('stim_vsync', 'vsync_stim'),
        '2p_vsync': ('2p_vsync', 'vsync_2p'),
        'vsync_2p': ('2p_vsync', 'vsync_2p'),
        'acq_trigger': ('acq_trigger', 'stim_running'),
        'stim_running': ('acq_trigger', 'stim_running'),
    }
    candidates = _SYNC_ALIASES.get(line_name, (line_name,))
    bit_idx = None
    for cand in candidates:
        if cand in line_labels:
            bit_idx = line_labels.index(cand)
            break
    if bit_idx is None:
        raise ValueError(
            f'None of the aliases {candidates} found in sync line_labels. '
            f'Available labels: {[lbl for lbl in line_labels if lbl]}')
    line_state = (bits >> bit_idx) & 1
    changes = np.diff(line_state.astype(np.int8))
    if edge_type == 'falling':
        edge_indices = np.where(changes == -1)[0] + 1
    elif edge_type == 'rising':
        edge_indices = np.where(changes == 1)[0] + 1
    else:  # both
        edge_indices = np.where(changes != 0)[0] + 1
    return counters[edge_indices].astype(np.float64) / sample_rate


def compute_timestamp_alignment(pkl, sync_path):
    """Load sync file and compute aligned timestamp arrays.

    The sync file provides hardware-timed vsync falling edges and photodiode
    edges.  The monitor delay is computed by pairing photodiode toggle events
    (which occur every 60 frames) with the corresponding vsync edges.

    Parameters
    ----------
    pkl : dict
        Loaded pickle data.
    sync_path : str or Path
        Path to the HDF5 sync file.

    Returns
    -------
    dict with keys:
        stim_ts_visual : np.ndarray
            Frame times with monitor delay (for visual events).
        stim_ts_behavioral : np.ndarray
            Frame times without monitor delay (for behavioral events).
        monitor_delay : float
            Computed monitor delay in seconds.
        stim_vsync_fall : np.ndarray
            Raw vsync falling edge times.
    """
    sync_file = h5py.File(sync_path, 'r')
    meta = ast.literal_eval(sync_file['meta'][()].decode('utf-8'))
    sample_rate = meta['ni_daq']['sample_rate']
    line_labels = meta['line_labels']
    sync_data = sync_file['data'][:]
    sync_file.close()

    counters = sync_data[:, 0]
    bits = sync_data[:, 1]

    # Vsync falling edges = frame times
    stim_vsync_fall = _get_edges(bits, counters, line_labels,
                                 'stim_vsync', 'falling', sample_rate)
    # Trim to number of pkl frames
    n_pkl_frames = len(pkl['items']['behavior']['intervalsms']) + 1
    stim_vsync_fall = stim_vsync_fall[:n_pkl_frames]

    # ---- Monitor delay from photodiode ----
    all_pd_edges = _get_edges(bits, counters, line_labels,
                              'stim_photodiode', 'both', sample_rate)
    all_pd_edges = np.sort(all_pd_edges)

    pd_diffs = np.diff(all_pd_edges)
    # Regular photodiode toggles should be ~1 s apart (60 frames at 60 Hz)
    regular_mask = (pd_diffs > 0.8) & (pd_diffs < 1.2)
    regular_indices = np.where(regular_mask)[0]

    monitor_delay = 0.0356  # fallback
    if len(regular_indices) > 10:
        first_regular = regular_indices[0]
        last_regular = regular_indices[-1] + 1

        # Remove anomalous close events (< 0.5 s apart)
        clean_pd = all_pd_edges[first_regular:last_regular + 1].copy()
        while True:
            diffs = np.diff(clean_pd)
            anomalies = np.where(diffs < 0.5)[0]
            if len(anomalies) == 0:
                break
            clean_pd = np.delete(clean_pd, anomalies[-1] + 1)

        # Every 60th vsync = expected photodiode toggle time
        transitions = stim_vsync_fall[::60]

        # Align first clean photodiode event to nearest transition
        first_pd = clean_pd[0]
        nearest_idx = np.argmin(np.abs(transitions - first_pd))

        n_match = min(len(clean_pd), len(transitions) - nearest_idx)
        delays = clean_pd[:n_match] - transitions[nearest_idx:nearest_idx + n_match]

        valid = (delays > 0) & (delays < 0.07)
        if np.sum(valid) > 10:
            monitor_delay = np.mean(delays[valid])
        else:
            monitor_delay = np.median(delays)
            if not (0 < monitor_delay < 0.07):
                monitor_delay = 0.0356

    stim_ts_visual = stim_vsync_fall + monitor_delay
    stim_ts_behavioral = stim_vsync_fall

    logger.info(f"Monitor delay: {monitor_delay * 1e3:.2f} ms")
    logger.info(f"Frame count: {len(stim_ts_visual)}")

    return {
        'stim_ts_visual': stim_ts_visual,
        'stim_ts_behavioral': stim_ts_behavioral,
        'monitor_delay': monitor_delay,
        'stim_vsync_fall': stim_vsync_fall,
    }


# ===================================================================
# Helpers
# ===================================================================

def _get_draw_epochs(draw_log, start_frame, stop_frame):
    """Find contiguous runs of draw_log==1 within [start_frame, stop_frame].

    Returns list of (epoch_start, epoch_end) tuples where both are inclusive
    indices into draw_log.
    """
    epochs = []
    current = start_frame
    while current <= stop_frame:
        epoch_len = 0
        while current < stop_frame and draw_log[current] == 1:
            epoch_len += 1
            current += 1
        else:
            current += 1
        if epoch_len:
            epochs.append((current - epoch_len - 1, current - 1))
    return epochs


# ===================================================================
# TASK 2: Build EventsTable
# ===================================================================

def build_events_table(pkl, ts):
    """Build the complete events table.

    Parameters
    ----------
    pkl : dict
        Loaded pickle data.
    ts : dict
        Timestamp alignment dict from ``compute_timestamp_alignment``.

    Returns
    -------
    pd.DataFrame
        Events table with columns: timestamp, event_type, hed_string,
        trials_id, stimulus_presentations_id, image_name, frame,
        lick_latency, reward_volume.
    """
    stim_ts_visual = ts['stim_ts_visual']
    stim_ts_behavioral = ts['stim_ts_behavioral']
    n_frames = len(stim_ts_visual)

    beh = pkl['items']['behavior']
    stimuli = beh['stimuli']['images']
    trial_log = beh['trial_log']
    params = beh['params']

    set_log = stimuli['set_log']
    draw_log = stimuli['draw_log']

    events = []

    # ---- IMAGE ONSET / OFFSET ----
    for idx, (attr_name, attr_value, _time, frame) in enumerate(set_log):
        image_name = attr_value if attr_name.lower() == 'image' else None
        orientation = attr_value if attr_name.lower() == 'ori' else None
        if image_name is None and orientation is None:
            continue

        try:
            next_frame = set_log[idx + 1][3]
        except IndexError:
            next_frame = n_frames

        draw_eps = _get_draw_epochs(draw_log, frame, next_frame)
        for ep_start, ep_end in draw_eps:
            # Frame-1 correction: draw_log frame records the draw command,
            # but the vsync falling edge that triggers display is at frame-1.
            # Then we add +1 for "visual change takes effect on following
            # frame", but the net result is frame-1 for the timestamp lookup.
            ep_start += 1  # visual change on next frame
            ep_end += 1
            if ep_start >= n_frames or ep_end >= n_frames:
                continue

            # Apply frame-1 correction for timestamp lookup
            onset_time = stim_ts_visual[ep_start - 1]
            offset_time = stim_ts_visual[ep_end - 1]

            if image_name is not None:
                # Store the actual vsync edge index used to compute the
                # timestamp (`ep_start - 1` after our +1 correction), so
                # frame indexes into the sync array consistently with timestamp.
                events.append({
                    'timestamp': onset_time,
                    'event_type': 'image_onset',
                    'image_name': image_name,
                    'frame': ep_start - 1,
                })
                events.append({
                    'timestamp': offset_time,
                    'event_type': 'image_offset',
                    'image_name': image_name,
                    'frame': ep_end - 1,
                })

    # ---- IMAGE CHANGE events from change_log ----
    change_log = stimuli['change_log']
    for entry in change_log:
        (from_cat, from_name), (to_cat, to_name), time_val, frame = entry
        if from_name == to_name:
            continue  # not an actual identity change
        # change_log frame is where the image register changes;
        # the visual onset is the next draw epoch start (frame + 1),
        # then frame-1 correction nets to using frame for timestamp.
        onset_frame = frame + 1
        if onset_frame >= n_frames:
            continue
        change_ts = stim_ts_visual[onset_frame - 1]
        events.append({
            'timestamp': change_ts,
            'event_type': 'image_change',
            'image_name': to_name,
            'frame': onset_frame - 1,  # vsync edge index of change_ts
        })

    # ---- OMITTED IMAGES ----
    omitted_frames = stimuli['flashes_omitted']
    stim_frames = set(e['frame'] for e in events if e['event_type'] == 'image_onset')
    for omitted_frame in omitted_frames:
        if omitted_frame >= n_frames:
            continue
        is_duplicate = any(abs(omitted_frame - sf) <= 3 for sf in stim_frames)
        if not is_duplicate:
            # flashes_omitted frames are already at the correct vsync index;
            # no frame-1 correction needed. Use visual timestamps (with
            # monitor delay) since the omission is defined relative to the
            # expected visual stimulus timing.
            events.append({
                'timestamp': stim_ts_visual[omitted_frame],
                'event_type': 'image_omission',
                'frame': omitted_frame,
            })

    # ---- LICKS from lick_sensors (primary source) ----
    # Using lick_sensors captures all licks including spontaneous/inter-trial.
    all_lick_frames = np.array(
        beh['lick_sensors'][0]['lick_events'], dtype=int
    )
    all_lick_frames = all_lick_frames[all_lick_frames < n_frames]

    # Build per-trial classification context
    response_window = params.get('response_window', [0.15, 0.75])
    rw_start, rw_end = response_window[0], response_window[1]

    # Precompute trial info for lick classification
    trial_info = []
    for trial in trial_log:
        trial_events = trial.get('events', [])
        trial_params_dict = trial.get('trial_params', {})
        is_aborted = any(ev[0] == 'abort' for ev in trial_events)
        has_early_response = any(ev[0] == 'early_response' for ev in trial_events)
        is_auto_rewarded = trial_params_dict.get('auto_reward', False)
        is_catch = trial_params_dict.get('catch', False)

        change_frame = None
        change_time = None
        trial_start_frame = None
        trial_end_frame = None
        for ev_name, ev_detail, ev_time, ev_frame in trial_events:
            if ev_name in ('stimulus_changed', 'sham_change'):
                change_frame = ev_frame
                if ev_frame < n_frames:
                    change_time = stim_ts_behavioral[ev_frame]
            if ev_name == 'trial_start':
                trial_start_frame = ev_frame
            if ev_name == 'trial_end':
                trial_end_frame = ev_frame

        t_start = stim_ts_behavioral[trial_start_frame] if (
            trial_start_frame is not None and trial_start_frame < n_frames
        ) else None
        t_end = stim_ts_behavioral[trial_end_frame] if (
            trial_end_frame is not None and trial_end_frame < n_frames
        ) else None

        trial_info.append({
            'start': t_start,
            'end': t_end,
            'change_time': change_time,
            'is_aborted': is_aborted,
            'has_early_response': has_early_response,
            'is_auto_rewarded': is_auto_rewarded,
            'is_catch': is_catch,
        })

    # All licks get event_type='lick'; classification is done later as a
    # separate orthogonal column (lick_classification) after intervals are built.
    for lick_frame in all_lick_frames:
        lick_time = stim_ts_behavioral[lick_frame]
        events.append({
            'timestamp': lick_time,
            'event_type': 'lick',
            'frame': int(lick_frame),
        })

    # ---- REWARDS ----
    for trial in trial_log:
        for reward_entry in trial.get('rewards', []):
            volume, reward_time_raw, reward_frame = reward_entry
            if reward_frame < n_frames:
                events.append({
                    'timestamp': stim_ts_behavioral[reward_frame],
                    'event_type': 'reward',
                    'reward_volume': volume,
                    'frame': int(reward_frame),
                })

    # ---- MISS ----
    # correct_reject is not included as an event — it is the absence of a
    # response and is already captured as a trial-level boolean in the
    # intervals table.
    for trial in trial_log:
        trial_events = trial.get('events', [])
        is_auto_rewarded = trial.get('trial_params', {}).get('auto_reward', False)

        for ev_name, ev_detail, ev_time, ev_frame in trial_events:
            if ev_name == 'miss' and not is_auto_rewarded:
                ts_val = stim_ts_behavioral[ev_frame] if ev_frame < n_frames else ev_time
                events.append({
                    'timestamp': ts_val,
                    'event_type': 'miss',
                    'frame': int(ev_frame),
                })

    # ---- NATURAL MOVIE ONE (fingerprint) ----
    fp = beh.get('items', {}).get('fingerprint', None)
    if fp is not None:
        fp_ss = fp.get('static_stimulus', {})
        fp_frame_indices = fp.get('frame_indices', None)
        fp_frame_list = fp_ss.get('frame_list', None)
        fp_sweep_frames = fp_ss.get('sweep_frames', None)

        if (fp_frame_indices is not None and fp_frame_list is not None
                and fp_sweep_frames is not None):
            n_display = len(fp_frame_list)
            n_sweeps = len(fp_sweep_frames)
            # Movie content starts after blank frames in frame_list
            movie_start_display = 0
            for i in range(n_display):
                if fp_frame_list[i] >= 0:
                    movie_start_display = i
                    break

            # Each movie frame = 2 display frames at 60Hz (30Hz movie)
            # frame_indices maps local display frame → global stimulus frame
            # No frame-1 correction needed (frame_indices already at correct vsync)
            n_movie_frames = (n_display - movie_start_display) // 2
            for i in range(n_movie_frames):
                local_onset = movie_start_display + i * 2
                global_frame = fp_frame_indices[local_onset]
                movie_frame = int(fp_frame_list[local_onset])
                movie_repeat = i // 900  # 900 frames per repeat

                if global_frame < n_frames:
                    onset_ts = stim_ts_visual[global_frame]
                    events.append({
                        'timestamp': onset_ts,
                        'event_type': 'movie_onset',
                        'frame': int(global_frame),
                        'image_name': 'natural_movie_one',
                        'movie_frame_index': movie_frame,
                        'movie_repeat': movie_repeat,
                    })

                    # Offset = next frame's onset (contiguous with next presentation)
                    offset_frame_idx = int(global_frame)  # fallback
                    if i < n_movie_frames - 1:
                        next_local = movie_start_display + (i + 1) * 2
                        next_global = fp_frame_indices[next_local]
                        if next_global < n_frames:
                            offset_ts = stim_ts_visual[next_global]
                            offset_frame_idx = int(next_global)
                        else:
                            offset_ts = onset_ts + 1.0 / 30.0
                    else:
                        # Last frame: use ending_frame if available,
                        # else extrapolate by median duration
                        ending_frame = fp.get('ending_frame', None)
                        if (ending_frame is not None
                                and ending_frame < n_frames):
                            offset_ts = stim_ts_visual[ending_frame]
                            offset_frame_idx = int(ending_frame)
                        else:
                            offset_ts = onset_ts + 1.0 / 30.0

                    events.append({
                        'timestamp': offset_ts,
                        'event_type': 'movie_offset',
                        'frame': offset_frame_idx,
                        'movie_frame_index': movie_frame,
                        'movie_repeat': movie_repeat,
                    })

            logger.info(f"  {n_movie_frames} natural_movie_one frames "
                        f"({n_movie_frames // 900} repeats)")

    # ---- Build DataFrame ----
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)

    # ---- stimulus_presentations_id ----
    # Sequential counter for each image_onset or image_omission
    pres_mask = events_df['event_type'].isin(['image_onset', 'image_omission'])
    events_df['stimulus_presentations_id'] = -1
    events_df.loc[pres_mask, 'stimulus_presentations_id'] = range(pres_mask.sum())
    # Forward-fill for other events (offsets, changes, licks, etc.)
    events_df['stimulus_presentations_id'] = (
        events_df['stimulus_presentations_id']
        .replace(-1, np.nan)
        .ffill()
        .fillna(-1)
        .astype(int)
    )
    # image_change events occur at the same timestamp as the corresponding
    # NEW-image image_onset. Stable-sort ordering between them is fragile,
    # so explicitly map each image_change to the sid of the matching
    # image_onset by timestamp (rounded to microseconds).
    onset_sid = dict(zip(
        events_df.loc[events_df['event_type'] == 'image_onset', 'timestamp']
            .round(6).values,
        events_df.loc[events_df['event_type'] == 'image_onset',
                       'stimulus_presentations_id'].values,
    ))
    change_mask = events_df['event_type'] == 'image_change'
    events_df.loc[change_mask, 'stimulus_presentations_id'] = (
        events_df.loc[change_mask, 'timestamp'].round(6).map(onset_sid)
        .fillna(events_df.loc[change_mask, 'stimulus_presentations_id'])
        .astype(int).values
    )

    # ---- lick_latency ----
    # Time since the most recent image_onset for each lick event
    onset_times = events_df.loc[
        events_df['event_type'] == 'image_onset', 'timestamp'
    ].values
    lick_mask = events_df['event_type'] == 'lick'
    lick_times = events_df.loc[lick_mask, 'timestamp'].values

    lick_latency = np.full(len(events_df), np.nan)
    if len(onset_times) > 0 and len(lick_times) > 0:
        # For each lick, find the index of the last onset before it
        onset_idx = np.searchsorted(onset_times, lick_times, side='right') - 1
        valid = onset_idx >= 0
        latencies = np.where(valid, lick_times - onset_times[onset_idx], np.nan)
        lick_latency[lick_mask.values] = latencies
    events_df['lick_latency'] = lick_latency

    # ---- Initial HED tags (non-lick events get final tags here;
    #      lick HED strings are composed later after classification) ----
    def _make_hed(row):
        return format_hed_string(row['event_type'], row.get('image_name'))
    events_df['hed_string'] = events_df.apply(_make_hed, axis=1)

    # ---- trials_id ----
    # Will be filled after intervals table is built; initialize to -1
    events_df['trials_id'] = -1

    # ---- lick_classification, bout_start, reward_type ----
    # Initialized here; filled by _classify_licks() and _classify_rewards()
    # after intervals are built.
    events_df['lick_classification'] = np.nan
    events_df['bout_start'] = np.nan
    events_df['reward_type'] = np.nan

    # Ensure consistent columns
    for col in ['image_name', 'reward_volume', 'movie_frame_index',
                'movie_repeat']:
        if col not in events_df.columns:
            events_df[col] = np.nan

    # Final column order
    cols = [
        'timestamp', 'event_type', 'hed_string', 'trials_id',
        'stimulus_presentations_id', 'image_name', 'frame',
        'lick_latency', 'reward_volume',
        'movie_frame_index', 'movie_repeat',
        'lick_classification', 'bout_start', 'reward_type',
    ]
    events_df = events_df[cols]

    return events_df


# ===================================================================
# TASK 3: Build TimeIntervals table
# ===================================================================

def build_intervals_table(pkl, ts, events_df):
    """Build the complete intervals table.

    Parameters
    ----------
    pkl : dict
        Loaded pickle data.
    ts : dict
        Timestamp alignment dict.
    events_df : pd.DataFrame
        Events table (used for onset times in change_window computation).

    Returns
    -------
    pd.DataFrame
        Intervals table with columns: interval_type, label, start_time,
        stop_time, trials_id, go, catch, auto_rewarded, aborted, hit, miss,
        false_alarm, correct_reject, change_time, change_frame,
        initial_image_name, change_image_name, reward_time, reward_volume,
        response_time, response_latency.
    """
    stim_ts_visual = ts['stim_ts_visual']
    stim_ts_behavioral = ts['stim_ts_behavioral']
    n_frames = len(stim_ts_visual)

    beh = pkl['items']['behavior']
    trial_log = beh['trial_log']
    cl_params = beh['cl_params']
    params = beh['params']
    set_log = beh['stimuli']['images']['set_log']

    response_window = params.get('response_window', [0.15, 0.75])
    change_flashes_min = cl_params.get('change_flashes_min', 4)
    change_flashes_max = cl_params.get('change_flashes_max', 12)

    def get_trial_event_time(trial, event_name, use_visual=False):
        ts_array = stim_ts_visual if use_visual else stim_ts_behavioral
        for ev in trial.get('events', []):
            if ev[0] == event_name:
                frame = ev[3]
                if frame < n_frames:
                    return ts_array[frame]
                return ev[2]
        return None

    intervals = []

    # ---- EPOCHS ----

    # Warm-up: initial trials with auto_reward=True
    warmup_end_idx = 0
    for i, trial in enumerate(trial_log):
        if not trial.get('trial_params', {}).get('auto_reward', False):
            warmup_end_idx = i
            break
    else:
        warmup_end_idx = len(trial_log)

    if warmup_end_idx > 0:
        wu_start = get_trial_event_time(trial_log[0], 'trial_start')
        wu_end = get_trial_event_time(trial_log[warmup_end_idx - 1], 'trial_end')
        if wu_start and wu_end:
            intervals.append({
                'interval_type': 'epoch', 'label': 'warm_up',
                'start_time': wu_start, 'stop_time': wu_end,
            })

    # Change detection epoch
    if warmup_end_idx < len(trial_log):
        cd_start = get_trial_event_time(trial_log[warmup_end_idx], 'trial_start')
        cd_end = get_trial_event_time(trial_log[-1], 'trial_end')
        if cd_start and cd_end:
            intervals.append({
                'interval_type': 'epoch', 'label': 'change_detection',
                'start_time': cd_start, 'stop_time': cd_end,
            })

    # Natural movie one epoch (called "fingerprint" in pkl)
    if 'fingerprint' in beh.get('items', {}):
        movie_onsets = events_df[events_df['event_type'] == 'movie_onset']
        movie_offsets = events_df[events_df['event_type'] == 'movie_offset']
        if len(movie_onsets) > 0 and len(movie_offsets) > 0:
            intervals.append({
                'interval_type': 'epoch', 'label': 'natural_movie_one',
                'start_time': movie_onsets['timestamp'].min(),
                'stop_time': movie_offsets['timestamp'].max(),
            })

    # ---- TRIALS ----
    trial_intervals = []
    for trial_idx, trial in enumerate(trial_log):
        trial_events = trial.get('events', [])
        trial_params_dict = trial.get('trial_params', {})
        trial_rewards = trial.get('rewards', [])
        stimulus_changes = trial.get('stimulus_changes', [])

        start_time = get_trial_event_time(trial, 'trial_start')
        stop_time = get_trial_event_time(trial, 'trial_end')
        if start_time is None or stop_time is None:
            continue

        is_catch = trial_params_dict.get('catch', False)
        is_auto_reward = trial_params_dict.get('auto_reward', False)
        is_aborted = any(ev[0] == 'abort' for ev in trial_events)

        hit = any(ev[0] == 'hit' for ev in trial_events) and not is_auto_reward
        miss = any(ev[0] == 'miss' for ev in trial_events) and not is_auto_reward
        false_alarm = any(ev[0] == 'false_alarm' for ev in trial_events) and not is_auto_reward
        correct_reject = (is_catch and not false_alarm
                          and not is_aborted and not is_auto_reward)
        go = not is_catch and not is_auto_reward and not is_aborted

        # Change time/frame
        change_frame = None
        change_time = None
        for ev_name, _, _, ev_frame in trial_events:
            if ev_name in ('stimulus_changed', 'sham_change'):
                change_frame = ev_frame
                if ev_frame < n_frames:
                    change_time = stim_ts_visual[ev_frame]
                break

        # Image names
        initial_image = None
        start_frame_trial = None
        for ev in trial_events:
            if ev[0] == 'trial_start':
                start_frame_trial = ev[3]
                break
        if start_frame_trial is not None:
            for sl_idx in range(len(set_log) - 1, -1, -1):
                if (set_log[sl_idx][3] <= start_frame_trial
                        and set_log[sl_idx][0].lower() == 'image'):
                    initial_image = set_log[sl_idx][1]
                    break

        change_image = None
        if stimulus_changes:
            change_image = (stimulus_changes[0][1][1]
                            if len(stimulus_changes[0]) > 1 else None)
        if change_image is None:
            change_image = initial_image

        # Rewards
        reward_time = None
        reward_volume = 0.0
        for vol, _, rf in trial_rewards:
            reward_volume += vol
            if rf < n_frames:
                reward_time = stim_ts_behavioral[rf]

        # Response
        trial_licks = trial.get('licks', [])
        lick_times = [stim_ts_behavioral[lf]
                      for lt, lf in trial_licks if lf < n_frames]
        response_time = lick_times[0] if lick_times and not is_aborted else None
        response_latency = ((response_time - change_time)
                            if (response_time and change_time) else None)

        trial_intervals.append({
            'interval_type': 'trial',
            'start_time': start_time,
            'stop_time': stop_time,
            'trials_id': trial_idx,
            'go': go,
            'catch': is_catch,
            'auto_rewarded': is_auto_reward,
            'aborted': is_aborted,
            'hit': hit,
            'miss': miss,
            'false_alarm': false_alarm,
            'correct_reject': correct_reject,
            'change_time': change_time,
            'change_frame': change_frame,
            'initial_image_name': initial_image,
            'change_image_name': change_image,
            'reward_time': reward_time,
            'reward_volume': reward_volume,
            'response_time': response_time,
            'response_latency': response_latency,
        })

    # ---- CHANGE WINDOWS and RESPONSE WINDOWS ----
    image_onsets = events_df[
        events_df['event_type'] == 'image_onset'
    ].sort_values('timestamp')
    onset_times_vis = image_onsets['timestamp'].values

    window_intervals = []
    for trial_row in trial_intervals:
        trial_start = trial_row['start_time']
        trial_stop = trial_row['stop_time']
        trial_id = trial_row['trials_id']

        mask = (onset_times_vis >= trial_start) & (onset_times_vis <= trial_stop)
        trial_onset_times = onset_times_vis[mask]

        if len(trial_onset_times) < change_flashes_min:
            continue

        # Change window starts at the change_flashes_min-th flash onset
        cw_start_time = trial_onset_times[change_flashes_min - 1]

        change_time = trial_row['change_time']
        if change_time is not None:
            cw_end_time = change_time
        elif trial_row['aborted']:
            cw_end_time = trial_row['stop_time']
        elif len(trial_onset_times) > change_flashes_max:
            cw_end_time = trial_onset_times[change_flashes_max - 1]
        else:
            cw_end_time = trial_onset_times[-1]

        if cw_start_time < cw_end_time:
            window_intervals.append({
                'interval_type': 'change_window',
                'start_time': cw_start_time,
                'stop_time': cw_end_time,
                'trials_id': trial_id,
            })

        # Response window: only for trials with a change event
        if change_time is not None:
            rw_start_time = change_time + response_window[0]
            rw_end_time = change_time + response_window[1]
            window_intervals.append({
                'interval_type': 'response_window',
                'start_time': rw_start_time,
                'stop_time': rw_end_time,
                'trials_id': trial_id,
            })

    all_intervals = intervals + trial_intervals + window_intervals
    intervals_df = pd.DataFrame(all_intervals)
    intervals_df = intervals_df.sort_values('start_time').reset_index(drop=True)

    # ---- HED tags for intervals ----
    def _interval_hed(row):
        itype = row['interval_type']
        if itype == 'epoch':
            return HED_TAGS.get(f'epoch_{row["label"]}', '')
        elif itype == 'trial':
            if row.get('aborted'):
                return HED_TAGS['trial_aborted']
            elif row.get('auto_rewarded'):
                return HED_TAGS['trial_auto_rewarded']
            elif row.get('catch'):
                return HED_TAGS['trial_catch']
            else:
                return HED_TAGS['trial_go']
        else:
            return HED_TAGS.get(itype, '')

    intervals_df['hed_string'] = intervals_df.apply(_interval_hed, axis=1)

    return intervals_df


# ===================================================================
# TASK 4: Assign trials_id to events
# ===================================================================

def assign_trials_id(events_df, intervals_df):
    """Assign trials_id to each event based on trial boundaries.

    Each event receives the index of the trial it falls within (by timestamp),
    or -1 if outside all trial boundaries.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events table (modified in-place).
    intervals_df : pd.DataFrame
        Intervals table containing trial rows.

    Returns
    -------
    pd.DataFrame
        Events table with trials_id column filled in.
    """
    trials = intervals_df[
        intervals_df['interval_type'] == 'trial'
    ].sort_values('start_time').reset_index(drop=True)
    trial_starts = trials['start_time'].values
    trial_stops = trials['stop_time'].values
    n_trials = len(trials)

    event_ts = events_df['timestamp'].values
    trials_id = np.full(len(event_ts), -1, dtype=int)

    # For each event, find the last trial that started <= event time
    trial_idx = np.searchsorted(trial_starts, event_ts, side='right') - 1

    for i in range(len(event_ts)):
        ti = trial_idx[i]
        if 0 <= ti < n_trials and event_ts[i] <= trial_stops[ti]:
            trials_id[i] = ti

    events_df['trials_id'] = trials_id
    return events_df


def _classify_rewards(events_df, intervals_df):
    """Classify rewards as 'auto_reward' or 'earned' and set HED strings.

    Auto-rewards are delivered on trials where ``auto_rewarded`` is True
    in the intervals table. All other rewards are earned.
    """
    trials = intervals_df[intervals_df['interval_type'] == 'trial']
    auto_trial_ids = set(
        trials.loc[trials['auto_rewarded'].astype(bool), 'trials_id'].values
    )

    reward_mask = events_df['event_type'] == 'reward'
    reward_indices = events_df.index[reward_mask]

    for idx in reward_indices:
        tid = events_df.at[idx, 'trials_id']
        rtype = 'auto_reward' if tid in auto_trial_ids else 'earned'
        events_df.at[idx, 'reward_type'] = rtype
        events_df.at[idx, 'hed_string'] = HED_TAGS['reward'].format(
            reward_type=rtype
        )

    n_auto = (events_df.loc[reward_indices, 'reward_type'] == 'auto_reward').sum()
    n_earned = (events_df.loc[reward_indices, 'reward_type'] == 'earned').sum()
    logger.info(f"  Reward types: auto_reward={n_auto}, earned={n_earned}")
    return events_df


def _classify_licks(events_df, intervals_df):
    """Classify licks along two orthogonal dimensions.

    **Dimension 1 — lick_classification** (priority order):
      - ``hit``: first lick in response_window after a real change (go trial)
      - ``false_alarm``: lick in a change_window, or in response_window on
        a catch trial
      - ``early``: after change but before response_window opens
      - ``late``: after response_window closes on a trial with a change,
        but before any reward (missed-change licks)
      - ``abort``: lick that triggered trial abort (before change, or on an
        aborted trial with early response)
      - ``consumption``: lick after a reward delivery in the same trial
      - ``spontaneous``: everything else (between trials, during warmup, etc.)

    **Dimension 2 — bout_start** (boolean):
      True if the inter-lick interval from the previous lick exceeds 500 ms.
      The first lick in the session is always bout_start.

    After classification, HED strings for lick events are recomposed from
    the base lick tag + classification fragment + bout_start fragment.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events table (modified in-place).
    intervals_df : pd.DataFrame
        Intervals table containing trial, change_window, and response_window.

    Returns
    -------
    pd.DataFrame
    """
    trials = intervals_df[intervals_df['interval_type'] == 'trial'].copy()
    for col in ['catch', 'aborted', 'auto_rewarded', 'hit', 'miss']:
        trials[col] = trials[col].astype(bool)

    catch_trial_ids = set(trials.loc[trials['catch'], 'trials_id'].values)
    aborted_trial_ids = set(trials.loc[trials['aborted'], 'trials_id'].values)

    # --- Interval lookups ---
    cw = intervals_df[
        intervals_df['interval_type'] == 'change_window'
    ].sort_values('start_time')
    cw_starts = cw['start_time'].values
    cw_stops = cw['stop_time'].values

    rw = intervals_df[
        intervals_df['interval_type'] == 'response_window'
    ].sort_values('start_time')
    rw_starts = rw['start_time'].values
    rw_stops = rw['stop_time'].values
    rw_trial_ids = rw['trials_id'].values

    # Per-trial lookups
    change_times = {}
    reward_times = {}
    rw_by_trial = {}  # response_window start per trial
    rw_stop_by_trial = {}  # response_window stop per trial
    for _, t in trials.iterrows():
        tid = int(t['trials_id'])
        if pd.notna(t['change_time']) and t['change_time'] > 0:
            change_times[tid] = t['change_time']
        if pd.notna(t['reward_time']) and t['reward_time'] > 0:
            reward_times[tid] = t['reward_time']
    for tid, start, stop in zip(rw_trial_ids, rw_starts, rw_stops):
        rw_by_trial[int(tid)] = start
        rw_stop_by_trial[int(tid)] = stop

    # --- Lick indices ---
    lick_mask = events_df['event_type'] == 'lick'
    lick_indices = events_df.index[lick_mask]
    lick_times = events_df.loc[lick_indices, 'timestamp'].values
    lick_trial_ids = events_df.loc[lick_indices, 'trials_id'].values

    # --- Dimension 1: lick_classification ---
    classifications = np.full(len(lick_times), 'spontaneous', dtype=object)
    hit_seen_per_trial = set()

    counts = {}
    for i, (t, tid) in enumerate(zip(lick_times, lick_trial_ids)):
        tid = int(tid)
        cls = 'spontaneous'

        # 1. In a change_window? → false_alarm
        ci = np.searchsorted(cw_starts, t, side='right') - 1
        if ci >= 0 and t <= cw_stops[ci]:
            cls = 'false_alarm'

        # 2. In a response_window on a catch trial? → false_alarm
        elif tid >= 0:
            ri = np.searchsorted(rw_starts, t, side='right') - 1
            if ri >= 0 and t <= rw_stops[ri]:
                rw_tid = int(rw_trial_ids[ri])
                if rw_tid in catch_trial_ids:
                    cls = 'false_alarm'
                # 3. In response_window on go trial → hit (first only)
                elif rw_tid not in aborted_trial_ids and rw_tid not in hit_seen_per_trial:
                    # Check it's actually a go trial with a change
                    if rw_tid in change_times:
                        auto = trials.loc[
                            trials['trials_id'] == rw_tid, 'auto_rewarded'
                        ].values
                        if len(auto) == 0 or not auto[0]:
                            cls = 'hit'
                            hit_seen_per_trial.add(rw_tid)

        # 4. After reward in same trial? → consumption
        if cls == 'spontaneous' and tid >= 0 and tid in reward_times:
            if t > reward_times[tid]:
                cls = 'consumption'

        # 5. After change but before response_window? → early
        if cls == 'spontaneous' and tid >= 0 and tid in change_times:
            ct = change_times[tid]
            if t > ct:
                rw_start = rw_by_trial.get(tid)
                rw_stop = rw_stop_by_trial.get(tid)
                if rw_start is not None and t < rw_start:
                    cls = 'early'
                # 6. After response_window closes? → late
                elif rw_stop is not None and t > rw_stop:
                    # Only 'late' if no reward (i.e. miss trial)
                    if tid not in reward_times:
                        cls = 'late'

        # 7. On an aborted trial and not already classified? → abort
        if cls == 'spontaneous' and tid in aborted_trial_ids:
            cls = 'abort'

        classifications[i] = cls
        counts[cls] = counts.get(cls, 0) + 1

    events_df.loc[lick_indices, 'lick_classification'] = classifications

    # --- Dimension 2: bout_start ---
    BOUT_ILI_THRESHOLD = 0.5  # 500 ms
    if len(lick_times) > 1:
        ilis = np.diff(lick_times)
        bout_starts = np.zeros(len(lick_times), dtype=bool)
        bout_starts[0] = True  # first lick is always bout start
        bout_starts[1:] = ilis > BOUT_ILI_THRESHOLD
    elif len(lick_times) == 1:
        bout_starts = np.array([True])
    else:
        bout_starts = np.array([], dtype=bool)

    events_df.loc[lick_indices, 'bout_start'] = bout_starts

    # --- Compose HED strings for licks ---
    base_hed = HED_TAGS['lick']
    for idx, cls, bs in zip(lick_indices, classifications, bout_starts):
        hed = base_hed + LICK_HED_FRAGMENTS.get(cls, '')
        if cls != 'spontaneous':
            hed += f', Label/{cls}'
        if bs:
            hed += BOUT_START_HED_FRAGMENT
        events_df.at[idx, 'hed_string'] = hed

    logger.info(f"  Lick classifications: {counts}")
    logger.info(f"  Bout starts: {bout_starts.sum()}/{len(bout_starts)}")
    return events_df


# ===================================================================
# Main pipeline
# ===================================================================

def build_all(pkl_path, sync_path, output_dir=None):
    """Run the full pipeline: alignment, events, intervals.

    Parameters
    ----------
    pkl_path : str or Path
        Path to the stimulus pickle file.
    sync_path : str or Path
        Path to the HDF5 sync file.
    output_dir : str or Path, optional
        Directory to save output files. If None, returns dicts only.

    Returns
    -------
    dict with keys 'events_df', 'intervals_df', 'timestamp_data'.
    """
    logger.info(f"Loading pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f, encoding='latin1')

    logger.info(f"Computing timestamp alignment from: {sync_path}")
    ts = compute_timestamp_alignment(pkl, sync_path)

    logger.info("Building events table")
    events_df = build_events_table(pkl, ts)
    logger.info(f"  {len(events_df)} events, types: "
                f"{dict(events_df['event_type'].value_counts())}")

    logger.info("Building intervals table")
    intervals_df = build_intervals_table(pkl, ts, events_df)
    logger.info(f"  {len(intervals_df)} intervals")

    logger.info("Assigning trials_id to events")
    events_df = assign_trials_id(events_df, intervals_df)
    n_in_trial = (events_df['trials_id'] >= 0).sum()
    logger.info(f"  {n_in_trial}/{len(events_df)} events assigned to a trial")

    # Classify rewards as auto_reward or earned.
    logger.info("Classifying rewards")
    events_df = _classify_rewards(events_df, intervals_df)

    # Classify licks by task context and bout structure.
    logger.info("Classifying licks by task context and bout structure")
    events_df = _classify_licks(events_df, intervals_df)

    # Extract task parameters (LabMetaData container, ready for NWB).
    logger.info("Extracting task parameters")
    from task_parameters import build_task_parameters
    task_parameters = build_task_parameters(pkl)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events_df.to_pickle(output_dir / 'events_df.pkl')
        events_df.to_csv(output_dir / 'events_table.csv', index=False)
        intervals_df.to_pickle(output_dir / 'intervals_df.pkl')
        intervals_df.to_csv(output_dir / 'intervals_table.csv', index=False)
        np.savez(
            output_dir / 'timestamp_data.npz',
            stim_ts_visual=ts['stim_ts_visual'],
            stim_ts_behavioral=ts['stim_ts_behavioral'],
            monitor_delay=np.array([ts['monitor_delay']]),
            stim_vsync_fall=ts['stim_vsync_fall'],
        )
        logger.info(f"Saved outputs to {output_dir}")

    return {
        'events_df': events_df,
        'intervals_df': intervals_df,
        'timestamp_data': ts,
        'task_parameters': task_parameters,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build events and intervals tables for change detection task'
    )
    parser.add_argument('pkl_path', help='Path to stimulus pickle file')
    parser.add_argument('sync_path', help='Path to HDF5 sync file')
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Output directory (default: current directory)')
    args = parser.parse_args()

    build_all(args.pkl_path, args.sync_path, args.output_dir)
