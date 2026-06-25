"""SDK-compatibility shim for the new HED-tagged NWB files.

`to_sdk_format(nwb)` returns `(stimulus_presentations, trials)` DataFrames
matching the shape that the AllenSDK's `BehaviorOphysExperiment` returns when
loading the old-format NWB. Drop-in compatible with existing analysis code
that consumes those two tables.

What the SDK actually does (verified against the source code in
`AllenSDK/allensdk/brain_observatory/behavior/data_objects/stimuli/presentations.py`):

- For `stimulus_presentations`: concatenates every TimeIntervals entry whose
  name ends with ``_presentations`` (in the reference NWB those are
  ``Natural_Images_...`_presentations``, ``natural_movie_one_presentations``,
  ``spontaneous_presentations``), drops the ``tags`` / ``timeseries`` /
  ``*_index`` columns, casts ints + booleans, sorts by start_time, resets the
  index, and renames ``stop_time`` → ``end_time``. The heavy lifting
  (`stimulus_block`, `image_index`, `start_frame`/`end_frame`, `is_image_novel`,
  `flashes_since_change`, `is_sham_change`, `active`, etc.) is done at write
  time by the SDK; the loader just hands those columns back.
- For `trials`: returned verbatim as stored.

Verified against the SDK loader on session 1050231786
-----------------------------------------------------
trials: all 21 columns match exactly.

stimulus_presentations: 14 of 19 columns match exactly. Remaining
deliberate gaps:

- ``start_frame`` / ``end_frame``: we don't store vsync frame indices in
  the new NWB. Filled with -99 (the SDK's "not applicable" sentinel).
  Could be reconstructed by re-loading the sync file at convert time.
- ``image_index``: the SDK uses a stable per-image-set ordering it pulls
  from internal asset metadata. We have no access to that mapping, so
  we use alphabetic ordering instead (im061→0, im062→1, ...). Values
  differ from the SDK; per-image grouping is still valid.
- ``is_image_novel``: SDK marks each presentation True/False based on
  whether the image is in the "novel" vs "familiar" set for this animal
  — a project-level fact we can't derive from the session alone. We
  mark first-appearance-per-session as True; this disagrees with the
  SDK on 8 rows.
- ``duration`` / ``start_time`` / ``end_time``: tiny boundary differences
  on the 3 spontaneous block rows. We use epoch boundaries; the SDK
  uses the first/last actual stim time of the adjacent block.

Note: a small upstream bug in `build_events_and_intervals.py` puts the
wrong `stimulus_presentations_id` on `image_change` events (off by one).
This converter sidesteps the bug by matching `is_change` via timestamp.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Default location of the novelty lookup table (project root).
_DEFAULT_NOVELTY_LOOKUP = (
    Path(__file__).resolve().parent.parent / 'image_novelty_lookup.csv'
)


def load_novelty_lookup(path: str | Path | None = None) -> dict:
    """Return {(image_set_name, image_name): is_novel} from the CSV.

    The CSV ships at the project root (`image_novelty_lookup.csv`) and is
    intended to be extended over time as more image sets get characterized.
    """
    p = Path(path) if path is not None else _DEFAULT_NOVELTY_LOOKUP
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {(row['image_set_name'], row['image_name']): bool(row['is_novel'])
            for _, row in df.iterrows()}


# Map our epoch_name → the SDK's stimulus_block_name vocabulary.
# Spontaneous before the first named epoch → 'initial_gray_screen_5min'.
# Spontaneous between change_detection and natural_movie_one → 'post_behavior_gray_screen_5min'.
# Spontaneous after natural_movie_one → 'post_movie_gray_screen' (synthesized).
_SDK_BLOCK_NAME = {
    'change_detection': 'change_detection_behavior',
    'natural_movie_one': 'natural_movie_one',
}


def _spontaneous_block_name(epoch_idx: int, n_epochs: int) -> str:
    """Pick the SDK's name for a spontaneous block by its position."""
    if epoch_idx == 0:
        return 'initial_gray_screen_5min'
    if epoch_idx == n_epochs - 1:
        return 'post_movie_gray_screen'  # SDK uses something similar
    return 'post_behavior_gray_screen_5min'


def to_sdk_format(nwb) -> dict:
    """Return SDK-shaped (stimulus_presentations, trials) from a new NWB.

    Parameters
    ----------
    nwb : NdxEventsNWBFile (or NWBFile)
        Loaded NWB file produced by `package_to_nwb`.

    Returns
    -------
    dict with keys:
        'stimulus_presentations' : DataFrame (~13.8k rows)
        'trials' : DataFrame (~600 rows)
    """
    return {
        'stimulus_presentations': build_allensdk_stimulus_presentations_table(nwb),
        'trials': build_allensdk_trials_table(nwb),
    }


# ── Public API ────────────────────────────────────────────────────────
def build_allensdk_stimulus_presentations_table(nwb) -> pd.DataFrame:
    """Convert a packaged NWB into an SDK-shaped stimulus_presentations table.

    The returned DataFrame matches the column set and conventions of the
    table returned by ``BehaviorOphysExperiment.stimulus_presentations``
    when loading the old AllenSDK-format NWB. Drop-in compatible with
    existing analysis code that consumes that table.

    Parameters
    ----------
    nwb : NdxEventsNWBFile (or NWBFile)
        Loaded NWB file produced by `package_to_nwb`.

    Returns
    -------
    DataFrame indexed by `stimulus_presentations_id` (int).
    """
    return _build_stim_presentations(nwb)


def build_allensdk_trials_table(nwb) -> pd.DataFrame:
    """Convert a packaged NWB into an SDK-shaped trials table.

    The returned DataFrame matches the column set and conventions of
    ``BehaviorOphysExperiment.trials`` from the old AllenSDK NWB.

    Parameters
    ----------
    nwb : NdxEventsNWBFile (or NWBFile)
        Loaded NWB file produced by `package_to_nwb`.

    Returns
    -------
    DataFrame indexed by `trials_id` (int).
    """
    return _build_trials(nwb)


# ── stimulus_presentations ──────────────────────────────────────────
def _build_stim_presentations(nwb) -> pd.DataFrame:
    """Concatenate active task + natural_movie_one + spontaneous rows."""
    blocks = []
    epochs = nwb.epochs.to_dataframe().sort_values('start_time').reset_index(drop=True)
    n_ep = len(epochs)

    # Get the stimulus image set name from task_parameters (for stimulus_name)
    try:
        raw_name = nwb.lab_meta_data['task_parameters'].image_set_name
    except (KeyError, AttributeError):
        raw_name = 'unknown'
    # SDK convention: strip trailing date suffix like '.2017.07.14'
    import re
    image_set_name = re.sub(r'\.\d{2}\.\d{2}$', '', raw_name)

    # Build a synthetic block per epoch row
    for epoch_idx, ep in epochs.iterrows():
        ep_name = ep['epoch_name']
        if ep_name == 'change_detection':
            df = _active_block(nwb, image_set_name)
        elif ep_name == 'natural_movie_one':
            df = _movie_block(nwb)
        else:  # spontaneous
            df = _spontaneous_block(ep, epoch_idx, n_ep)
        df['stimulus_block'] = epoch_idx
        blocks.append(df)

    table = pd.concat(blocks, ignore_index=True, sort=False)
    table = table.sort_values('start_time').reset_index(drop=True)
    table.index.name = 'stimulus_presentations_id'

    # is_change: upstream events_df has an off-by-one bug where the change
    # event's stimulus_presentations_id points at the previous flash. Match
    # by start_time against the events table's image_change timestamps so we
    # land on the correct row regardless.
    if hasattr(nwb, 'get_events_table'):
        et = nwb.get_events_table('events').to_dataframe()
    else:
        et = nwb.events__events_tables['events'].to_dataframe()
    change_times = et.loc[et['event_type'] == 'image_change', 'timestamp'].values
    if len(change_times):
        is_change_mask = np.isin(table['start_time'].values.round(6),
                                  change_times.round(6))
        table['is_change'] = is_change_mask

    # is_image_novel: look up by (image_set_name, image_name) in the
    # external novelty CSV. The SDK's flag reflects which images are novel
    # *for this animal at this session type* — that's a project-level fact
    # we can't derive from one session. Returns <NA> if the image isn't in
    # the lookup or if it's a non-image row.
    novelty = load_novelty_lookup()
    novel = []
    for n in table['image_name']:
        if not isinstance(n, str) or n == 'omitted':
            novel.append(pd.NA)
        else:
            v = novelty.get((image_set_name, n))
            novel.append(v if v is not None else pd.NA)
    table['is_image_novel'] = pd.array(novel, dtype='boolean')

    # flashes_since_change — counts every row, resets to 0 on each is_change,
    # does NOT increment on omitted rows (the omitted slot inherits the
    # previous count). Counter runs across ALL blocks, not just active.
    fsc = []
    count = -1  # so first row becomes 0
    for _, row in table.iterrows():
        if row['is_change']:
            count = 0
            fsc.append(0)
            continue
        om = row.get('omitted')
        if om is True:  # exact bool match; avoids pd.NA ambiguity
            fsc.append(max(count, 0))
            continue
        count += 1
        fsc.append(count)
    table['flashes_since_change'] = fsc

    # Re-derive trials_id, but only within the active block (movie /
    # spontaneous rows stay at -99). SDK's convention: (start, next_start]
    # so flashes in the gap between trials get the just-ended trial's id.
    trials_df = nwb.trials.to_dataframe()
    starts = trials_df['start_time'].values
    last_stop = trials_df['stop_time'].iloc[-1]
    next_starts = np.append(starts[1:], last_stop)  # cap at last trial's stop
    new_tid = np.full(len(table), -99, dtype=int)
    times = table['start_time'].values
    active_mask = (table['stimulus_block_name'] == 'change_detection_behavior').values
    for i, (s, ns) in enumerate(zip(starts, next_starts)):
        mask = active_mask & (times > s) & (times <= ns)
        new_tid[mask] = i
    table['trials_id'] = new_tid

    # is_sham_change: per-flash flag — True if this flash is the "would-be"
    # change on a catch trial (i.e., matches a catch trial's change_time).
    sham_times = trials_df.loc[trials_df['catch'].astype(bool)
                               & trials_df['change_time'].notna(),
                               'change_time'].values
    sham_mask = np.zeros(len(table), dtype=bool)
    if len(sham_times):
        for st in sham_times:
            # Within 1 frame (~16ms) tolerance
            close = np.abs(table['start_time'].values - st) < 0.02
            sham_mask |= close
    table['is_sham_change'] = sham_mask

    # Final column order to match SDK
    sdk_cols = [
        'stimulus_block', 'stimulus_block_name', 'image_index', 'image_name',
        'movie_frame_index', 'duration', 'start_time', 'end_time',
        'start_frame', 'end_frame', 'is_change', 'is_image_novel', 'omitted',
        'movie_repeat', 'flashes_since_change', 'trials_id', 'is_sham_change',
        'stimulus_name', 'active',
    ]
    for c in sdk_cols:
        if c not in table.columns:
            table[c] = pd.NA
    return table[sdk_cols]


def _active_block(nwb, image_set_name: str) -> pd.DataFrame:
    """Rows for image flashes (active task)."""
    sp = nwb.intervals['stimulus_presentations'].to_dataframe()
    # image_index: sorted alphabetic mapping over image names, including
    # 'omitted' as its own index (SDK convention; total = N images + 1).
    real_names = sorted(set(sp['image_name']))  # includes 'omitted'
    name_to_idx = {n: i for i, n in enumerate(real_names)}
    image_names = sp['image_name'].astype(str).values  # keep 'omitted' literal
    # Frame columns: SDK uses end_frame; we store stop_frame.
    start_frames = (sp['start_frame'].astype(int).values
                    if 'start_frame' in sp.columns else np.full(len(sp), -99))
    end_frames = (sp['stop_frame'].astype(int).values
                  if 'stop_frame' in sp.columns else np.full(len(sp), -99))
    return pd.DataFrame({
        'stimulus_block_name': 'change_detection_behavior',
        'image_index': [name_to_idx.get(n, -99) for n in image_names],
        'image_name': image_names,
        'movie_frame_index': -99,
        'duration': sp['stop_time'].values - sp['start_time'].values,
        'start_time': sp['start_time'].values,
        'end_time': sp['stop_time'].values,
        'start_frame': start_frames,
        'end_frame': end_frames,
        'is_change': sp['is_change'].astype(bool).values,
        'omitted': pd.array(sp['omitted'].astype(bool).values, dtype='boolean'),
        'movie_repeat': -99,
        'trials_id': sp['trials_id'].astype(int).replace(-1, -99).values,
        'stimulus_name': image_set_name,
        'active': True,
    })


def _movie_block(nwb) -> pd.DataFrame:
    """Rows for natural_movie_one frames."""
    mt = nwb.intervals['natural_movie_one_presentations'].to_dataframe()
    n = len(mt)
    start_frames = (mt['start_frame'].astype(int).values
                    if 'start_frame' in mt.columns else np.full(n, -99))
    end_frames = (mt['stop_frame'].astype(int).values
                  if 'stop_frame' in mt.columns else np.full(n, -99))
    return pd.DataFrame({
        'stimulus_block_name': 'natural_movie_one',
        'image_index': -99,
        'image_name': pd.array([pd.NA] * n, dtype='string'),
        'movie_frame_index': mt['movie_frame_index'].astype(int).values,
        'duration': mt['stop_time'].values - mt['start_time'].values,
        'start_time': mt['start_time'].values,
        'end_time': mt['stop_time'].values,
        'start_frame': start_frames,
        'end_frame': end_frames,
        'is_change': False,
        'omitted': pd.array([pd.NA] * n, dtype='boolean'),
        'movie_repeat': mt['movie_repeat'].astype(int).values,
        'trials_id': -99,
        'stimulus_name': 'natural_movie_one',
        'active': False,
    })


def _spontaneous_block(ep: pd.Series, epoch_idx: int, n_ep: int) -> pd.DataFrame:
    """One row representing a spontaneous (gray-screen) interval."""
    return pd.DataFrame({
        'stimulus_block_name': [_spontaneous_block_name(epoch_idx, n_ep)],
        'image_index': [-99],
        'image_name': pd.array([pd.NA], dtype='string'),
        'movie_frame_index': [-99],
        'duration': [ep['stop_time'] - ep['start_time']],
        'start_time': [ep['start_time']],
        'end_time': [ep['stop_time']],
        'start_frame': [-99],
        'end_frame': [-99],
        'is_change': [False],
        'omitted': pd.array([pd.NA], dtype='boolean'),
        'movie_repeat': [-99],
        'trials_id': [-99],
        'stimulus_name': ['spontaneous'],
        'active': [False],
    })


# ── trials ──────────────────────────────────────────────────────────
def _build_trials(nwb) -> pd.DataFrame:
    """Rebuild SDK-shape trials table.

    Drops our extras (change_window_*, response_window_*, epoch_name, HED,
    warm_up) and adds the SDK extras (lick_times, trial_length, is_change).
    """
    t = nwb.trials.to_dataframe()
    aborted = t['aborted'].astype(bool)

    is_change = ((t['go'].astype(bool) | t['auto_rewarded'].astype(bool))
                 & t['change_time'].notna())

    # lick_times: SDK assigns licks to the trial that's "ending" — licks in
    # (trial_start, next_trial_start]. The last trial extends to +inf.
    if hasattr(nwb, 'get_events_table'):
        et = nwb.get_events_table('events').to_dataframe()
    else:
        et = nwb.events__events_tables['events'].to_dataframe()
    lick_ts = np.sort(et.loc[
        et['event_type'].astype(str).str.startswith('lick_'), 'timestamp'
    ].values)
    starts = t['start_time'].values
    stops = t['stop_time'].values
    # For trial i, include licks in (start_i, next_start_i] — except for the
    # last trial, where we use (start, stop] (no "next trial" to extend into).
    next_starts = np.append(starts[1:], stops[-1])
    lick_times = []
    for s, ns in zip(starts, next_starts):
        bucket = lick_ts[(lick_ts > s) & (lick_ts <= ns)]
        lick_times.append([float(x) for x in bucket])

    # change_frame: SDK uses -99 sentinel for missing, we use -1
    cf = t['change_frame'].copy()
    cf = cf.where(cf != -1, -99)

    # SDK's response_time / response_latency on trials with a change:
    # response_time = first lick in the trial (regardless of whether it
    # precedes or follows change_time — gives negative latency for
    # anticipatory licks). response_latency = first_lick - change_time, or
    # inf when no lick occurred.
    rt = t['response_time'].copy().astype(float)
    rl = t['response_latency'].copy().astype(float)
    for i, (lts, ct) in enumerate(zip(lick_times, t['change_time'].values)):
        if pd.isna(ct):
            continue
        if lts:
            rt.iloc[i] = lts[0]
            rl.iloc[i] = lts[0] - ct
        else:
            rl.iloc[i] = np.inf

    out = pd.DataFrame({
        'start_time': t['start_time'].values,
        'stop_time': t['stop_time'].values,
        'initial_image_name': t['initial_image_name'].values,
        'change_image_name': t['change_image_name'].values,
        'is_change': is_change.values,
        'change_time': t['change_time'].values,
        'go': t['go'].astype(bool).values,
        # SDK: catch/auto_rewarded are only True on non-aborted trials.
        'catch': (t['catch'].astype(bool) & ~aborted).values,
        'lick_times': lick_times,
        'response_time': rt.values,
        'response_latency': rl.values,
        'reward_time': t['reward_time'].values,
        'reward_volume': t['reward_volume'].values,
        'hit': t['hit'].astype(bool).values,
        'false_alarm': t['false_alarm'].astype(bool).values,
        'miss': t['miss'].astype(bool).values,
        'correct_reject': t['correct_reject'].astype(bool).values,
        'aborted': aborted.values,
        'auto_rewarded': (t['auto_rewarded'].astype(bool) & ~aborted).values,
        'change_frame': cf.values,
        'trial_length': t['stop_time'].values - t['start_time'].values,
    }, index=t.index)
    out.index.name = 'trials_id'
    return out
