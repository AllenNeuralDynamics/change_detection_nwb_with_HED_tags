# Plan: HED-compliant NWB packaging for passive SweepStim sessions

**Status:** design only — no code written yet.
**Author context:** Marina G. (marinag@alleninstitute.org), 2026-06-25.
**Trigger:** session `multiplane-ophys_790322_2025-07-12_10-41-51`
(`1448610339_stim.pkl`, stage `STAGE_0`) crashes the change-detection
packager at `compute_timestamp_alignment` →
`KeyError: 'behavior'`.

---

## 1. Why SweepStim sessions are fundamentally different

The current pipeline (`build_events_and_intervals.py` + `package_to_nwb.py`)
assumes a **camstim `DoC` / Foraging behavior session**: a `behavior` item
with `trial_log`, `stimuli['images'|'grating']`, licks, rewards, change/response
windows. A passive **SweepStim** session has none of that. Comparison:

| Aspect | Change-detection (`behavior`) | Passive SweepStim (this session) |
|---|---|---|
| behavior item key | `pkl['items']['behavior']` | `pkl['items']['foraging']` (mostly empty: no `trial_log`, empty `items`/`stimuli`) |
| stimulus location | `behavior['stimuli']['images'\|'grating']` (dict) | top-level `pkl['stimuli']` (**list** of 56 SweepStim objects) |
| task structure | trials, changes, licks, rewards | **none** — passive viewing, no trials/licks/rewards |
| stimulus content | flashed images / continuous gratings | 56 warped natural-movie clips (`ds_warped_15_Session_0_XX.npy`) + 1 repeated "test" clip |
| frame-count source | `behavior['intervalsms']` | top-level `pkl['intervalsms']` / `total_frames` / `vsynccount` |
| stage | `TRAINING_*` / `DoC` | `STAGE_0` (passive habituation/mapping) |
| shipped table | `*_stim_table.csv` | `*_vsync_table.csv` (already-parsed frame table) |

Conclusion: this is a **separate code path**, not a branch inside the
change-detection builders. Trying to thread it through `build_events_table`
(needs `trial_log`, change windows, lick/reward classification) would be
forcing a square peg.

---

## 2. SweepStim data model (verified from the pkl)

Top level:
- `pkl['stimuli']` — `list` of 56 stimulus objects (one per movie clip), in
  presentation order. Object 55 is `Session_test_0_0` (the repeated test movie,
  `runs=20`).
- `pkl['intervalsms']` — 274,979 inter-vsync intervals (ms). `n_frames = len + 1`.
- `pkl['fps']` = 60.0, `pkl['total_frames']` = 274,740, `pkl['vsynccount']` = 274,980.
- `pkl['pre_blank_sec']` = 2, `pkl['post_blank_sec']` = 2, `pkl['stage']` = `STAGE_0`.
- `pkl['items']['foraging']['encoders']` — running-wheel encoder (same shape as
  `behavior['encoders']`, so running-speed extraction is reusable).

Each stimulus object (e.g. `pkl['stimuli'][0]`):
- `movie_path` / `stim_path` — clip identity, e.g. `…ds_warped_15_Session_0_00.npy`.
- `display_sequence` — `[[start_sec, stop_sec]]`, the wall-clock window this clip
  owns (e.g. `[[0, 53]]`, `[[53, 98]]`, …, test clip `[[1538, 3158]]`). These
  tile the session.
- `runs` — number of repeats of the clip within its window (10; test clip 20).
- `frame_list` — per-display-frame movie-frame index for **one run**
  (`[0,0,1,1,2,2,…]` → each movie frame held for 2 display frames = 30 Hz movie
  on a 60 Hz monitor; `-1` marks blank frames).
- `sweep_order` — length `n_sweeps` (e.g. 15,900), the movie-frame index shown at
  each sweep across all runs.
- `sweep_frames` — length `n_sweeps`, list of `(start_display_frame,
  stop_display_frame)` tuples giving each sweep's frame span.
- `sweep_params` — sweep parameter values (format varies: list/dict keyed by
  `dimnames`, here `['ReplaceImage']`); not needed for timing.
- `blank_sweeps`, `blank_length` — inter-sweep blank config (0 here).

**This is the same SweepStim shape the code already parses** for the
change-detection "fingerprint" natural_movie_one clip
(`build_events_and_intervals.py:533-592`, using `frame_list`, `sweep_frames`,
`frame_indices`). That block is the working template for frame→time mapping.

### Shipped `*_vsync_table.csv` (137,371 rows)
Columns: `start_time, stop_time, stim_name, stim_type, stim_block, frame,
stim_index`. One row per displayed movie frame, plus a leading `spontaneous`
row. `stim_type` = `ImageStimNumpyuByte`, 56 `stim_block`s. This is a
ready-made cross-check (and a possible fast path), but its times appear to be
on the stimulus clock — **we should still align via the sync file** to stay
consistent with the change-detection path (monitor delay, vsync edges) and use
the CSV only for validation.

---

## 3. Proposed architecture

Add a **session-type dispatch** at the top of the pipeline, keep the two paths
separate, and converge on a shared NWB-writing layer.

```
run_capsule.main()
  └─ package_to_nwb(pkl, sync, out)
       ├─ detect_session_type(pkl)  →  'behavior' | 'sweepstim'
       ├─ if 'behavior':  build_all() ... (existing, unchanged)
       └─ if 'sweepstim': build_all_sweepstim()  (NEW)
              ├─ compute_timestamp_alignment(pkl, sync)   # generalized frame-count lookup
              ├─ build_stimulus_presentations_sweepstim() # per-frame movie table + HED
              ├─ build_epochs_sweepstim()                 # per-clip + spontaneous epochs
              └─ (reuse) add_running_speed(), build_nwbfile(), HED sidecar, write
```

### 3.1 Session-type detection (`detect_session_type`)
Decide by structure, not stage string (robust to future stages):
1. `'behavior' in pkl['items']` **and** that item has a non-empty `trial_log`
   → `'behavior'`.
2. `pkl['stimuli']` is a non-empty `list` (SweepStim objects) → `'sweepstim'`.
3. Else: raise a clear, actionable error naming the observed `items`/`stimuli`
   keys.

`run_capsule.py` should additionally **catch-and-skip** any session that raises
an "unsupported session type" error, logging a warning, so one odd session
never aborts a whole batch.

### 3.2 Generalize `compute_timestamp_alignment`
Only one line is session-specific:
```python
n_pkl_frames = len(pkl['items']['behavior']['intervalsms']) + 1
```
Replace with a helper that resolves the frame count from, in order:
`pkl['items']['behavior']['intervalsms']` → `pkl['items']['foraging']['intervalsms']`
→ top-level `pkl['intervalsms']` → `pkl['vsynccount']` / `pkl['total_frames']`.
Everything downstream (vsync falling edges, photodiode monitor delay) is
stimulus-agnostic and works as-is.

---

## 4. Stimulus-presentation extraction (the core new work)

Produce one NWB `TimeIntervals` table, `stimulus_presentations`, with **one row
per displayed movie frame** (mirroring `natural_movie_one_presentations`).

Algorithm, per stimulus object `s` in `pkl['stimuli']`:
1. Resolve clip label from `movie_path`/`stim_path` basename, stripped to e.g.
   `Session_0_00` (and `Session_test_0_0`).
2. Walk `sweep_order` + `sweep_frames`. For sweep `k`:
   - `movie_frame_index = sweep_order[k]` (frame within the clip).
   - `start_display_frame, stop_display_frame = sweep_frames[k]` — these are
     **global vsync frame indices**; map to time via `stim_ts_visual[frame]`
     (the visual array that includes monitor delay), exactly like the
     fingerprint block. Skip sweeps whose frame ≥ `n_frames`.
   - `run_index` = `k // sweeps_per_run` (where `sweeps_per_run = n_sweeps //
     runs`), to populate a `movie_repeat`-style column.
3. Emit columns: `start_time, stop_time, start_frame, stop_frame,
   movie_name (clip label), movie_frame_index, movie_repeat, stim_block,
   epoch_name, HED`.

Notes / edge cases to handle:
- **Blank frames** (`frame_list == -1`, `blank_sweeps`): represent as gaps or as
  explicit `blank`/`spontaneous` rows — pick one and document (recommend gaps,
  with spontaneous covered by the epoch layer in §5).
- `display_sequence` defines each clip's wall-clock window; use it to assign
  `stim_block` and to sanity-check that sweep times fall inside the window.
- Validate the reconstructed row count/timing against `*_vsync_table.csv`
  (137,371 frame rows here) as an automated test.

A discrete **events table** is optional for passive sessions. If we keep one for
schema parity, emit only `movie_onset`/`movie_offset` rows (no licks/rewards/
changes). Otherwise omit `nwb.add_events_table` for this path.

---

## 5. Epochs / intervals

No trials. Build the flat intervals / epoch layer from `display_sequence`:
- One `epoch` row per movie clip window (`label = clip name`, e.g.
  `Session_0_00`), from `display_sequence` start→stop.
- `spontaneous` rows for the pre-blank (`pre_blank_sec`), inter-clip gaps, and
  post-blank (`post_blank_sec`) — reuse the gap-filling logic in
  `build_epoch_lookup` (`package_to_nwb.py:1150`).
- The repeated test clip (`Session_test_0_0`, `runs=20`) is its own epoch and is
  the natural candidate for a "fingerprint"-style repeated-movie analysis.

---

## 6. HED tagging scheme (HED v8.3.0 base schema — the compliance requirement)

Follow the existing orthogonal-composition conventions
(`HED_TAGS`, `_stim_presentation_hed`, the movie tag at
`package_to_nwb.py:1018`). All tags validate against HED 8.3.0.

Per-frame movie presentation (`stimulus_presentations.HED`):
```
Sensory-event, Visual-presentation, (Movie, Label/<clip_name>)
```
- `<clip_name>` = sanitized clip label (`Session_0_00`). HED `Label/` values
  must be alphanumeric/underscore — sanitize dots/dashes.
- Optionally add `(Movie, Label/<clip>), (Temporal-marker, Label/frame_<idx>)`
  if per-frame granularity in HED is wanted; recommend keeping the frame index
  in the `movie_frame_index` **column** and the HED at clip granularity to avoid
  137k near-identical tag strings (column-level HED via the sidecar is the
  HED-idiomatic place for per-row numeric metadata).

Repeated test clip: same tag with its own label, optionally
`…, Experimental-stimulus` to mark it as the analysis stimulus.

Spontaneous / blank epochs (`intervals.HED`, reuse existing spontaneous tag):
```
Experimental-procedure, (Task, Label/spontaneous)
```
or, for a true gray screen, `Sensory-event, Visual-presentation,
(Background-view, Grayscale)`.

Per-clip epoch rows (`intervals.HED`):
```
Experimental-procedure, (Task, Label/passive_viewing), (Movie, Label/<clip>)
```

**Sidecar** (`build_events_sidecar`, `package_to_nwb.py:1556`): add column
descriptions + column-level HED for the new columns (`movie_name`,
`movie_frame_index`, `movie_repeat`, `stim_block`). Keep `hed_schema_version`
on the `HedLabMetaData` as today.

The `ndx-hed` validator should be run over the produced file as the compliance
gate (see §9).

---

## 7. Reuse map (what already exists)

| Need | Reuse |
|---|---|
| Sync load, vsync edges, monitor delay | `compute_timestamp_alignment` (only the frame-count line is session-specific) |
| SweepStim frame→time mapping | fingerprint block `build_events_and_intervals.py:533-592` (lift into a shared `iter_sweepstim_frames()` helper) |
| Per-frame movie `TimeIntervals` | `build_natural_movie_one_presentations` (`:1007`) as the structural template |
| Epoch list + spontaneous gap fill | `build_epoch_lookup` (`:1150`), `add_epochs` (`:1210`) |
| Running speed + raw encoder | `compute_running_speed`/`add_running_speed` (`:726`/`:790`) — change `pkl['items']['behavior']['encoders']` to a resolved-item lookup |
| NWBFile, Subject, HED metadata, sidecar, writer | `build_nwbfile` (`:849`), `build_subject` (`:836`), `build_events_sidecar` (`:1556`), `package_to_nwb` tail (`:1711`) |

Refactor opportunity: extract `get_behavior_item(pkl)` /
`get_frame_count(pkl)` helpers used by both paths so `behavior`-vs-`foraging`
access lives in one place.

---

## 8. Explicitly out of scope for the passive path
- trials table, change/response windows, lick & reward classification,
  task_parameters (`ChangeDetectionTaskParameters`) — none apply. Provide a
  minimal passive `LabMetaData` (or none) instead of forcing the change-detection
  schema.

---

## 9. Validation
1. **Frame-table cross-check:** reconstructed `stimulus_presentations` start/stop
   times and frame counts match `1448610339_vsync_table.csv` (137,371 rows,
   56 blocks) within vsync tolerance.
2. **Timeline coverage:** epochs + spontaneous tile `[0, session_end]` with no
   gaps/overlaps; clip windows match `display_sequence`.
3. **HED validation:** run the `ndx-hed`/`hed-python` validator on the written
   NWB sidecar + columns; zero errors against schema 8.3.0.
4. **Regression:** the three existing change-detection sessions
   (`new_ophys_mfish_data`, `old_vbo_data`, gratings `…2025-08-13…`) produce
   byte-identical output to before (dispatch must not touch their path).
5. **NWB read-back:** `pynwb` opens the file; `stimulus_presentations`,
   `epochs`/intervals, running speed all present and well-typed.

---

## 10. File-by-file change list (when implementation is approved)
- `build_events_and_intervals.py`
  - `get_frame_count(pkl)` / generalize `compute_timestamp_alignment:235`.
  - `iter_sweepstim_frames(stim_obj, stim_ts_visual, n_frames)` (lift from fingerprint block).
  - `build_all_sweepstim(pkl_path, sync_path)` returning a stim-presentation df + epoch df + alignment.
- `package_to_nwb.py`
  - `detect_session_type(pkl)`; branch in `package_to_nwb`.
  - `build_stimulus_presentations_sweepstim(df, epoch_list)`; `_movie_presentation_hed()`.
  - `build_passive_epochs(...)` (or reuse epoch lookup with clip+spontaneous rows).
  - generalize encoder access in `compute_running_speed:749`.
  - extend `build_events_sidecar` with the new columns/HED.
- `run_capsule.py`
  - wrap `package_to_nwb` in try/except for "unsupported session" → warn + skip.
- tests/: add a SweepStim fixture + the §9 checks.

---

## 11. Open decisions (defaults chosen; flag if you disagree)
1. **Per-frame vs per-sweep-block rows.** Default: one row per displayed frame
   (matches `vsync_table.csv` and `natural_movie_one_presentations`). Alternative:
   collapse contiguous frames of a run into one row per run (smaller table).
2. **HED granularity.** Default: clip-level HED string + numeric columns for
   frame/repeat. Avoids 137k unique tag strings.
3. **Discrete events table.** Default: omit for passive sessions (no point
   events); keep only `stimulus_presentations` + epochs + running. Revisit if
   downstream tools require an `events` table to exist.
4. **Clip naming.** Default: `Session_0_00` style from `movie_path` basename;
   confirm whether a friendlier scientific label is preferred.
```
