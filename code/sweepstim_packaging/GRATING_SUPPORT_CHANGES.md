# SweepStim NWB packaging — grating support & cleanup

Session summary, written so another agent can reproduce the changes from scratch in
a different checkout/environment. It covers **what changed, why, and how to verify**.

Scope: the passive **SweepStim** packaging path only (`code/sweepstim_packaging/`),
which packages passive visual-stimulus sessions (no behavior/licks/rewards). The
change-detection path (`behavior_nwb*`, `sdk_compat.py`) is unrelated and untouched.

---

## 1. Background: two SweepStim session types

`pkl['stimuli']` is a list of stimulus **blocks**. Historically the packager assumed
every block was a **movie** (natural-movie clips shown frame by frame). But passive
sessions come in (at least) two flavours:

| session_type | example SID | blocks | each presentation is | key pkl fields |
|---|---|---|---|---|
| movie (STAGE_0)    | `1412914743` | 56 | one movie **frame** (~33 ms, 30 Hz) | `movie_path`, `dimnames == ['ReplaceImage']`, `sweep_table is None` |
| gratings (STAGE_1) | `1414269789` | 2  | one grating **sweep** (~2 s)         | `dimnames == ['Contrast','TF','SF','Ori']`, `sweep_table` = list of param tuples |

Test data used this session (paths are environment-specific):
- Gratings: `/data/multiplane-ophys_755252_2025-01-13_09-33-41/behavior/1414269789.pkl`
  (+ sync `1414269789_20250113T093325.h5`)
- Movie: `/data/multiplane-ophys_755252_2025-01-03_12-36-54/behavior/1412914743_stim.pkl`
  (+ sync `1412914743_sync.h5`)

**The bug:** for a grating block, the packager read the `frame_list` run value and
stored it as `movie_frame_index`. For gratings that value is a **condition index into
`sweep_table`**, so the real stimulus parameters (orientation, contrast, temporal &
spatial frequency) were silently discarded — the NWB `stimulus_presentations` table
had only `movie_name`/`movie_frame_index`/`movie_repeat`, meaningless for gratings.

---

## 2. The pkl data model (ground truth)

Per **grating** block in `pkl['stimuli'][i]`:

- `dimnames` — parameter names **and** the tuple order of each `sweep_table` row,
  e.g. `['Contrast','TF','SF','Ori']`.
- `sweep_table` — list of per-condition tuples, e.g. `sweep_table[23] == (0.2, 1.0, 0.04, 315)`
  meaning Contrast=0.2, TF=1.0, SF=0.04, Ori=315.
- `sweep_params` — `{param: ([unique values], dim_index)}`.
- `frame_list` — indexed by **global display frame** (60 Hz from session start; aligns
  1:1 with the vsync timebase). Value = the on-screen condition index (grating) or
  movie frame index (movie); `-1` where the block is not on screen. **Authoritative
  record of what was shown when.** Each maximal run of a constant value `>= 0` is one
  presentation.
- `sweep_order`, `sweep_frames` — per-sweep condition indices / frame spans (used only
  by the no-`frame_list` fallback).
- `stim` — a repr string of the PsychoPy stim, e.g. `GratingStim(...)` or
  `ImageStimNumpyuByte(...)`; first identifier = the stim class.

Per **movie** block: `sweep_table is None`, `dimnames == ['ReplaceImage']`,
`movie_path` set; `frame_list` value = movie frame index.

Verified the mapping against the session's `1414269789_stim_table.csv` (an AllenSDK
export): parameter **values** matched exactly (row 0 → Ori 315, Contrast 0.2, TF 1,
SF 0.04). NOTE: the CSV was treated only as a column-name reference, **not** ground
truth — its **row counts are wrong** (claims 560 sweeps/block; `frame_list` shows
546/547 actually drawn because the session ended before all 600 planned sweeps played).
Ground truth = pkl + sync.

---

## 3. Changes to `package.py`

All edits are in `code/sweepstim_packaging/package.py`.

### 3a. Per-block type detection + parameter resolution (new helpers)

Add near the other small helpers (after `_clip_name`):

- `_dimname_to_column(dimname)` — maps a camstim dimension name to a snake_case column
  via a dict `_DIMNAME_COLUMN = {'Contrast':'contrast','TF':'temporal_frequency',
  'SF':'spatial_frequency','Ori':'orientation','Phase':'phase','Pos':'position',
  'Size':'size'}`, falling back to a normalized lowercase name.
- `_is_grating_block(stim_obj)` — `True` iff `sweep_table` is non-empty **and**
  `dimnames != ['ReplaceImage']` (module constant `_MOVIE_DIMNAMES = ['ReplaceImage']`).
- `_stim_type(stim_obj)` — regex the leading identifier out of the `stim` repr string
  (`GratingStim`, `ImageStimNumpyuByte`); fallback `'GratingStim'`/`'MovieStim'`.
- `_grating_params(sweep_table, dim_columns, value)` — returns `{column: float(value)}`
  from `sweep_table[value]` zipped with the dim columns; `{}` if `value` out of range.

### 3b. Populate rows per block (`_block_rows_from_frame_list`, and the
`_block_rows_from_sweep_frames` fallback)

For each presentation run, branch on `_is_grating_block`:
- **grating**: emit `condition_index` (the `frame_list` value), `condition_repeat`
  (0-based count of prior occurrences of that condition), and the resolved parameter
  columns (`row.update(_grating_params(...))`).
- **movie**: emit `movie_frame_index` and `movie_repeat` (unchanged behaviour).

Both branches also set the shared fields (see 3d). The fallback additionally now skips
blank (`value < 0`) sweeps.

### 3c. HED tags (per-row, validated on write)

HED 8.3.0 has **no `Grating` tag** (verified with `hed.schema.load_schema_version`;
`Movie` exists, `Grating` does not). So:
- movie rows: `Sensory-event, Visual-presentation, (Movie, Label/<clip>)` (unchanged)
- grating rows: `Sensory-event, Visual-presentation, (Image, Label/<clip>)`
- epoch HED (`_build_epoch_list`): media tag chosen per block —
  `Image` for gratings, `Movie` for movies.

Parameter **values** are not embedded in per-row HED (kept coarse, like movies);
they live in columns + the sidecar as `Label/<name>-#` templates.

### 3d. Session-driven column schema (`build_stimulus_presentations_sweepstim`)

The `stimulus_presentations` `TimeIntervals` is built dynamically:
- **Base columns present on every row** (module constant `_BASE_PRESENTATION_SPECS`, in
  output order): `start_time`, `stop_time`, `epoch_name`, `stim_type`, `stim_block`,
  `start_frame`, `stop_frame`. `HED` is appended last.
- **Optional columns** are session-driven: `_presentation_extra_columns(rows)` returns
  the union of non-base keys actually used by any row, ordered by preference
  (`_EXTRA_COLUMN_ORDER`) then alphabetically. Missing values are filled with `np.nan`.
  → a movie session gets `movie_frame_index`/`movie_repeat`; a grating session gets the
  grating params + `condition_index`/`condition_repeat`; a mixed session gets the union.

Column descriptions come from `_COLUMN_DESCRIPTIONS`; HED sidecar templates from
`_COLUMN_HED_TEMPLATE`.

### 3e. `epoch_name` carries the stimulus label; `stim_name` removed

Originally each row had `stim_name` = clip/grating name and `epoch_name` = constant
`"passive_viewing"`. Changed so:
- `epoch_name` = the clip/grating name (i.e. the label of the epoch the presentation
  falls in), and
- `stim_name` is **removed** entirely.

This also fixes a latent inconsistency: a presentation's `epoch_name` now equals the
`label` of its containing epoch in the `intervals` table. Concretely:
- in both row-builders, drop `"stim_name": clip` and set `"epoch_name": clip`;
- `_BASE_PRESENTATION_SPECS`: remove the `stim_name` entry, move `epoch_name` up to the
  3rd position with description `"Epoch label: movie clip or grating stimulus name."`;
- `build_intervals_table_sweepstim`: the stimulus_presentation rows' `label` now comes
  from `row["epoch_name"]` (was `row["stim_name"]`);
- `build_sweepstim_sidecar`: drop `stim_name`, update `epoch_name` description.

### 3f. Sidecar (`build_sweepstim_sidecar(rows)`)

Now takes `rows` and emits base column descriptions plus one entry per session-driven
optional column (reusing `_presentation_extra_columns`). Called as
`build_sweepstim_sidecar(rows)` in `package_sweepstim_to_nwb`.

---

## 4. Remove raw encoder voltages (`running.py`)

`v_sig` and `v_in` (raw rotary-encoder voltages) were written to `nwb.acquisition`.
They're unused downstream and clutter the file. In `add_running_speed`
(`code/sweepstim_packaging/running.py`): delete the `v_sig_ts`/`v_in_ts` `TimeSeries`
and their two `nwb.add_acquisition(...)` calls. The voltages are **still read** to
*compute* speed (`compute_running_speed`) — only the stored copies are removed. Result:
`nwb.acquisition` is empty; the `running` processing module keeps `speed` and `dx`.

---

## 5. Inspection notebooks (split by stim type)

The old generic `inspect_sweepstim_nwb.ipynb` was split into two stim-specific
notebooks (both packaged on demand to `/results/<sid>.nwb`, reused on re-run):

- `inspect_sweepstim_nwb.ipynb` — **movie-specific** (default SID `1412914743`).
  Frames-per-clip + block timeline; ~33 ms/30 Hz frame-timing; `movie_frame_index`
  ramp-and-repeat plot; repeats-per-block; movie-specific consistency checks (frame
  index monotonic within each repeat, no grating columns leaked).
- `inspect_sweepstim_gratings_nwb.ipynb` — **new, grating-specific** (default SID
  `1414269789`). Parameter-count bars + orientation polar plot; per-block 2-D
  parameter-grid coverage heatmaps; `condition_index`-over-time + sweep-duration
  histogram; **independent re-derivation of every parameter from the pkl `sweep_table`**
  (must be 0 mismatches); grating-specific consistency checks (params populated, each
  `condition_index` maps to one fixed parameter tuple, no movie columns leaked).

Both notebooks: read with **`NWBZarrIO`** (see §6), classify the session type via
`_is_grating_block` in a `resolve_session` helper, and assert the loaded session is the
expected type (pointing to the other notebook otherwise).

The notebooks were generated by a Python builder script using `nbformat` (not committed;
lived in a scratch dir). Regenerating by hand-editing cells is fine too — the important
thing is the cell content described above.

### Reader-backend bug fixed first
The packager writes **NWB-Zarr** (`NWBZarrIO`), which is a **directory** on disk. The
original notebook opened it with `NWBHDF5IO`, which raised
`IsADirectoryError: Is a directory`. Fix: import and use
`from hdmf_zarr.nwb import NWBZarrIO` for reading.

---

## 6. Timing investigation — concluded NO change needed

While cross-checking, the packager placed the first grating at **t≈21.68 s** but the
CSV at **41.70 s** (a ~20 s gap == `pre_blank_sec=20`). This looked like a possible
off-by-pre_blank bug in `frame_list → stim_ts_visual` alignment.

**Resolution (do not re-litigate):** the **photodiode** line in the sync file is the
arbiter — it physically toggles only when a stimulus is drawn (static during gray
pre-blank). The first regular ~1 Hz photodiode edge is at **21.683 s**, matching the
packager, **not** the CSV. So the packager's timing is **correct**; the CSV inserted a
phantom 20 s pre-blank. `compute_sweepstim_timestamp_alignment` was left unchanged. The
`vsynccount − total_frames ≈ (pre+post blank)×60` coincidence is not evidence of a bug.

To re-verify: extract stim_vsync falling edges and stim_photodiode "both" edges from the
sync h5, find the first photodiode edge whose neighbour spacing is ~1 s, and compare to
`stim_ts_visual[0]`.

---

## 7. Final `stimulus_presentations` schema

Movie session columns:
```
start_time, stop_time, epoch_name, stim_type, stim_block,
start_frame, stop_frame, movie_frame_index, movie_repeat, HED
```
Gratings session columns:
```
start_time, stop_time, epoch_name, stim_type, stim_block,
start_frame, stop_frame, orientation, spatial_frequency,
temporal_frequency, contrast, condition_index, condition_repeat, HED
```
(`epoch_name` holds the clip/grating name; `stim_type` is the PsychoPy class;
`acquisition` is empty; `processing/running` has `speed` + `dx`.)

---

## 8. How to verify (reproduce the checks)

Re-package both sessions and inspect:

```python
from sweepstim_packaging import package_sweepstim_to_nwb
package_sweepstim_to_nwb(PKL, SYNC, "/results/<sid>.nwb")   # writes NWB-Zarr + .events.json

from hdmf_zarr.nwb import NWBZarrIO
import ndx_events, ndx_hed                                   # register namespaces
nwb = NWBZarrIO("/results/<sid>.nwb", "r", load_namespaces=True).read()
sp = nwb.intervals["stimulus_presentations"].to_dataframe()
```

Expected on the test data:
- Gratings `1414269789`: 1093 rows; `stim_type == GratingStim`; orientation
  {0,45,…,315}; contrast {0.05,0.1,0.2,0.4,0.8}; TF {1,2,4,8,15}; SF {0.04};
  2 blocks × 40 conditions. Independent re-derivation from `sweep_table` → **0
  mismatches** over 4372 parameter values.
- Movie `1412914743`: 137370 rows; `stim_type == ImageStimNumpyuByte`; 56 clips;
  frame duration median ≈ 33.4 ms; `movie_repeat` up to 19; no grating columns.
- Both: `nwb.acquisition` empty; `epoch_name` equals the containing epoch's label;
  HED writes/validate (uses `Image`/`Movie`, never `Grating`).

Running both notebooks end-to-end (`jupyter nbconvert --to notebook --execute --inplace`)
must complete with every consistency-check cell printing "All … checks passed."

---

## 9. Files changed

- `code/sweepstim_packaging/package.py` — grating detection + parameter extraction,
  session-driven columns, HED `Image` tag, `epoch_name` replaces `stim_name`, dynamic
  sidecar.
- `code/sweepstim_packaging/running.py` — stop storing `v_sig`/`v_in`.
- `code/sweepstim_packaging/inspect_sweepstim_nwb.ipynb` — now movie-specific
  (also switched reader to `NWBZarrIO`).
- `code/sweepstim_packaging/inspect_sweepstim_gratings_nwb.ipynb` — new gratings notebook.

`compute_sweepstim_timestamp_alignment` (timestamp_alignment.py) intentionally
**unchanged** (see §6).
