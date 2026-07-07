# Change-detection / SweepStim NWB + HED packaging capsule

This Code Ocean capsule turns raw camstim sessions (`*_stim.pkl`/`*.pkl` +
`*_sync.h5`) into **HED-annotated NWB files** (written as **Zarr**) and emits
**one AIND-compliant derived data asset per session**. It does packaging only —
QC and validation plots are produced by a separate downstream capsule that
consumes the NWB output.

Two session types are supported:

- **change-detection** (active behavior) — packaged via `package_to_nwb.py`;
- **passive SweepStim** — packaged via the standalone `sweepstim_packaging/`
  module.

Unknown pkl structures are logged and skipped.

## What it produces

For each packaged session, one self-contained AIND **derived data asset** folder
under `/results`:

```
/results/
├── <primary-asset-name>_behavior-nwb_<date>_<time>/     one folder per session
│   ├── behavior.nwb.zarr/        NWB written as Zarr (a directory, not a file)
│   ├── behavior.events.json      BIDS-style column sidecar with HED tags
│   ├── data_description.json     authored + validated (name == folder name)
│   ├── processing.json           authored + validated (real monitor delay, HED version)
│   ├── subject.json    ┐
│   ├── procedures.json ├─ inherited byte-for-byte from the primary raw asset
│   ├── session.json    │
│   └── rig.json        ┘
├── manifest.json                 index of the combined asset (sessions/subjects)
├── session_metrics.csv           cross-session summary table
└── session_task_parameters.csv   cross-session task parameters
```

The NWB (`behavior.nwb.zarr`) contains:

- an `EventsTable` of point events (licks, rewards, image change, omissions);
- `TimeIntervals` tables for stimulus presentations, natural-movie frames,
  trials, and session epochs;
- task parameters via the custom `ndx-change-detection-task` extension;
- subject metadata (sex, genotype, strain, age, date-of-birth, species) threaded
  in from the primary asset's `subject.json`.

> **Zarr, not HDF5.** The NWB is written with `NWBZarrIO` as a `.nwb.zarr`
> directory. Any downstream code that reads these files **must use `NWBZarrIO`**,
> not `NWBHDF5IO`. This is the one coordinated change outside this capsule.

## Registering the output

Because every session folder carries its own valid AIND metadata and the whole
`/results` tree is self-describing (via `manifest.json` + the summary CSVs), the
intended workflow is:

1. run the capsule, then
2. **capture all of `/results` as a single combined Code Ocean data asset**, and
3. add that asset to a Code Ocean collection.

A combined multi-subject asset cannot carry one valid AIND `data_description` at
its root (that record is per-subject), so `manifest.json` is the honest root
descriptor. Each per-session folder remains independently registerable later if
you ever want per-session AIND assets.

## Which sessions are packaged

Two filters apply, in order:

1. **Mouse allow-list.** Only sessions whose subject (mouse) id is in
   `ALLOWED_MICE` (top of `run_capsule.py`) are packaged; all others are skipped
   and reported. The subject id is resolved cheaply from the primary asset
   (`data_description.json` → `subject.json` → the primary folder-name token)
   *before* the large pkl is loaded, so excluded mice cost almost nothing.
   Current default:

   ```
   782149, 788406, 790322, 800792, 800995, 804363
   ```

   Override without editing code via the `MICE` env var (comma-separated), or set
   `MICE=all` to package every discovered session.

2. **Session type.** change-detection and sweepstim are packaged; anything else
   is skipped and reported as an unsupported type.

## How sessions are discovered

`run_capsule.py` searches `/data` recursively for every `*_stim.pkl` and bare
`<id>.pkl` (numeric ids only), pairing each with the `*_sync.h5` in the same
folder. A session's **primary (raw) asset** is the top-level `/data/<asset>`
folder it came from (e.g. `multiplane-ophys_800792_2025-08-26_12-30-21`); the
capsule reads that asset's AIND metadata JSONs to build the derived asset.

## AIND derived-asset metadata

- `data_description.json` and `processing.json` are authored with
  **aind-data-schema 1.4.0** (emitted at schema **1.0.4**, `data_level=derived`)
  and validated on construction *and* re-validated after writing.
- **Robust to primary schema-version drift.** Only the required core fields are
  extracted from the primary `data_description.json`; drift in fields we don't
  use cannot break packaging. A genuinely missing required field fails **that
  session** loudly (it is reported as failed, not silently skipped).
- `data_description.name` always equals the asset folder name.
- `subject/procedures/session/rig.json` are copied **byte-for-byte** from the
  primary asset (never round-tripped through the library).
- **No silent failures.** Metadata is authored + validated *before* the NWB is
  written; if metadata cannot be written for an already-packaged session, its
  output folder is removed. An NWB is never left on disk without its metadata.
- `processing.json` records `processor_full_name = "Learning mFISH / V1 omFISH
  team"`, the **per-session monitor delay** (measured from the photodiode), the
  HED schema version, and the code url/version.

## Layout

```
code/
  run                            CO master script (runs `python -u run_capsule.py`)
  run_capsule.py                 entry logic: discover, mouse filter, package, metadata, manifest
  aind_metadata.py               AIND derived-asset metadata (data_description + processing)
  package_to_nwb.py              change-detection: builds NWB tables, writes Zarr + sidecar
  build_events_and_intervals.py  pkl + sync -> events_df / intervals_df (+ monitor delay)
  task_parameters.py             task-parameter LabMetaData builder
  ndx_change_detection_task/     custom NWB extension (task parameters)
  sweepstim_packaging/           passive SweepStim packaging (package.py, classify.py, ...)
  summarize_sessions.py          cross-session summary tables (Zarr-aware)
data/         raw primary assets (mounted at /data); one folder per session
environment/  Dockerfile (Code Ocean base image + dependencies)
metadata/     metadata.yml
results/       outputs land here (one derived-asset folder per session)
```

## Run locally (outside Code Ocean)

```bash
cd code

# default: only the six mice in ALLOWED_MICE
DATA_DIR=../data RESULTS_DIR=../results python run_capsule.py

# a custom subset of mice
DATA_DIR=../data RESULTS_DIR=../results MICE=800792,804363 python run_capsule.py

# every discovered session (no mouse filter)
DATA_DIR=../data RESULTS_DIR=../results MICE=all python run_capsule.py
```

The startup log prints `Restricting to N requested mouse id(s): …` and the final
tally reports how many sessions were skipped for each reason (mouse not
requested, unsupported type, no sync, failed).

## Environment

Requires **Python ≥ 3.10** (the `ndx-hed` 0.2.0 dependency does not support
3.9). Dependency stack: numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, h5py 3.16.0,
six 1.17.0, pynwb 3.1.3, hdmf 4.3.1, ndx-events 0.4.0, ndx-hed 0.2.0, hedtools
1.1.0, plus **aind-data-schema 1.4.0, hdmf-zarr 0.13.0, zarr 2.18.3**. Set the
Python version and packages via the Code Ocean Environment editor (see
`environment/Dockerfile`).

> The first line of `environment/Dockerfile` is a Code-Ocean-managed `sha256`
> hash. It is regenerated automatically when you edit the environment through the
> CO UI — do not hand-edit that line.

## Notes

- Timestamps come from the sync file's vsync falling edges. The **monitor
  delay** is computed per session from the photodiode and applied **only** to
  visual events; licks and rewards use bare vsync times. The measured value is
  recorded in each session's `processing.json`.
- Passive SweepStim sessions do not measure a monitor delay, so their
  `processing.json` records only the HED schema version.
- The pandas `FutureWarning`s during reward/lick classification are harmless on
  the pinned pandas but should eventually be fixed by casting the target columns
  to `object` before assignment.
```
