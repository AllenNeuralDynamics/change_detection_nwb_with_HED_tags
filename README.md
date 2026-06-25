# Change-detection NWB + HED packaging capsule

This Code Ocean capsule turns a raw camstim change-detection session
(`*_stim.pkl` + `*_sync.h5`) into a **HED-annotated NWB file** and a
**BIDS-style events sidecar JSON**. It does packaging only — QC and
validation plots are produced by a separate downstream capsule that
consumes the NWB output.

## What it produces

For each session it writes to `/results`:

- `<id>.nwb` — NWB v2 file with `ndx-events` + `ndx-hed`, containing:
  - an `EventsTable` of point events (licks, rewards, image change, omissions)
  - `TimeIntervals` tables for stimulus presentations, natural-movie frames,
    trials, and session epochs
  - task parameters via the custom `ndx-change-detection-task` extension
- `<id>.events.json` — column-level sidecar describing every field and its
  HED tags (levels, value templates).

## Layout

```
code/        pipeline + entry point (`run`) + test notebook
  run                          bash entry point (Code Ocean runs this)
  package_to_nwb.py            Stage 2: builds NWB tables, writes file + sidecar
  build_events_and_intervals.py Stage 1: pkl + sync -> events_df / intervals_df
  task_parameters.py           task-parameter LabMetaData builder
  ndx_change_detection_task/   custom NWB extension (task parameters)
  test_nwb_output.ipynb        loads a packaged NWB and inspects the tables
data/        embedded test sessions (mounted read-only at /data)
  new_ophys_mfish_data/  1464696201_stim.pkl + _sync.h5
  old_vbo_data/          1050231786_stim.pkl + _sync.h5
environment/ Dockerfile (pinned dependency versions)
metadata/    metadata.yml
results/     outputs land here
```

## How `run` finds sessions

`run` searches `/data` recursively for every `*_stim.pkl`, pairs each with
the `*_sync.h5` in the same folder, and packages it to `/results/<id>.nwb`
(`<id>` = the pkl filename minus `_stim`). Drop a new session folder into
`/data` and it will be packaged with no code changes.

## Run locally (outside Code Ocean)

```bash
DATA_DIR=./data RESULTS_DIR=./results bash code/run
```

## Pinned environment

Python 3.10 with: numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, h5py 3.16.0,
six 1.17.0, pynwb 3.1.3, hdmf 4.3.1, ndx-events 0.4.0, ndx-hed 0.2.0,
hedtools 1.1.0. See `environment/Dockerfile`.

## Notes

- Timestamps come from the sync file's vsync falling edges. Monitor delay
  (~35 ms) is applied **only** to visual events; licks and rewards use bare
  vsync times.
- The pandas `FutureWarning`s during reward/lick classification are harmless
  on the pinned pandas but should eventually be fixed by casting the target
  columns to `object` before assignment.
