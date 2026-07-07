# Work summary — AIND-compliant per-session derived assets

Session date: 2026-07-07. Subject of work: making the behavior-NWB capsule emit
**one AIND derived data asset per session** under `/results` — NWB written as
Zarr, with authored + validated `data_description.json` / `processing.json` and
inherited-verbatim `subject/procedures/session/rig.json` in each session folder —
so the whole `/results` directory can be captured as a single combined data
asset and added to a Code Ocean collection.

This builds on (and supersedes) the staging work in
[behavior_nwb_updates/](behavior_nwb_updates/) (its `IMPLEMENTATION_NOTES.md` and
`capsule.diff` remain as reference). All changes below are now applied to the
live `code/`.

---

## Decisions made this session (on top of the staging plan)

- **Registration model:** simple manual capture. The capsule produces one
  self-contained folder per session in `/results`; the user manually captures
  all of `/results` as one combined data asset and adds it to a collection. We
  explicitly did **not** use the aind-data-transfer service (risk of disturbing
  shared S3/DocDB) and did **not** build a register script. The per-session
  folders still carry full valid metadata, so individual registration remains
  possible later — this choice is forward-compatible, not a dead end.
- **`processor_full_name`** = `"Learning mFISH / V1 omFISH team"` (the project/
  team, not the capsule name — the capsule identity lives in `code_url` /
  `software_version`).
- **Per-session monitor delay must be real** (was hardcoded `0.035`, which was
  also wrong — the builder fallback is `0.0356`).
- **No silent failures** — every packaged NWB must have its metadata.
- **Robust to primary-asset schema-version drift.**

---

## 1. New module: derived-asset metadata

**New file** — [aind_metadata.py](aind_metadata.py):
- `find_primary_asset_dir` / `load_primary_metadata` — locate the raw `/data`
  asset a session's pkl came from and read its core-metadata JSONs.
- `subject_fields_for_nwb` — extract sex/genotype/strain/dob/species and compute
  ISO-8601 `age` (dob → session start) to thread into the NWB Subject.
- `build_derived_data_description` — **extracts** only the required core fields
  (`name, platform, subject_id, institution, funding_source, investigators,
  modality`, + optional `project_name/group`) from the primary
  `data_description.json` and constructs a fresh `DerivedDataDescription`.
  Deliberately does **not** whole-validate the base document, so drift in fields
  we don't use can't break it; a genuinely missing required field raises a clear
  `ValueError`.
- `build_processing` — authors a v1 `Processing`/`PipelineProcess`/`DataProcess`
  with the project team name and per-session `parameters`.
- `write_derived_metadata` — writes the two authored files (with a post-write
  re-validation gate) and copies the four inherited files **byte-for-byte**.

**Impact:** the derived metadata is authored/validated with `aind-data-schema
1.4.0` (emitted at schema 1.0.4, `data_level=derived`). Inherited files stay
identical to the primary asset. Schema drift in the primary no longer silently
corrupts or blocks a session.

---

## 2. Reworked entry point

**Changes** — [run_capsule.py](run_capsule.py):
- New per-session order: **build + validate derived metadata → package NWB →
  write metadata**. The schema-sensitive step happens before any NWB is written.
- **Output folder is named from `derived_dd.name`**, so `data_description.name`
  always equals the asset folder name (what AIND tooling expects). This fixes a
  latent mismatch in the staging version, which named the folder from the
  `/data` folder name while the metadata name came from the base JSON's `name`.
- **No silent fails:** removed the swallowing `try/except` around metadata. If
  metadata can't be authored, the session is skipped with no NWB written; if
  metadata can't be written for an already-packaged NWB, the output folder is
  removed (`shutil.rmtree`) and the session is marked failed. Invariant: **no NWB
  is ever left on disk without its metadata.** Sessions with no primary asset
  (e.g. local test data) are packaged without derived metadata but tracked in a
  `no_metadata` list and reported — not silent.
- Threads subject metadata into **both** `package_to_nwb` and
  `package_sweepstim_to_nwb` (the staging version omitted it for sweepstim).
- Wires the **real per-session monitor delay** (and HED version) returned by the
  packager into `processing.json` parameters.
- Writes **`/results/manifest.json`** — a lightweight (non-schema) index of the
  combined asset (sessions, subjects, kind, whether each has derived metadata).
  A combined multi-subject asset can't carry a single valid root
  `data_description` (AIND `subject_id` is one value), so the manifest is the
  honest root descriptor.

**Impact:** one clean derived-asset folder per session; the whole `/results`
tree (per-session folders + `manifest.json` + summary CSVs) is directly
capturable as one combined asset.

---

## Mouse allow-list — only package requested subjects (added later this session)

**Change** — [run_capsule.py](run_capsule.py): the capsule now packages only
sessions belonging to a specific set of mice, not every session under `/data`.

- New `ALLOWED_MICE` constant (default:
  `782149, 788406, 790322, 800792, 800995, 804363`). Overridable at runtime with
  the `MICE` env var (comma-separated); `MICE=all` disables filtering.
- The subject id is resolved **cheaply, before the pkl is loaded** —
  `_resolve_subject_id` reads it from the primary asset's `data_description.json`
  → `subject.json` → the primary folder-name numeric token (e.g.
  `multiplane-ophys_800792_…` → `800792`). Only if none of those yield an id does
  it fall back to the pkl's `mouse_id` after the classification load.
- Excluded mice are skipped without opening their (large) pkls, and are reported
  in the final tally as `skipped (mouse not requested)`.
- Fail-safe: if a mouse filter is active and a session's mouse cannot be
  identified at all, it is skipped (not packaged) and logged — nothing outside
  the list slips through.

**Why it matters here:** the `/data` mount contains ~15 mice
(`multiplane-ophys_<subject>_…` folders); this restricts packaging to the six
requested subjects.

**Impact:** far less wasted work (excluded sessions never load their pkl), and
`/results` contains only the requested mice.

---

## 3. Packagers: Zarr, subject fields, monitor-delay return

**Changes** — [package_to_nwb.py](package_to_nwb.py):
- Writes NWB via `NWBZarrIO` (`behavior.nwb.zarr`); sidecar naming fixed for the
  `.nwb.zarr` double extension; `build_subject` populates `strain` +
  (tz-aware) `date_of_birth`.
- **Now returns** `{output_path, monitor_delay_sec, hed_schema_version}` —
  `monitor_delay_sec` is the value measured from the photodiode
  (`build_all()['timestamp_data']['monitor_delay']`); `hed_schema_version` is the
  module constant (no more hardcoded string in `run_capsule`).

**Changes** — [sweepstim_packaging/package.py](sweepstim_packaging/package.py):
- Same Zarr + sidecar-naming changes; `Subject` gains `strain` + tz-aware
  `date_of_birth`. Returns the same dict shape, with `monitor_delay_sec=None`
  (passive SweepStim measures no monitor delay — its `processing.json` carries
  only `hed_schema_version`).

**Changes** — [summarize_sessions.py](summarize_sessions.py):
- Reads `*.nwb.zarr` directories via `NWBZarrIO`; session_id from the NWB
  identifier / asset folder; glob excludes files inside zarr trees. Cross-session
  CSV content unchanged.

**Impact:** downstream readers of these behavior NWBs must switch from
`NWBHDF5IO` to `NWBZarrIO` — the one coordinated change outside this capsule.

---

## 4. Environment

**Changes** — [../environment/Dockerfile](../environment/Dockerfile):
- Added `aind-data-schema==1.4.0`, `hdmf-zarr==0.13.0`, `zarr==2.18.3`
  (compatible with the pinned `hdmf==4.3.1`).
- **Note:** the line-1 `sha256` hash is now stale; Code Ocean regenerates it when
  the environment is edited through its UI — do not hand-edit that line.

---

## Verification done this session

Verified locally against `aind-data-schema 1.4.0` + `hdmf-zarr 0.13.0`:
- Full import chain loads (`run_capsule` → packagers → `aind_metadata` →
  `summarize_sessions`); run_capsule's imported `aind_metadata` API matches.
- `data_description.name` == asset folder name; schema 1.0.4; `data_level=derived`.
- Schema-drift tolerance (junk / old-version base still builds) and loud failure
  on a missing required field.
- End-to-end metadata sequence: 6 files written (2 authored + 4 inherited),
  inherited files byte-identical, real monitor delay + HED version in
  `processing.json`, `processor_full_name` correct.

**Not verified locally (no test data in the sandbox):** actual pkl/sync → NWB
packaging. Run the capsule in Code Ocean against a real primary asset to confirm
the full path end-to-end.

---

## Files changed

- **new** [aind_metadata.py](aind_metadata.py)
- [run_capsule.py](run_capsule.py)
- [package_to_nwb.py](package_to_nwb.py)
- [sweepstim_packaging/package.py](sweepstim_packaging/package.py)
- [summarize_sessions.py](summarize_sessions.py)
- [../environment/Dockerfile](../environment/Dockerfile)
- staging [behavior_nwb_updates/](behavior_nwb_updates/) is now superseded (kept
  for reference)
