# Behavior NWB capsule — AIND-compliance changes (implementation + handoff)

Capsule: **Change-detection NWB + HED packaging** (CO `dea06020-819d-4f61-8d2b-89826c7403a9`,
slug 9485660, GitHub `AllenNeuralDynamics/change_detection_nwb_with_HED_tags`).

These changes make the capsule emit **one AIND derived data asset per session**
directly, instead of a flat pool of `<session_id>.nwb` + custom sidecars.

## What each derived asset now looks like

```
<primary-asset-name>_behavior-nwb_<YYYY-MM-DD>_<HH-MM-SS>/
├── behavior.nwb.zarr/          NWB written as Zarr (hdmf-zarr), <modality>.nwb.zarr at root
├── behavior.events.json        HED/BIDS column sidecar (kept — colleague request)
├── data_description.json       REGENERATED: DerivedDataDescription (validated)
├── processing.json             NEW: Processing (validated)
├── subject.json                inherited byte-for-byte from primary asset
├── procedures.json             inherited byte-for-byte
├── session.json                inherited byte-for-byte
└── rig.json                    inherited byte-for-byte
```

`<primary-asset-name>` = the mounted `/data/<name>` folder the session's pkl came
from (e.g. `multiplane-ophys_800792_2025-08-26_12-30-21`).

## Design decisions realized (all confirmed with you)

| Decision | Implementation |
|---|---|
| Fix at source | The capsule produces the compliant structure; no post-hoc converter. |
| Schema v1.x | `aind-data-schema==1.4.0`. Keeps `session`/`rig` (no acquisition/instrument conversion). |
| Base metadata source | JSON files inside the mounted `/data/` asset (no DocDB / no new API). |
| Inherited files | Copied **byte-for-byte** — never round-tripped through the library, so they stay identical to the multiplane-ophys parent. |
| Validate authored only | `data_description` + `processing` built as pydantic models (validate on construction) and re-read as their proper class after writing (post-write gate). |
| NWB subject fields | `sex`, `genotype`, `age` (ISO-8601, from dob→session-start), `date_of_birth`, `strain`, `species` threaded into the NWB. |
| NWB IO | `hdmf-zarr` `NWBZarrIO`, file `behavior.nwb.zarr`. |
| Process label | asset token **`behavior-nwb`** (hyphen: the v1 validator forbids `_` in the token); `behavior_nwb` is used where underscores are allowed (DataProcess params). |
| BIDS sidecar | kept (`behavior.events.json`). |
| data_description version | emitted at **1.0.4** (what 1.4.0 produces). Inherited files keep their own versions (subject 1.0.3, procedures 1.2.1, rig 1.0.1) since they're copied. |

## Files changed

1. **`code/aind_metadata.py`** — NEW module. Primary-asset discovery, primary
   metadata loading, `subject_fields_for_nwb`, derived-name builder, and
   `write_derived_metadata` (authored+validated / inherited-verbatim).
2. **`code/run_capsule.py`** — per-session block now: find primary asset → load
   its metadata → build derived-asset folder name → thread subject fields into
   the NWB → write `behavior.nwb.zarr` → write derived metadata. Summary step
   switched to `sidecar=False` (schema JSONs supersede the custom
   `<id>.metadata.json`).
3. **`code/package_to_nwb.py`** — `NWBHDF5IO`→`NWBZarrIO`; `build_subject` adds
   strain/date_of_birth (dob coerced to tz-aware datetime for pynwb); sidecar
   naming fixed for the `.nwb.zarr` double extension.
4. **`code/sweepstim_packaging/package.py`** — same Zarr + sidecar-naming change,
   for passive SweepStim sessions.
5. **`code/summarize_sessions.py`** — reads `*.nwb.zarr` dirs (via `NWBZarrIO`),
   session_id from the NWB identifier / asset folder; glob excludes files inside
   zarr trees. Cross-session CSVs unchanged in content.
6. **`environment/Dockerfile`** — add `aind-data-schema==1.4.0`,
   `hdmf-zarr==0.13.0`, `zarr==2.18.3` (compatible with the pinned `hdmf==4.3.1`).

## Dry-run verification (done here, not just proposed)

Ran the full `run_capsule.py` on a real scenario: the change-detection test pkl/sync
placed inside a primary-asset folder carrying the **real** `multiplane-ophys_800792`
metadata JSONs pulled from `aind-open-data`. Result:
- 1 change-detection session packaged, 0 failed.
- Derived asset created with the correct name; all 6 metadata files present
  (2 authored + 4 inherited); inherited files byte-identical to source.
- NWB-Zarr read back cleanly; subject fields present (`sex=F`,
  genotype, `age=P170D` from dob→pkl start_time, `date_of_birth`, species).
- Summary step read the Zarr and wrote the cross-session CSVs.

Two fixes were found *by* the dry-run (both now applied): pynwb requires a
tz-aware `datetime` for `date_of_birth`; the `.events.json` sidecar name must be
derived by stripping all suffixes (not `with_suffix`, which mishandles the
`.nwb.zarr` double extension).

## Notes for the in-capsule agent / reviewer

- **Downstream readers must switch to `NWBZarrIO`.** Any capsule that reads these
  behavior NWBs with `NWBHDF5IO` will break — this is the one coordinated change
  outside this capsule.
- **`processor_full_name`** in `processing.json` currently defaults to
  `"AIND Behavior"` in `run_capsule` (was "Marina Garrett" in the standalone
  test). Set it to whatever provenance you want recorded.
- **HED cache**: in this sandbox hedtools couldn't write `~/.hedtools`; irrelevant
  in Code Ocean (writable home). No code change needed.
- The `monitor_delay_sec`/`hed_schema_version` recorded in `processing.json`
  parameters are currently hardcoded (0.035 / "8.3.0"); wire them to the real
  values if you want them exact per-session (the packager already computes the
  actual monitor delay — it's logged).
