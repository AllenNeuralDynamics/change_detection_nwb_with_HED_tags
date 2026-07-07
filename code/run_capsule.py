"""Capsule entry point.

Discovers every session under /data and packages supported session types into
HED-annotated NWB-Zarr files + a BIDS-style events sidecar, emitting one AIND
*derived data asset* folder per session under /results.

Two camstim file-naming conventions are supported for the same kind of session:
  - ``<id>_stim.pkl`` paired with ``<id>_sync.h5`` (older), and
  - ``<id>.pkl``      paired with ``<id>_<timestamp>.h5`` (newer).

Not every pkl under /data is the same task type. Sessions are classified by
structure before packaging:
- change-detection sessions are packaged via ``package_to_nwb.py``;
- passive SweepStim sessions are packaged via the standalone
    ``sweepstim_packaging/`` module;
- unknown structures are logged and skipped.

Per-session output layout (one AIND derived asset per session):

    /results/<primary-asset-name>_behavior-nwb_<date>_<time>/
        behavior.nwb.zarr/         NWB written as Zarr
        behavior.events.json       HED/BIDS column sidecar
        data_description.json      authored + validated (name == folder name)
        processing.json            authored + validated (real monitor delay)
        subject/procedures/session/rig.json   inherited verbatim from primary

Plus /results/manifest.json + the cross-session summary CSVs at the root, so the
whole /results directory can be captured as a single combined data asset.

Robustness contract (no silent fails): for any session with an identifiable
primary AIND asset, we author + validate the derived metadata *before* writing
the NWB, and if metadata cannot be written for a packaged session we remove its
output. An NWB is therefore never left on disk without its metadata; sessions
that cannot get metadata are reported as failures.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import shutil
from pathlib import Path

from package_to_nwb import package_to_nwb
from summarize_sessions import summarize_results
from sweepstim_packaging import package_sweepstim_to_nwb
from aind_metadata import (
    find_primary_asset_dir,
    load_primary_metadata,
    subject_fields_for_nwb,
    build_derived_data_description,
    build_processing,
    write_derived_metadata,
    PROCESS_LABEL,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# NWB filename (modality token) inside each derived asset. AIND convention:
# <modality>.nwb.zarr at the asset root.
NWB_FILENAME = "behavior.nwb.zarr"

# Only sessions for these subject (mouse) IDs are packaged; all others are
# skipped. Override at runtime with the MICE env var (comma-separated), or set
# MICE=all to package every discovered session.
ALLOWED_MICE = ("782149", "788406", "790322", "800792", "800995", "804363")


def _get_allowed_mice() -> set[str]:
    """Resolve the set of subject (mouse) ids to package.

    The ``MICE`` env var (comma-separated) overrides the ``ALLOWED_MICE``
    default; ``MICE=all`` (or empty) disables filtering entirely (package
    everything). An empty returned set means "no filter".
    """
    env = os.environ.get("MICE")
    if env is not None:
        if env.strip().lower() in ("", "all"):
            return set()
        return {tok.strip() for tok in env.split(",") if tok.strip()}
    return {str(m) for m in ALLOWED_MICE}


def _resolve_subject_id(primary_dir, primary_meta) -> str | None:
    """Best-effort subject id from the primary asset — cheap, no pkl load.

    Tries data_description.json, then subject.json, then the primary folder name
    (which encodes the subject id as a numeric token). Returns None if it can't
    be determined without opening the pkl.
    """
    dd = primary_meta.get("data_description.json") or {}
    if dd.get("subject_id"):
        return str(dd["subject_id"])
    subj = primary_meta.get("subject.json") or {}
    if subj.get("subject_id"):
        return str(subj["subject_id"])
    if primary_dir is not None:
        for tok in primary_dir.name.split("_"):
            if tok.isdigit() and len(tok) >= 6:
                return tok
    return None


def _mouse_id_from_pkl_data(data) -> str | None:
    """Subject id from an already-loaded camstim pkl (change-detection/sweepstim)."""
    try:
        params = ((data.get("items") or {}).get("behavior") or {}).get("params") or {}
        if params.get("mouse_id") not in (None, ""):
            return str(params["mouse_id"])
    except Exception:
        pass
    if data.get("mouseid") not in (None, ""):
        return str(data["mouseid"])
    return None


def _capsule_code_version() -> str:
    """Best-effort capsule/code version for processing provenance.

    Prefers the git commit of the checked-out code; falls back to an env var or
    'unknown'. Never raises — provenance is nice-to-have, not run-critical.
    """
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip() or "unknown"
    except Exception:
        return os.environ.get("CODE_VERSION", "unknown")


def discover_sessions(data_dir: Path) -> dict[str, Path]:
    """Map session id -> stimulus pkl path, across both naming conventions.

    Accepts ``<id>_stim.pkl`` and bare ``<id>.pkl`` (numeric id only, to avoid
    picking up unrelated pickles). When both exist for the same id, the
    canonical ``*_stim.pkl`` wins.
    """
    by_sid: dict[str, Path] = {}
    for pkl in sorted(data_dir.rglob("*.pkl")):
        if pkl.name.endswith("_stim.pkl"):
            sid = pkl.name[: -len("_stim.pkl")]
        elif pkl.stem.isdigit():
            sid = pkl.stem
        else:
            continue  # not an Allen session pkl
        # Prefer a *_stim.pkl over a bare <id>.pkl for the same id.
        if sid in by_sid and by_sid[sid].name.endswith("_stim.pkl"):
            continue
        by_sid[sid] = pkl
    return by_sid


def find_sync(pkl: Path, sid: str) -> Path | None:
    """Locate the sync .h5 next to a pkl, across both naming conventions."""
    cands = sorted(pkl.parent.glob("*_sync.h5")) or sorted(pkl.parent.glob(f"{sid}_*.h5"))
    return cands[0] if cands else None


def classify_session(data: dict) -> tuple[str, str]:
    """Classify a loaded camstim pkl by structure.

    Detection is *positive for change-detection* — a session is only treated as
    change-detection when it has the defining structure (a ``behavior`` item
    with a non-empty ``trial_log`` and a stimulus dict). This guarantees we never
    accidentally skip a real behavior session; anything else is reported as an
    unsupported type rather than crashing the pipeline.

    Returns ``(kind, detail)`` where ``kind`` is one of:
      - ``"change_detection"`` — package it.
      - ``"sweepstim"``        — passive SweepStim session; package via
        standalone ``sweepstim_packaging`` module.
      - ``"unknown"``          — neither signature; skip and report loudly.
    """
    items = data.get("items") or {}
    beh = items.get("behavior")

    # Change-detection signature: behavior item + actual trials.
    if isinstance(beh, dict) and beh.get("trial_log"):
        n = len(beh["trial_log"])
        stim = beh.get("stimuli")
        if isinstance(stim, dict) and stim:
            return "change_detection", f"behavior.trial_log={n} trials, stimuli={list(stim)}"
        # Behavior + trials but no stimulus dict: still a behavior session (don't
        # skip it); packaging will surface any problem.
        return "change_detection", f"behavior.trial_log={n} trials, stimuli={stim!r}"

    # Passive SweepStim signature: a top-level stimuli *list* and/or a foraging
    # item, with no behavior trial_log.
    if isinstance(data.get("stimuli"), list) and data["stimuli"]:
        return "sweepstim", (f"top-level stimuli list ({len(data['stimuli'])} entries), "
                             f"no behavior trial_log; items={list(items)}")
    if "foraging" in items:
        return "sweepstim", f"items.foraging present, no behavior trial_log; items={list(items)}"

    return "unknown", f"items={list(items)}, top-level stimuli={type(data.get('stimuli')).__name__}"


def _read_pkl_start_time(pkl: Path):
    """Best-effort read of the session start_time from the pkl (may be None)."""
    try:
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        return d.get("start_time")
    except Exception:
        return None


def _write_manifest(results_dir: Path, entries: list[dict]) -> None:
    """Write a lightweight (non-schema) manifest describing the combined asset.

    A combined asset spans multiple subjects, so it cannot carry a single valid
    AIND data_description at its root; this manifest is the honest, self-
    describing index of what the asset contains.
    """
    manifest = {
        "description": ("Combined behavior-NWB derived assets — one folder per "
                        "session. Each folder is a self-contained AIND derived "
                        "data asset (NWB-Zarr + data_description/processing + "
                        "inherited subject/procedures/session/rig)."),
        "process_label": PROCESS_LABEL,
        "generated": datetime.datetime.now().astimezone().isoformat(),
        "n_sessions": len(entries),
        "sessions": entries,
    }
    path = results_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s (%d session(s)).", path, len(entries))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(DATA_DIR)
    if not sessions:
        raise SystemExit(f"ERROR: no session *.pkl found under {DATA_DIR}")

    logging.info("Found %d candidate session(s) under %s.", len(sessions), DATA_DIR)

    packaged_cd, packaged_sweepstim = [], []
    no_sync, skipped_type, skipped_mouse, failed, no_metadata = [], [], [], [], []
    manifest_entries: list[dict] = []
    code_version = _capsule_code_version()

    allowed_mice = _get_allowed_mice()
    if allowed_mice:
        logging.info("Restricting to %d requested mouse id(s): %s",
                     len(allowed_mice), ", ".join(sorted(allowed_mice)))
    else:
        logging.info("No mouse filter set (MICE=all) — packaging all sessions.")

    for sid, pkl in sorted(sessions.items()):
        sync = find_sync(pkl, sid)
        if sync is None:
            logging.warning("No sync .h5 next to %s — skipping.", pkl)
            no_sync.append(sid)
            continue

        # ── Locate the primary (raw) asset this session came from, and load its
        #    AIND core metadata JSONs. Done first so we can filter by mouse id
        #    (from the cheap metadata / folder name) BEFORE loading the large pkl.
        primary_dir = find_primary_asset_dir(pkl, DATA_DIR)
        primary_meta = load_primary_metadata(primary_dir)
        primary_name = primary_dir.name if primary_dir else None
        base_dd_json = primary_meta.get("data_description.json")

        subject_id = _resolve_subject_id(primary_dir, primary_meta)
        if allowed_mice and subject_id is not None and subject_id not in allowed_mice:
            logging.info("Skipping session %s — mouse %s not in the requested list.",
                         sid, subject_id)
            skipped_mouse.append((sid, subject_id))
            continue

        # Classify by structure before packaging so non-change-detection
        # sessions (e.g. passive SweepStim) are handled correctly, not crashed.
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            kind, detail = classify_session(data)
            if subject_id is None:
                subject_id = _mouse_id_from_pkl_data(data)
            del data  # free the (potentially large) pkl before packaging reloads it
        except Exception:
            logging.exception("Could not read/classify %s — skipping.", pkl)
            failed.append(sid)
            continue

        # Enforce the mouse filter now that we have the best available subject id
        # (metadata, folder name, or pkl mouse_id). If a filter is active and we
        # still can't identify the mouse, skip loudly rather than package it.
        if allowed_mice:
            if subject_id is None:
                logging.warning("Skipping session %s — could not determine its "
                                "mouse id to match the requested list.", sid)
                skipped_mouse.append((sid, None))
                continue
            if subject_id not in allowed_mice:
                logging.info("Skipping session %s — mouse %s not in the requested "
                             "list.", sid, subject_id)
                skipped_mouse.append((sid, subject_id))
                continue

        if kind not in ("change_detection", "sweepstim"):
            logging.warning("Skipping session %s — unsupported type (%s: %s).",
                            sid, kind, detail)
            skipped_type.append((sid, kind))
            continue

        creation_time = datetime.datetime.now().astimezone()

        # ── Author + validate the derived data_description FIRST. This is the
        #    schema-version-sensitive step; doing it before writing any NWB means
        #    a schema problem fails the session loudly with no orphaned output.
        #    The output folder is named from derived_dd.name so that
        #    data_description.name always equals the folder name.
        derived_dd = None
        if primary_name and base_dd_json is not None:
            try:
                derived_dd = build_derived_data_description(base_dd_json, creation_time)
            except Exception:
                logging.exception(
                    "Session %s: could not author a derived data_description from "
                    "primary asset %s — SKIPPING (no NWB written).",
                    sid, primary_name)
                failed.append(sid)
                continue
            out_dir = RESULTS_DIR / derived_dd.name
        else:
            # No identifiable primary asset (e.g. local test data without AIND
            # metadata). We still package the NWB, but there is no derived
            # metadata to author; this is tracked and reported, not silent.
            logging.warning(
                "Session %s: no primary AIND asset/data_description found under %s "
                "— packaging NWB WITHOUT derived metadata.", sid, primary_dir)
            out_dir = RESULTS_DIR / f"{sid}_{PROCESS_LABEL}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / NWB_FILENAME

        logging.info("=" * 62)
        logging.info("Packaging session %s (%s: %s)", sid, kind, detail)
        logging.info("  pkl     : %s", pkl)
        logging.info("  sync    : %s", sync)
        logging.info("  primary : %s", primary_name)
        logging.info("  out     : %s", out)
        logging.info("=" * 62)

        # Thread AIND subject metadata (sex/genotype/age/strain/dob/species) from
        # the primary subject.json into the NWB itself.
        start_dt = _read_pkl_start_time(pkl)
        nwb_meta = subject_fields_for_nwb(primary_meta, start_dt or creation_time)

        # ── Package the NWB. On any failure, remove the partial output so we
        #    never leave a half-written asset behind.
        try:
            if kind == "change_detection":
                pkg = package_to_nwb(str(pkl), str(sync), str(out), metadata=nwb_meta)
            else:  # sweepstim
                pkg = package_sweepstim_to_nwb(str(pkl), str(sync), str(out), metadata=nwb_meta)
        except Exception:
            logging.exception("FAILED to package session %s — removing %s.", sid, out_dir)
            shutil.rmtree(out_dir, ignore_errors=True)
            failed.append(sid)
            continue

        # ── Author + write the derived metadata for sessions that have a primary
        #    asset. If this fails for a packaged session we remove the NWB output
        #    (invariant: no NWB on disk without its metadata) and mark it failed.
        if derived_dd is not None:
            try:
                params = {"hed_schema_version": pkg.get("hed_schema_version")}
                if pkg.get("monitor_delay_sec") is not None:
                    params["monitor_delay_sec"] = pkg["monitor_delay_sec"]
                notes = ("camstim pkl+sync -> HED-annotated NWB-Zarr (behavior + stimulus)"
                         if kind == "change_detection"
                         else "camstim pkl+sync -> HED-annotated NWB-Zarr (passive SweepStim)")
                processing = build_processing(
                    primary_name=primary_name,
                    out_asset_name=out_dir.name,
                    start=creation_time,
                    end=datetime.datetime.now().astimezone(),
                    code_version=code_version,
                    parameters=params,
                    notes=notes,
                )
                summary = write_derived_metadata(out_dir, derived_dd, processing, primary_dir)
                logging.info("  metadata: authored=%s inherited=%s missing=%s",
                             summary["authored"], summary["inherited"], summary["missing"])
            except Exception:
                logging.exception(
                    "Derived metadata FAILED for session %s — removing NWB output "
                    "at %s so no NWB is left without metadata.", sid, out_dir)
                shutil.rmtree(out_dir, ignore_errors=True)
                failed.append(sid)
                continue
        else:
            no_metadata.append(sid)

        # Success: record it.
        (packaged_cd if kind == "change_detection" else packaged_sweepstim).append(sid)
        manifest_entries.append({
            "session_id": sid,
            "asset_folder": out_dir.name,
            "kind": kind,
            "subject_id": subject_id or sid,
            "has_derived_metadata": derived_dd is not None,
        })

    logging.info("=" * 62)
    packaged_total = len(packaged_cd) + len(packaged_sweepstim)
    logging.info("Done. %d packaged (%d change-detection, %d sweepstim), "
                 "%d skipped (mouse not requested), %d skipped (unsupported type), "
                 "%d skipped (no sync), %d failed.",
                 packaged_total, len(packaged_cd), len(packaged_sweepstim),
                 len(skipped_mouse), len(skipped_type), len(no_sync), len(failed))
    if skipped_mouse:
        logging.info("Skipped (mouse not requested): %s",
                     ", ".join(f"{s} [{m}]" for s, m in skipped_mouse))
    if skipped_type:
        logging.info("Skipped (unsupported type): %s",
                     ", ".join(f"{s} [{k}]" for s, k in skipped_type))
    if no_sync:
        logging.warning("Skipped (no sync): %s", ", ".join(no_sync))
    if no_metadata:
        logging.warning("Packaged WITHOUT derived metadata (no primary asset): %s",
                        ", ".join(no_metadata))
    if failed:
        logging.warning("Failed to package: %s", ", ".join(failed))
    logging.info("Results in %s", RESULTS_DIR)

    # Manifest describing the combined asset (written whenever anything packaged).
    if manifest_entries:
        _write_manifest(RESULTS_DIR, manifest_entries)

    # Build the cross-session summary tables (session_metrics.csv +
    # session_task_parameters.csv) from the NWBs we just wrote, and copy them
    # into /results as a convenience index across the per-session derived assets.
    # sidecar=False: the per-session <id>.metadata.json is now superseded by the
    # AIND schema JSONs written into each derived asset. A summary hiccup must
    # not fail the packaging run.
    if packaged_total:
        try:
            summarize_results(RESULTS_DIR, sidecar=False, mirror_csv_dir=RESULTS_DIR)
        except Exception:
            logging.exception("Session summary step failed "
                              "(NWBs were still written to %s).", RESULTS_DIR)
    else:
        logging.info("No NWBs packaged — skipping session summary.")


if __name__ == "__main__":
    main()
