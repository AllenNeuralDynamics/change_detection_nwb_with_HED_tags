"""Capsule entry point.

Discovers every session under /data and packages supported session types into
HED-annotated NWB files + BIDS-style events sidecar JSON written to /results.

Two camstim file-naming conventions are supported for the same kind of session:
  - ``<id>_stim.pkl`` paired with ``<id>_sync.h5`` (older), and
  - ``<id>.pkl``      paired with ``<id>_<timestamp>.h5`` (newer).

Not every pkl under /data is the same task type. Sessions are classified by
structure before packaging:
- change-detection sessions are packaged via ``package_to_nwb.py``;
- passive SweepStim sessions are packaged via the standalone
    ``sweepstim_packaging/`` module;
- unknown structures are logged and skipped.

Any session that errors during packaging is logged and skipped, so nothing
aborts the whole batch.

Output naming: <id>.nwb and <id>.events.json, where <id> is the numeric session
id (the pkl filename with any trailing "_stim" and the ".pkl" removed).
"""
from __future__ import annotations

import datetime
import logging
import os
import pickle
from pathlib import Path

from package_to_nwb import package_to_nwb
from summarize_sessions import summarize_results
from sweepstim_packaging import package_sweepstim_to_nwb
from aind_metadata import (
    find_primary_asset_dir,
    load_primary_metadata,
    subject_fields_for_nwb,
    build_derived_name,
    write_derived_metadata,
    PROCESS_LABEL,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# NWB filename (modality token) inside each derived asset. AIND convention:
# <modality>.nwb.zarr at the asset root.
NWB_FILENAME = "behavior.nwb.zarr"


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(DATA_DIR)
    if not sessions:
        raise SystemExit(f"ERROR: no session *.pkl found under {DATA_DIR}")

    logging.info("Found %d candidate session(s) under %s.", len(sessions), DATA_DIR)

    packaged_cd, packaged_sweepstim, no_sync, skipped_type, failed = [], [], [], [], []
    for sid, pkl in sorted(sessions.items()):
        sync = find_sync(pkl, sid)
        if sync is None:
            logging.warning("No sync .h5 next to %s — skipping.", pkl)
            no_sync.append(sid)
            continue

        # Classify by structure before packaging so non-change-detection
        # sessions (e.g. passive SweepStim) are skipped cleanly, not crashed.
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            kind, detail = classify_session(data)
            del data  # free the (potentially large) pkl before packaging reloads it
        except Exception:
            logging.exception("Could not read/classify %s — skipping.", pkl)
            failed.append(sid)
            continue

        # ── Locate the primary (raw) asset this session came from, and load its
        #    AIND core metadata JSONs. The behavior NWB is a *derived* asset:
        #    named <primary-asset-name>_<PROCESS_LABEL>_<date>_<time>, with the
        #    NWB-Zarr + regenerated data_description/processing + inherited
        #    subject/procedures/session/rig at its root.
        primary_dir = find_primary_asset_dir(pkl, DATA_DIR)
        primary_meta = load_primary_metadata(primary_dir)
        primary_name = primary_dir.name if primary_dir else None

        if primary_name and "data_description.json" in primary_meta:
            creation_time = datetime.datetime.now().astimezone()
            out_dir = RESULTS_DIR / build_derived_name(primary_name, creation_time)
        else:
            # Fallback: no identifiable primary asset (e.g. local test data
            # without AIND metadata). Keep a session-id folder so the run still
            # produces output; derived metadata is skipped for this session.
            logging.warning(
                "Session %s: no primary AIND asset/data_description found under "
                "%s — writing NWB without derived metadata.", sid, primary_dir)
            creation_time = datetime.datetime.now().astimezone()
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
        try:
            # Thread AIND subject metadata (sex/genotype/age/strain/dob/species)
            # from the primary subject.json into the NWB itself.
            start_dt = None
            try:
                with open(pkl, "rb") as f:
                    _d = pickle.load(f, encoding="latin1")
                start_dt = _d.get("start_time")
                del _d
            except Exception:
                pass
            nwb_meta = subject_fields_for_nwb(
                primary_meta, start_dt or creation_time)

            if kind == "change_detection":
                package_to_nwb(str(pkl), str(sync), str(out), metadata=nwb_meta)
                packaged_cd.append(sid)
            elif kind == "sweepstim":
                package_sweepstim_to_nwb(str(pkl), str(sync), str(out))
                packaged_sweepstim.append(sid)
            else:
                logging.warning("Skipping session %s — unsupported type "
                                "(%s: %s).", sid, kind, detail)
                skipped_type.append((sid, kind))

            # ── Write the derived-asset metadata (validated authored files +
            #    verbatim inherited files) next to the NWB, only for packaged
            #    sessions that have an identifiable primary asset.
            if (kind in ("change_detection", "sweepstim")
                    and primary_name and "data_description.json" in primary_meta):
                try:
                    summary = write_derived_metadata(
                        out_dir=out_dir,
                        primary_dir=primary_dir,
                        primary_meta=primary_meta,
                        primary_name=primary_name,
                        creation_time=creation_time,
                        end_time=datetime.datetime.now().astimezone(),
                        code_version=_capsule_code_version(),
                        parameters={"monitor_delay_sec": 0.035,
                                    "hed_schema_version": "8.3.0"},
                    )
                    logging.info("  metadata: authored=%s inherited=%s missing=%s",
                                 summary["authored"], summary["inherited"],
                                 summary["missing"])
                except Exception:
                    logging.exception(
                        "Derived metadata FAILED for %s — NWB was still "
                        "written to %s.", sid, out_dir)
        except Exception:
            logging.exception("FAILED to package session %s — skipping.", sid)
            failed.append(sid)

    logging.info("=" * 62)
    packaged_total = len(packaged_cd) + len(packaged_sweepstim)
    logging.info("Done. %d packaged (%d change-detection, %d sweepstim), "
                 "%d skipped (unsupported type), %d skipped (no sync), %d failed.",
                 packaged_total, len(packaged_cd), len(packaged_sweepstim),
                 len(skipped_type), len(no_sync), len(failed))
    if skipped_type:
        logging.info("Skipped (unsupported type): %s",
                     ", ".join(f"{s} [{k}]" for s, k in skipped_type))
    if no_sync:
        logging.warning("Skipped (no sync): %s", ", ".join(no_sync))
    if failed:
        logging.warning("Failed to package: %s", ", ".join(failed))
    logging.info("Results in %s", RESULTS_DIR)

    # Build the cross-session summary tables (session_metrics.csv +
    # session_task_parameters.csv) from the NWBs we just wrote, and copy them
    # into /results as a convenience index across the per-session derived assets.
    # sidecar=False: the per-session <id>.metadata.json is now superseded by the
    # AIND schema JSONs (data_description/processing/subject/...) written into
    # each derived asset. A summary hiccup must not fail the packaging run.
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
