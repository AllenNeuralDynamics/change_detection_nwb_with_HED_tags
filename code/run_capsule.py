"""Capsule entry point.

Discovers every session under /data and packages each **change-detection**
session into a HED-annotated NWB file + BIDS-style events sidecar JSON written
to /results.

Two camstim file-naming conventions are supported for the same kind of session:
  - ``<id>_stim.pkl`` paired with ``<id>_sync.h5`` (older), and
  - ``<id>.pkl``      paired with ``<id>_<timestamp>.h5`` (newer).

Not every pkl under /data is a change-detection session — passive **SweepStim**
sessions (sync_square / foraging items, a top-level ``stimuli`` *list*, no
behavior ``trial_log``) live alongside them and are NOT handled by this
pipeline. Each session is classified by structure before packaging:
change-detection sessions are packaged; SweepStim / unrecognized sessions are
logged and skipped. A session that errors during packaging is also logged and
skipped, so nothing aborts the whole batch.

Output naming: <id>.nwb and <id>.events.json, where <id> is the numeric session
id (the pkl filename with any trailing "_stim" and the ".pkl" removed).
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from package_to_nwb import package_to_nwb

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))


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
      - ``"sweepstim"``        — passive SweepStim session; skip (not supported).
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

    packaged, no_sync, skipped_type, failed = [], [], [], []
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

        if kind != "change_detection":
            logging.warning("Skipping session %s — not a change-detection session "
                            "(%s: %s).", sid, kind, detail)
            skipped_type.append((sid, kind))
            continue

        out = RESULTS_DIR / f"{sid}.nwb"
        logging.info("=" * 62)
        logging.info("Packaging change-detection session %s (%s)", sid, detail)
        logging.info("  pkl  : %s", pkl)
        logging.info("  sync : %s", sync)
        logging.info("  out  : %s", out)
        logging.info("=" * 62)
        try:
            package_to_nwb(str(pkl), str(sync), str(out))
            packaged.append(sid)
        except Exception:
            logging.exception("FAILED to package session %s — skipping.", sid)
            failed.append(sid)

    logging.info("=" * 62)
    logging.info("Done. %d packaged, %d skipped (unsupported type), "
                 "%d skipped (no sync), %d failed.",
                 len(packaged), len(skipped_type), len(no_sync), len(failed))
    if skipped_type:
        logging.info("Skipped (unsupported type): %s",
                     ", ".join(f"{s} [{k}]" for s, k in skipped_type))
    if no_sync:
        logging.warning("Skipped (no sync): %s", ", ".join(no_sync))
    if failed:
        logging.warning("Failed to package: %s", ", ".join(failed))
    logging.info("Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
