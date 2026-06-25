"""Capsule entry point.

Discovers every session under /data (any subfolder, or /data itself, that
holds exactly one *_stim.pkl and one *_sync.h5) and packages each into a
HED-annotated NWB file + BIDS-style events sidecar JSON written to /results.

Output naming: <id>.nwb and <id>.events.json, where <id> is the stim pkl
filename with the trailing "_stim" removed (e.g. 1464696201_stim.pkl ->
1464696201.nwb).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from package_to_nwb import package_to_nwb

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(DATA_DIR.rglob("*_stim.pkl"))
    if not pkl_files:
        raise SystemExit(f"ERROR: no *_stim.pkl found under {DATA_DIR}")

    logging.info("Found %d session(s) to package.", len(pkl_files))

    for pkl in pkl_files:
        stem = pkl.name[: -len("_stim.pkl")]
        sync_matches = sorted(pkl.parent.glob("*_sync.h5"))
        if not sync_matches:
            logging.warning("No *_sync.h5 next to %s — skipping.", pkl)
            continue
        sync = sync_matches[0]
        out = RESULTS_DIR / f"{stem}.nwb"

        logging.info("=" * 62)
        logging.info("Packaging session %s", stem)
        logging.info("  pkl  : %s", pkl)
        logging.info("  sync : %s", sync)
        logging.info("  out  : %s", out)
        logging.info("=" * 62)
        package_to_nwb(str(pkl), str(sync), str(out))

    logging.info("All sessions packaged. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
