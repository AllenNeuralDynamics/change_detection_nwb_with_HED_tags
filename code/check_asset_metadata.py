"""Cross-check each /data asset against its metadata and packaged NWB.

For every asset folder under DATA_DIR we gather the subject id, datetime, and
session type from four independent sources:

  1. the asset *folder name*  (e.g. multiplane-ophys_<subj>_<date>_<time>)
  2. AIND metadata JSON       (subject.json / session.json / data_description.json)
  3. the stim *pkl*           (mouse_id, start_time, stage)
  4. the packaged *NWB*       (results/<session_id>.nwb)

and report every place they disagree.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import ndx_change_detection_task  # noqa: F401
import ndx_events  # noqa: F401
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

FOLDER_RE = re.compile(
    r"^(?P<platform>.+?)_(?P<subj>\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$")


def load_json(path: Path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def find_pkl(asset: Path):
    for pkl in sorted(asset.rglob("*.pkl")):
        if pkl.name.endswith("_stim.pkl"):
            return pkl, pkl.name[: -len("_stim.pkl")]
        if pkl.stem.isdigit():
            return pkl, pkl.stem
    return None, None


def find_sync(pkl: Path, sid: str):
    cands = sorted(pkl.parent.glob("*_sync.h5")) or sorted(pkl.parent.glob(f"{sid}_*.h5"))
    return cands[0] if cands else None


def pkl_meta(pkl: Path) -> dict:
    try:
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        beh = d.get("items", {}).get("behavior")
        if beh is None:
            return {"not_behavior": True}
        params = beh.get("params", {})
        return {
            "mouse_id": str(params.get("mouse_id", "")) or None,
            "stage": params.get("stage"),
            "start_time": d.get("start_time"),
        }
    except Exception as e:
        return {"pkl_error": str(e)}


def nwb_meta(path: Path) -> dict:
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        subj = nwb.subject
        params = nwb.lab_meta_data.get("task_parameters")
        return {
            "subject_id": str(subj.subject_id) if subj and subj.subject_id else None,
            "sex": getattr(subj, "sex", None) if subj else None,
            "genotype": getattr(subj, "genotype", None) if subj else None,
            "acquisition_date": nwb.session_start_time,
            "session_type": getattr(params, "session_type", None) if params else None,
        }


def parse_iso(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    assets = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"Scanning {len(assets)} asset folder(s) under {DATA_DIR}\n")

    discrepancies: list[str] = []

    for asset in assets:
        name = asset.name
        D = []  # per-asset discrepancy lines

        m = FOLDER_RE.match(name)
        folder_subj = m.group("subj") if m else None
        folder_dt = None
        if m:
            folder_dt = dt.datetime.strptime(
                f"{m.group('date')} {m.group('time')}", "%Y-%m-%d %H-%M-%S")

        subj_j = load_json(asset / "subject.json") or {}
        sess_j = load_json(asset / "session.json") or {}
        dd_j = load_json(asset / "data_description.json") or {}

        pkl, sid = find_pkl(asset)
        sync = find_sync(pkl, sid) if pkl else None
        pm = pkl_meta(pkl) if pkl else {}

        nwb_path = RESULTS_DIR / f"{sid}.nwb" if sid else None
        packaged = nwb_path is not None and nwb_path.exists()
        nm = nwb_meta(nwb_path) if packaged else {}

        # ---- subject_id across all sources ----
        subj_sources = {
            "folder": folder_subj,
            "subject.json": str(subj_j.get("subject_id")) if subj_j.get("subject_id") else None,
            "session.json": str(sess_j.get("subject_id")) if sess_j.get("subject_id") else None,
            "data_description": str(dd_j.get("subject_id")) if dd_j.get("subject_id") else None,
            "pkl.mouse_id": pm.get("mouse_id"),
            "nwb": nm.get("subject_id"),
        }
        present = {k: v for k, v in subj_sources.items() if v}
        if len(set(present.values())) > 1:
            D.append("  subject_id mismatch: "
                     + ", ".join(f"{k}={v}" for k, v in present.items()))

        # ---- data_description.name vs folder name ----
        if dd_j.get("name") and dd_j["name"] != name:
            D.append(f"  data_description.name={dd_j['name']!r} != folder={name!r}")

        # ---- session_type across session.json / pkl.stage / nwb ----
        st_sources = {
            "session.json": sess_j.get("session_type"),
            "pkl.stage": pm.get("stage"),
            "nwb": nm.get("session_type"),
        }
        st_present = {k: v for k, v in st_sources.items() if v}
        if len(set(st_present.values())) > 1:
            D.append("  session_type mismatch: "
                     + ", ".join(f"{k}={v}" for k, v in st_present.items()))

        # ---- datetime: folder vs session.json vs pkl vs nwb ----
        sess_dt = parse_iso(sess_j.get("session_start_time"))
        nwb_dt = nm.get("acquisition_date")
        pkl_dt = pm.get("start_time")

        # wall-clock (ignore tz) folder vs session.json
        if folder_dt and sess_dt and abs(
                (folder_dt - sess_dt.replace(tzinfo=None)).total_seconds()) > 2:
            D.append(f"  datetime: folder={folder_dt} vs "
                     f"session.json wall-clock={sess_dt.replace(tzinfo=None)}")

        # timezone label: NWB vs session.json absolute instant
        if sess_dt is not None and nwb_dt is not None:
            nwb_off = nwb_dt.utcoffset()
            sess_off = sess_dt.utcoffset()
            if nwb_off != sess_off:
                D.append(f"  timezone: nwb tz offset={nwb_off} vs "
                         f"session.json tz offset={sess_off} "
                         f"(nwb={nwb_dt.isoformat()}, "
                         f"session.json={sess_dt.isoformat()})")
            # absolute-instant difference
            delta = abs((nwb_dt - sess_dt).total_seconds())
            if delta > 60:
                D.append(f"  acquisition instant off by {delta/3600:.2f} h "
                         f"(nwb={nwb_dt.isoformat()} vs "
                         f"session.json={sess_dt.isoformat()})")

        # ---- NWB subject metadata completeness vs subject.json ----
        if packaged:
            sj_sex = {"Male": "M", "Female": "F"}.get(subj_j.get("sex"), subj_j.get("sex"))
            if sj_sex and nm.get("sex") in (None, "U") and nm.get("sex") != sj_sex:
                D.append(f"  nwb.sex={nm.get('sex')!r} but subject.json sex={subj_j.get('sex')!r}")
            if subj_j.get("genotype") and not nm.get("genotype"):
                D.append(f"  nwb.genotype missing but subject.json genotype="
                         f"{subj_j.get('genotype')!r}")

        # ---- packaging status ----
        if not pkl:
            D.append("  NOT PACKAGED: no stim .pkl found in asset")
        elif pm.get("not_behavior") or pm.get("pkl_error"):
            why = pm.get("pkl_error", "pkl has no items.behavior (not a change-detection stim pkl)")
            D.append(f"  NOT PACKAGED: pkl {pkl.name} unreadable as behavior ({why})")
        elif not packaged:
            reason = "no sync .h5 next to pkl" if sync is None else "NWB absent in /results"
            D.append(f"  NOT PACKAGED: session {sid} ({reason})")

        status = "OK" if not D else f"{len(D)} issue(s)"
        print(f"[{status}] {name}  (session={sid}, packaged={packaged})")
        for line in D:
            print(line)
            discrepancies.append(f"{name}: {line.strip()}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(discrepancies)} discrepancy line(s) across "
          f"{len(assets)} assets")
    print("=" * 70)


if __name__ == "__main__":
    main()
