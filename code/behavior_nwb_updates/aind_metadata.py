"""AIND derived-asset metadata for the change-detection NWB capsule.

Builds the metadata for a *derived* data asset from the *primary* (raw)
multiplane-ophys asset that a session's pkl/sync came from, following AIND
data-organization conventions
(https://docs.allenneuraldynamics.org/en/latest/policies_practices/data_organization.html):

    <primary-asset-name>_behavior-nwb_<proc-date>_<proc-time>/
      behavior.nwb.zarr/            NWB-Zarr (written by package_to_nwb)
      behavior.events.json          HED/BIDS column sidecar (written by package_to_nwb)
      data_description.json         REGENERATED  - DerivedDataDescription, validated
      processing.json               NEW          - Processing, validated
      subject.json                  inherited verbatim from primary asset
      procedures.json               inherited verbatim
      session.json                  inherited verbatim
      rig.json                      inherited verbatim

Design rules (from the AIND docs + project decisions):

- Immutability: the behavior NWB is a *derived* asset. We never write back into
  the primary /data asset; the derived asset is self-contained.
- Only files we *author* (data_description, processing) are built with
  aind-data-schema and validated. Inherited files are copied byte-for-byte so
  they stay identical to the primary multiplane-ophys asset (which spans several
  schema patch versions). We do NOT round-trip inherited files through the
  library — that would measure library drift, not correctness.
- Base metadata comes from the JSON files inside the mounted /data asset
  (no metadata DocDB dependency).
- Schema: aind-data-schema v1.x (keeps session/rig, not acquisition/instrument).
  The regenerated data_description is emitted at the version the pinned library
  produces (1.0.4 for aind-data-schema 1.4.0); inherited files keep their own.
"""
from __future__ import annotations

import datetime
import json
import logging
import shutil
from pathlib import Path

import aind_data_schema.core.data_description as ds
import aind_data_schema.core.processing as ps
from aind_data_schema_models.process_names import ProcessName

logger = logging.getLogger("aind_metadata")

# Asset-name token for this process. MUST be underscore-free: the AIND naming
# convention uses '_' as the token separator, and the schema validator enforces
# ^[^<>:;"/|? \\_]+$ on the token. Use a hyphen.
PROCESS_LABEL = "behavior-nwb"
# Label used where underscores ARE allowed (DataProcess parameters, NWB fields).
PROCESS_LABEL_SNAKE = "behavior_nwb"

REPO_URL = "https://github.com/AllenNeuralDynamics/change_detection_nwb_with_HED_tags"

# Files copied verbatim from the primary asset into the derived asset.
INHERITED_FILES = ("subject.json", "procedures.json", "session.json", "rig.json")


# ── primary-asset discovery + loading ──────────────────────────────────────
def find_primary_asset_dir(pkl_path: Path, data_dir: Path) -> Path | None:
    """Return the top-level /data/<primary-asset-name> folder containing ``pkl``.

    The primary asset is the first path component beneath ``data_dir`` (matches
    the ``build_asset_map`` logic in summarize_sessions.py).
    """
    pkl_path = Path(pkl_path)
    data_dir = Path(data_dir)
    try:
        rel = pkl_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return None
    if not rel.parts:
        return None
    return data_dir / rel.parts[0]


def load_primary_metadata(primary_dir: Path | None) -> dict[str, dict]:
    """Read the AIND core-metadata JSONs present in ``primary_dir``.

    Returns a dict keyed by filename (e.g. ``"subject.json"``) -> parsed JSON.
    Tolerant of a missing directory or missing files (returns what it finds).
    """
    out: dict[str, dict] = {}
    if primary_dir is None or not Path(primary_dir).is_dir():
        return out
    for fn in ("data_description.json",) + INHERITED_FILES:
        p = Path(primary_dir) / fn
        if p.is_file():
            try:
                out[fn] = json.load(open(p))
            except Exception:
                logger.exception("Could not parse %s", p)
    return out


# ── NWB subject fields (threaded into the NWB itself) ───────────────────────
_SEX_MAP = {"Female": "F", "Male": "M", "female": "F", "male": "M"}


def subject_fields_for_nwb(primary_meta: dict[str, dict],
                           session_start: datetime.datetime) -> dict:
    """Extract subject metadata for the NWB Subject from the primary subject.json.

    Returns a dict with keys understood by package_to_nwb's ``metadata`` param:
    ``sex``, ``genotype``, ``age``, ``date_of_birth``, ``strain``, ``species``.
    Missing values are simply omitted (packager keeps its own defaults).
    """
    subj = primary_meta.get("subject.json") or {}
    out: dict = {}

    if subj.get("sex"):
        out["sex"] = _SEX_MAP.get(subj["sex"], "U")
    if subj.get("genotype"):
        out["genotype"] = subj["genotype"]
    if subj.get("background_strain"):
        out["strain"] = subj["background_strain"]

    dob_str = subj.get("date_of_birth")
    if dob_str:
        out["date_of_birth"] = dob_str
        try:
            dob = datetime.date.fromisoformat(dob_str)
            start_date = session_start.date() if isinstance(
                session_start, datetime.datetime) else session_start
            age_days = (start_date - dob).days
            if age_days >= 0:
                out["age"] = f"P{age_days}D"  # ISO-8601 duration
        except Exception:
            logger.warning("Could not compute age from date_of_birth=%r", dob_str)

    species = subj.get("species")
    if isinstance(species, dict) and species.get("name"):
        out["species"] = species["name"]
    elif isinstance(species, str):
        out["species"] = species
    return out


# ── derived metadata (authored + validated) ────────────────────────────────
def build_derived_name(primary_name: str, creation_time: datetime.datetime) -> str:
    """<primary-asset-name>_behavior-nwb_<YYYY-MM-DD>_<HH-MM-SS> (local time)."""
    stamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{primary_name}_{PROCESS_LABEL}_{stamp}"


def _build_derived_data_description(base_dd_json: dict,
                                    creation_time: datetime.datetime) -> ds.DerivedDataDescription:
    """DerivedDataDescription from the primary data_description, validated on build."""
    base_dd = ds.DataDescription.model_validate(base_dd_json)
    derived = ds.DerivedDataDescription.from_data_description(
        base_dd, process_name=PROCESS_LABEL, creation_time=creation_time,
    )
    return derived


def _build_processing(primary_name: str, out_asset_name: str,
                      start: datetime.datetime, end: datetime.datetime,
                      code_version: str, parameters: dict,
                      processor_full_name: str) -> ps.Processing:
    """v1 Processing = PipelineProcess([DataProcess(...)]), validated on build."""
    dp = ps.DataProcess(
        name=ProcessName.FILE_FORMAT_CONVERSION,
        software_version=code_version,
        start_date_time=start,
        end_date_time=end,
        input_location=f"/data/{primary_name}",
        output_location=f"/results/{out_asset_name}",
        code_url=REPO_URL,
        code_version=code_version,
        parameters={**parameters, "process_label": PROCESS_LABEL_SNAKE},
        notes="camstim pkl+sync -> HED-annotated NWB-Zarr (behavior + stimulus)",
    )
    return ps.Processing(
        processing_pipeline=ps.PipelineProcess(
            processor_full_name=processor_full_name,
            data_processes=[dp],
        )
    )


def write_derived_metadata(out_dir: Path,
                           primary_dir: Path | None,
                           primary_meta: dict[str, dict],
                           primary_name: str,
                           creation_time: datetime.datetime,
                           end_time: datetime.datetime | None = None,
                           code_version: str = "unknown",
                           parameters: dict | None = None,
                           processor_full_name: str = "AIND Behavior") -> dict:
    """Write the derived asset's metadata into ``out_dir``.

    - data_description.json / processing.json : built with aind-data-schema and
      validated (construction validates; we also re-read each written file as its
      proper class as a post-write gate).
    - subject/procedures/session/rig : copied byte-for-byte from ``primary_dir``.

    Returns a summary dict {authored: [...], inherited: [...], missing: [...]}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_time = end_time or creation_time
    parameters = parameters or {}
    summary = {"authored": [], "inherited": [], "missing": []}

    out_asset_name = out_dir.name

    # --- authored: data_description.json ---
    base_dd_json = primary_meta.get("data_description.json")
    if base_dd_json is not None:
        derived_dd = _build_derived_data_description(base_dd_json, creation_time)
        derived_dd.write_standard_file(output_directory=out_dir)
        # post-write validation gate
        ds.DerivedDataDescription.model_validate(
            json.load(open(out_dir / "data_description.json")))
        summary["authored"].append("data_description.json")
    else:
        logger.warning("No data_description.json in primary asset %s — "
                       "cannot regenerate a derived one.", primary_name)
        summary["missing"].append("data_description.json")

    # --- authored: processing.json ---
    processing = _build_processing(
        primary_name, out_asset_name, creation_time, end_time,
        code_version, parameters, processor_full_name)
    processing.write_standard_file(output_directory=out_dir)
    ps.Processing.model_validate(json.load(open(out_dir / "processing.json")))
    summary["authored"].append("processing.json")

    # --- inherited: copy verbatim ---
    for fn in INHERITED_FILES:
        src = Path(primary_dir) / fn if primary_dir else None
        if src and src.is_file():
            shutil.copyfile(src, out_dir / fn)
            summary["inherited"].append(fn)
        else:
            summary["missing"].append(fn)
    return summary
