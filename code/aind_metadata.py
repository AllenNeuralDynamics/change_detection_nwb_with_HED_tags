"""AIND derived-asset metadata for the change-detection / sweepstim NWB capsule.

Builds the metadata for a *derived* data asset from the *primary* (raw) asset
that a session's pkl/sync came from, following AIND data-organization
conventions
(https://docs.allenneuraldynamics.org/en/latest/policies_practices/data_organization.html):

    <primary-asset-name>_behavior-nwb_<proc-date>_<proc-time>/
      behavior.nwb.zarr/            NWB-Zarr (written by the packager)
      behavior.events.json          HED/BIDS column sidecar (written by the packager)
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
  they stay identical to the primary asset (which may span several schema patch
  versions). We do NOT round-trip inherited files through the library.
- Robust to schema-version drift in the primary data_description: we do NOT
  whole-validate the base document (that would fail on drift in fields we never
  use). We extract only the fields a DerivedDataDescription needs and construct
  a fresh one at the pinned schema version. If a required core field is missing
  or malformed the construction raises loudly — the caller turns that into a
  hard per-session failure (never a silent metadata-less NWB).
- The derived data_description ``name`` is authored by the library and the
  output folder is named from it, so ``data_description.name`` always equals the
  asset folder name (what AIND tooling expects).
- Base metadata comes from the JSON files inside the mounted /data asset
  (no metadata DocDB dependency).
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
# a no-underscore rule on the token. Use a hyphen.
PROCESS_LABEL = "behavior-nwb"
# Label used where underscores ARE allowed (DataProcess parameters, NWB fields).
PROCESS_LABEL_SNAKE = "behavior_nwb"

# Provenance recorded in processing.json — the project / team responsible.
PROCESSOR_FULL_NAME = "Learning mFISH / V1 omFISH team"

REPO_URL = "https://github.com/AllenNeuralDynamics/change_detection_nwb_with_HED_tags"

# Files copied verbatim from the primary asset into the derived asset.
INHERITED_FILES = ("subject.json", "procedures.json", "session.json", "rig.json")

# Fields pulled from the primary data_description.json to author the derived one.
# These are the DerivedDataDescription-required core fields plus a couple of
# optional ones; everything else in the base document is intentionally ignored
# so drift there cannot break us.
_REQUIRED_DD_FIELDS = (
    "name", "platform", "subject_id", "institution",
    "funding_source", "investigators", "modality",
)
_OPTIONAL_DD_FIELDS = ("project_name", "group")


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

    Returns a dict with keys understood by the packagers' ``metadata`` param:
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


# ── derived data_description (authored + validated) ─────────────────────────
def build_derived_data_description(
        base_dd_json: dict,
        creation_time: datetime.datetime) -> ds.DerivedDataDescription:
    """Author a DerivedDataDescription from the primary data_description JSON.

    Robust to schema-version drift: only the fields a DerivedDataDescription
    needs are extracted from ``base_dd_json`` and coerced by pydantic on
    construction. Drift in unrelated parts of the base document is ignored.
    Raises (ValueError / pydantic.ValidationError) if a required core field is
    absent or malformed — the caller must treat that as a hard failure.

    The returned model's ``name`` is set by the library (a post-validator) to
    ``build_data_name(f"{base_name}_{PROCESS_LABEL}", creation_time)``; the
    output asset folder MUST be named from ``derived.name`` so that
    ``data_description.name`` equals the folder name.
    """
    missing = [f for f in _REQUIRED_DD_FIELDS if base_dd_json.get(f) is None]
    if missing:
        raise ValueError(
            "primary data_description.json is missing required field(s) "
            f"{missing}; cannot author a derived data_description")

    fields = dict(
        creation_time=creation_time,
        process_name=PROCESS_LABEL,
        input_data_name=base_dd_json["name"],
        platform=base_dd_json["platform"],
        subject_id=str(base_dd_json["subject_id"]),
        institution=base_dd_json["institution"],
        funding_source=base_dd_json["funding_source"],
        investigators=base_dd_json["investigators"],
        modality=base_dd_json["modality"],
    )
    for f in _OPTIONAL_DD_FIELDS:
        if base_dd_json.get(f) is not None:
            fields[f] = base_dd_json[f]

    return ds.DerivedDataDescription(**fields)


def build_processing(primary_name: str,
                     out_asset_name: str,
                     start: datetime.datetime,
                     end: datetime.datetime,
                     code_version: str,
                     parameters: dict,
                     notes: str,
                     processor_full_name: str = PROCESSOR_FULL_NAME) -> ps.Processing:
    """Author a v1 Processing = PipelineProcess([DataProcess(...)]).

    Validated on construction. ``parameters`` carries e.g. the real per-session
    ``monitor_delay_sec`` and ``hed_schema_version``.
    """
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
        notes=notes,
    )
    return ps.Processing(
        processing_pipeline=ps.PipelineProcess(
            processor_full_name=processor_full_name,
            data_processes=[dp],
        )
    )


def write_derived_metadata(out_dir: Path,
                           derived_dd: ds.DerivedDataDescription,
                           processing: ps.Processing,
                           primary_dir: Path | None) -> dict:
    """Write the derived asset's metadata into ``out_dir``.

    ``derived_dd`` and ``processing`` are already-built, already-validated
    models (constructing them validates). We write them, re-read each as its
    proper class as a post-write gate, and copy the inherited files verbatim.

    Returns a summary dict {authored: [...], inherited: [...], missing: [...]}.
    Raises if an authored file fails to write or re-validate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"authored": [], "inherited": [], "missing": []}

    # --- authored: data_description.json (+ post-write gate) ---
    derived_dd.write_standard_file(output_directory=out_dir)
    ds.DerivedDataDescription.model_validate(
        json.load(open(out_dir / "data_description.json")))
    summary["authored"].append("data_description.json")

    # --- authored: processing.json (+ post-write gate) ---
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
