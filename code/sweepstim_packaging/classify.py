"""Session classification for standalone SweepStim packaging."""

from __future__ import annotations


def classify_sweepstim_session(data: dict) -> tuple[bool, str]:
    """Return ``(is_sweepstim, detail)`` based on pkl structure.

    A passive SweepStim pkl is identified by one or both of:
    - a top-level ``stimuli`` list with entries, and/or
    - an ``items.foraging`` payload,
    while lacking a non-empty ``items.behavior.trial_log``.
    """
    items = data.get("items") or {}
    behavior = items.get("behavior") or {}
    has_behavior_trials = bool((behavior.get("trial_log") or []))
    has_stim_list = isinstance(data.get("stimuli"), list) and len(data["stimuli"]) > 0
    has_foraging = "foraging" in items

    if has_behavior_trials:
        return False, "behavior session (non-empty behavior.trial_log)"
    if has_stim_list or has_foraging:
        return True, (
            f"sweepstim-like structure: top-level stimuli list={has_stim_list}, "
            f"items.foraging present={has_foraging}"
        )
    return False, "missing sweepstim signatures (no stimuli list/foraging item)"
