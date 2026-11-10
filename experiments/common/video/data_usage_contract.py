"""Pure metadata checks for the current cross-camera data-use contract.

These checks do not define labels, infer similarity families, select splits,
sample media, or authorize training. Family membership is supplied evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
import re
import unicodedata


PROTOCOL_ID = "cross-camera-fixed-view-v1"


def canonical_camera_identity(record: dict) -> str:
    """Use the registered fixed-view name; source/site-code/date are not keys."""
    name = record.get("source_camera")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing registered source_camera; do not infer one from a D code or filename")
    name = unicodedata.normalize("NFC", name.strip())
    if re.fullmatch(r"D\d+", name, flags=re.IGNORECASE):
        raise ValueError("A D code alone is not a registered fixed-view name")
    return "s|" + name


def validate_camera_partitions(roles: dict, sources: dict, lineage: dict | None = None) -> dict:
    """Check disjoint roles, including explicit ancestors of derived samples.

    Records require source_camera. Derived/synthetic records (and all records
    passed in lineage) additionally require nonempty parent_ids. Standard
    nested CV roles are checked as inner and outer partitions; other supplied
    roles are checked pairwise. Unassigned sources remain visible in the audit.
    """
    lineage = lineage or {}
    if set(sources) & set(lineage):
        raise ValueError("Source and derived-record identities must be distinct")
    records = {**sources, **lineage}
    cameras = {uid: canonical_camera_identity(record) for uid, record in records.items()}
    media_hashes = {}
    for uid, record in records.items():
        if "source_media_sha256" not in record:
            continue
        value = record["source_media_sha256"]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("A full source-media SHA256 is required when supplied: " + uid)
        media_hashes[uid] = value.lower()
    ancestors, visiting = {}, set()

    def parents_for(uid):
        if uid in ancestors:
            return ancestors[uid]
        if uid in visiting:
            raise ValueError("Cyclic parent lineage: " + uid)
        visiting.add(uid)
        record = records[uid]
        derived = uid in lineage or record.get("synthetic") or record.get("derived") or "parent_ids" in record
        parents = record.get("parent_ids", [])
        if derived and (not isinstance(parents, list) or not parents):
            raise ValueError("Derived sample requires explicit parent_ids: " + uid)
        result = {uid}
        for parent in parents:
            if parent not in records:
                raise ValueError("Missing parent record: " + str(parent))
            if cameras[parent] != cameras[uid]:
                raise ValueError("Derived sample camera differs from parent: " + uid)
            result.update(parents_for(parent))
        visiting.remove(uid)
        ancestors[uid] = result
        return result

    for uid in records:
        parents_for(uid)
    sides = {}
    expanded = {}
    for role, ids in roles.items():
        if len(ids) != len(set(ids)) or not set(ids) <= set(records):
            raise ValueError("Duplicate or unknown sample in role: " + role)
        sides[role] = set(ids)
        expanded[role] = set().union(*(ancestors[uid] for uid in ids)) if ids else set()
    nested = {"inner_train", "inner_val", "outer_train", "outer_eval"}
    if set(sides) == nested:
        if sides["outer_train"] != sides["inner_train"] | sides["inner_val"]:
            raise ValueError("Outer training role must equal the inner refit union")
        pairs = [("inner_train", "inner_val"), ("outer_train", "outer_eval")]
    else:
        pairs = list(itertools.combinations(sorted(sides), 2))
    camera_sets = {role: {cameras[uid] for uid in ids} for role, ids in expanded.items()}
    media_sets = {role: {media_hashes[uid] for uid in ids if uid in media_hashes}
                  for role, ids in expanded.items()}
    for first, second in pairs:
        if camera_sets[first] & camera_sets[second]:
            raise ValueError(f"Camera crossed {first}/{second}: {sorted(camera_sets[first] & camera_sets[second])}")
        if media_sets[first] & media_sets[second]:
            raise ValueError(f"Source media crossed {first}/{second}")
    return {"protocol_id": PROTOCOL_ID,
            "role_camera_keys": {role: sorted(keys) for role, keys in camera_sets.items()},
            "role_expanded_sample_counts": {role: len(ids) for role, ids in expanded.items()},
            "unassigned_samples": sorted(set(records) - set().union(*sides.values())) if sides else sorted(records),
            "media_hash_coverage": len(media_hashes),
            "historical_protocols_accepted_as_current": False}


def physical_window_key(source_media_sha256: str, ordered_indices: list[int]) -> str:
    """Identity is source bytes plus original-order indices, never an alias/label."""
    if not isinstance(source_media_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_media_sha256):
        raise ValueError("A full source-media SHA256 is required")
    indices = list(ordered_indices)
    if not indices or any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("Source frame indices must be nonnegative integers")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Physical source frame indices must be distinct and increasing")
    value = [source_media_sha256.lower(), indices]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


def _window_states(row):
    """Charge every known nonzero time share; MIX is never an extra budget."""
    shares = row.get("time_shares")
    if shares is None:
        if row.get("state") not in {"N", "T", "P"}:
            raise ValueError("Pure N/T/P state or explicit time_shares required; MIX/U is not a new budget")
        return ((row["state"], 1.0),)
    if not isinstance(shares, dict) or set(shares) != {"N", "T", "P"}:
        raise ValueError("time_shares must explicitly contain N,T,P")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in shares.values()):
        raise ValueError("Time shares must be finite nonnegative fractions")
    if not math.isclose(math.fsum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Complete known time shares must sum to one; do not normalize unknown time")
    signature = tuple((state, float(shares[state])) for state in ("N", "T", "P") if shares[state] > 0)
    if row.get("state") in {"N", "T", "P"} and [state for state, share in signature] != [row["state"]]:
        raise ValueError("Pure-state name conflicts with its time shares")
    return signature


def validate_window_quotas(windows: list[dict], *, camera_cap: int, family_cap: int, video_cap: int) -> dict:
    """Validate an explicitly specified legacy quota selection.

    This helper does not define the current full-coverage sampling policy.
    All caps must be supplied from the historical card being checked; four is
    not a project-wide default. Current coefficients live in full_coverage_sampling.

    Each row requires source_camera, source_media_sha256, frame_indices, state,
    and either family_id or family_status='unresolved'. Unresolved rows still
    consume video/camera budgets; they are not silently merged into one family.
    Pure states are N/T/P; mixed rows provide complete time_shares and consume
    each nonzero state's quota while remaining one physical training window.
    These metadata checks are not certification of the labels' truth.
    basic_sampling_ready (and compatibility alias quota_ready) means required
    identity/budget checks passed; unresolved families do not block a baseline.
    family_review_complete separately reports whether all families are resolved.
    """
    if any(type(cap) is not int or cap < 1 for cap in (camera_cap, family_cap, video_cap)):
        raise ValueError("Explicit positive integer caps required")
    video, family, camera = Counter(), Counter(), Counter()
    seen, frames = {}, set()
    unresolved, total_frames, mixed = 0, 0, 0
    for row in windows:
        view = canonical_camera_identity(row)
        signature = _window_states(row)
        states = [state for state, share in signature]
        mixed += int(len(states) > 1)
        media = row.get("source_media_sha256")
        key = physical_window_key(media, row.get("frame_indices", []))
        if key in seen:
            if seen[key] != signature:
                raise ValueError("Same physical window has conflicting states")
            raise ValueError("Same physical window repeated within one epoch/selection")
        seen[key] = signature
        media = media.lower()
        family_id = row.get("family_id")
        if not family_id or row.get("family_status") == "unresolved":
            if row.get("family_status") != "unresolved" or family_id:
                raise ValueError("Missing family must be explicitly unresolved without an invented family_id")
            unresolved += 1
        else:
            if len(states) > 1:
                raise ValueError("A mixed-state window cannot be assigned to a pure-state content family")
            family[(view, states[0], family_id)] += 1
        for state in states:
            video[(media, state)] += 1
            camera[(view, state)] += 1
        frames.update((media, index) for index in row["frame_indices"])
        total_frames += len(row["frame_indices"])
    if max(video.values(), default=0) > video_cap:
        raise ValueError("Original-video/state cap exceeded; CSV fragments or file aliases do not add quota")
    if max(family.values(), default=0) > family_cap:
        raise ValueError("Content-family/state shared cap exceeded")
    if max(camera.values(), default=0) > camera_cap:
        raise ValueError("Camera/state total cap exceeded")
    return {"protocol_id": PROTOCOL_ID, "window_count": len(windows), "mixed_window_count": mixed,
            "video_state_counts": [{"source_media_sha256": key[0], "state": key[1], "count": count}
                                   for key, count in sorted(video.items())],
            "family_state_counts": [{"camera_key": key[0], "state": key[1], "family_id": key[2], "count": count}
                                    for key, count in sorted(family.items())],
            "camera_state_counts": [{"camera_key": key[0], "state": key[1], "count": count}
                                    for key, count in sorted(camera.items())],
            "unresolved_family_window_count": unresolved, "basic_sampling_ready": True,
            "family_review_complete": unresolved == 0, "quota_ready": True,
            "near_duplicate_removal_verified": False,
            "source_frame_exposures": total_frames, "unique_source_frames": len(frames),
            "reused_source_frame_exposures": total_frames - len(frames)}
