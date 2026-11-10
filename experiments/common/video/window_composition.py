"""Time composition of one real-time window; independent of frames and models.

Only confirmed N/P intervals define a point target. Legacy weak-state T is
unknown, whereas the new T is a whole-window composition anchor. All intervals
and windows use the same caller-supplied time coordinate and [start, end).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


CONTRACT_VERSION = "window-composition-20261108-v1"
KINDS = {"n": "N", "neg": "N", "p": "P", "pos": "P",
         "u": "U", "unknown": "U", "transition": "Tlegacy",
         "tlegacy": "Tlegacy", "t": "Tlegacy", "cut": "cut"}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    kind: str


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _anchor(anchor: float) -> float:
    anchor = _finite(anchor, "anchor")
    if not 0 < anchor < 1:
        raise ValueError("anchor must be strictly between zero and one")
    return anchor


def normalize_intervals(intervals: Iterable[Interval | Mapping]) -> tuple[Interval, ...]:
    """Validate disjoint intervals and merge exactly adjacent equal kinds.

    Overlaps, including same-label overlaps, are errors rather than permission
    to double count. Unknown aliases are rejected; callers must use explicit U.
    """
    result = []
    for interval in intervals:
        if isinstance(interval, Interval):
            start, end, kind = interval.start, interval.end, interval.kind
        else:
            start, end, kind = interval["start"], interval["end"], interval["kind"]
        start, end = _finite(start, "interval start"), _finite(end, "interval end")
        if end <= start:
            raise ValueError("interval start must precede end")
        try:
            kind = KINDS[str(kind).strip().lower()]
        except KeyError:
            raise ValueError("unsupported interval kind: " + str(kind)) from None
        result.append(Interval(start, end, kind))
    result.sort(key=lambda item: (item.start, item.end))
    merged = []
    for interval in result:
        if merged and interval.start < merged[-1].end:
            raise ValueError("intervals overlap")
        if merged and interval.start == merged[-1].end and interval.kind == merged[-1].kind:
            merged[-1] = Interval(merged[-1].start, interval.end, interval.kind)
        else:
            merged.append(interval)
    return tuple(merged)


def soft_target(r: float, anchor: float = .5) -> tuple[float, float, float]:
    """Return (yN,yT,yP) for confirmed positive time fraction r."""
    r, anchor = _finite(r, "positive fraction"), _anchor(anchor)
    if not 0 <= r <= 1:
        raise ValueError("positive fraction must be in [0,1]")
    if r <= anchor:
        return (1 - r / anchor, r / anchor, 0.0)
    return (0.0, (1 - r) / (1 - anchor), (r - anchor) / (1 - anchor))


def reconstruct_fraction(probabilities: Iterable[float], anchor: float = .5) -> float:
    """Decode pP + anchor*pT; a window composition, not last-frame state.

    Model probabilities may sum to one within 1e-6 to accommodate FP32
    softmax rounding. Values remain finite and in [0,1]; no renormalization
    is performed. Only a residual decoded endpoint overshoot is clamped.
    This input tolerance does not relax the three-anchor target algebra.
    """
    anchor = _anchor(anchor)
    values = tuple(_finite(value, "probability") for value in probabilities)
    if len(values) != 3 or any(value < 0 or value > 1 for value in values):
        raise ValueError("expected three nonnegative probabilities in N,T,P order")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    return min(1.0, max(0.0, values[2] + anchor * values[1]))


def dominant_class(probabilities: Iterable[float]) -> str:
    """Argmax the supplied vector; ties within 1e-12 favor T, then N, then P.

    Applied to g(r), this gives the target's composition display. Applied
    directly to a model p, it gives probability argmax, which can differ
    from the composition display derived from r_hat; use composition_class
    for the latter.
    """
    values = tuple(_finite(value, "probability") for value in probabilities)
    reconstruct_fraction(values)  # Validate the probability vector.
    maximum = max(values)
    return next(label for label, index in (("T", 1), ("N", 0), ("P", 2))
                if math.isclose(values[index], maximum, rel_tol=0, abs_tol=1e-12))


def composition_class(probabilities: Iterable[float], anchor: float = .5) -> str:
    """Display argmax g(r_hat), separate from direct probability argmax.

    At a=.5 the composition thresholds are r_hat=.25 and .75, with T at
    both ties. For example p=(.4,.2,.4) has r_hat=.5 and composition T,
    even though its direct probability argmax ties N and P.
    """
    return dominant_class(soft_target(reconstruct_fraction(probabilities, anchor), anchor))


def compose_window(start: float, end: float, intervals: Iterable[Interval | Mapping],
                   *, anchor: float = .5, forbidden_ranges: Iterable[tuple[float, float]] = ()) -> dict:
    """Integrate confirmed time, unknown time, and separate technical vetoes.

    Empty/unannotated time is U, never N. Any positive-duration intersection
    with a technical forbidden range invalidates the whole window. An untyped
    legacy cut also blocks it, with its own reason; it is not reclassified as U.
    No cutting, concatenation, or duration renormalization is performed.
    """
    start, end = _finite(start, "window start"), _finite(end, "window end")
    anchor = _anchor(anchor)
    if end <= start:
        raise ValueError("window must have positive duration")
    normalized = normalize_intervals(intervals)
    forbidden = []
    for left, right in forbidden_ranges:
        left, right = _finite(left, "forbidden start"), _finite(right, "forbidden end")
        if right <= left:
            raise ValueError("forbidden range start must precede end")
        forbidden.append((left, right))
    pieces = {kind: [] for kind in ("N", "P", "U", "Tlegacy", "cut")}
    cursor, gaps = start, []
    for interval in normalized:
        left, right = max(start, interval.start), min(end, interval.end)
        duration = right - left
        if duration > 0:
            if left > cursor:
                gaps.append(left - cursor)
            cursor = right
            pieces[interval.kind].append(duration)
    if cursor < end:
        gaps.append(end - cursor)
    seconds = {kind: math.fsum(values) for kind, values in pieces.items()}
    length = end - start
    gap = math.fsum(gaps)
    unknown = math.fsum((seconds["U"], seconds["Tlegacy"], gap))
    reasons = []
    if any(max(start, left) < min(end, right) for left, right in forbidden):
        reasons.append("technical_forbidden_range")
    if seconds["cut"] > 0:
        reasons.append("untyped_cut")
    result = {"start_sec": start, "end_sec": end, "duration_sec": length,
              "anchor": anchor, "seconds": {**seconds, "implicit_gap": gap, "unknown_total": unknown},
              "status": "technical_invalid" if reasons else "U" if unknown > 0 else "confirmed",
              "reasons": reasons, "r_min": None, "r_max": None,
              "target": None, "r": None, "dominant_class": None,
              "genuinely_mixed": False}
    if reasons:
        return result
    result["r_min"] = min(1.0, max(0.0, seconds["P"] / length))
    result["r_max"] = min(1.0, max(0.0, (seconds["P"] + unknown) / length))
    if unknown > 0:
        result["reasons"] = (["legacy_T_pending"] if seconds["Tlegacy"] > 0 else [])
        if seconds["U"] > 0:
            result["reasons"].append("explicit_unknown")
        if gap > 0:
            result["reasons"].append("implicit_gap")
        return result
    r = result["r_min"]
    target = soft_target(r, anchor)
    result.update(status="N" if r == 0 else "P" if r == 1 else "MIX",
                  target=target, r=r, dominant_class=dominant_class(target),
                  genuinely_mixed=0 < r < 1)
    return result
