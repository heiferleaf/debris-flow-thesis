"""Historical, withdrawn occupancy-candidate reference; not current labels.

The20% occupancy proposal was withdrawn on2026-11-08 for insufficient semantic
fit to pulsed/small debris flows and ordinary water. Keep its reference tests
and mathematical evidence, but never use classify_visual_state to release data.

No training or inference pipeline is connected here. Completeness refers to an
intention to outline all identifiable target evidence, never all actual fluid
or independently certified geological truth. A possible polygon by itself
does not certify a whole-target upper bound.
"""
from __future__ import annotations

import math
from numbers import Real


ONTOLOGY = "visual-evidence-occupancy-v1"
MASS_TOLERANCE = 1e-12
PREDICTION_SUM_TOLERANCE = 1e-6


def _unit_interval(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite number in [0,1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be a finite number in [0,1]")
    return value


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _flag(record: dict, name: str) -> bool:
    value = record.get(name, False)
    if type(value) is not bool:
        raise ValueError(name + " must be an explicit boolean")
    return value


def classify_visual_state(record: dict, *, theta: float = 0.2) -> str:
    """Return N/T/P/A/U from an explicit current-ontology evidence record.

    E is 0, 1, or None/'unknown'. s_lower, when supplied, asserts a reliable
    conservative lower bound of identifiable target evidence in the fixed ROI;
    it must not be an arbitrary possible-region area. s_upper is usable only
    when visible_evidence_complete or upper_bound_validated is explicitly true.
    Otherwise its effective value is 1, so an incomplete small mask cannot T.
    A means existence confirmed but salience unresolved; U means E unresolved.
    Legacy kind='transition' is never converted without the new ontology.
    """
    if record.get("ontology") != ONTOLOGY:
        raise ValueError("Explicit visual-evidence-occupancy-v1 ontology is required; legacy transition is not accepted")
    if "E" not in record:
        raise ValueError("E must be explicitly 0, 1, or unknown")
    existence = record["E"]
    if existence is not None and existence != "unknown" and existence not in (0, 1):
        raise ValueError("E must be explicitly 0, 1, or unknown")
    threshold = _unit_interval(theta, "theta")
    if threshold == 0:
        raise ValueError("theta must be greater than zero")
    lower = _unit_interval(record.get("s_lower", 0.0), "s_lower")
    upper = _unit_interval(record.get("s_upper", 1.0), "s_upper")
    complete = _flag(record, "visible_evidence_complete")
    validated_upper = _flag(record, "upper_bound_validated")
    effective_upper = upper if complete or validated_upper else 1.0
    if lower > effective_upper:
        raise ValueError("Reliable salience bounds contradict each other")
    if existence == 0:
        return "N"
    if existence is None or existence == "unknown":
        return "U"
    if effective_upper < threshold:
        return "T"
    if lower >= threshold:
        return "P"
    return "A"


def factorized_probs(e: float, h: float) -> tuple[float, float, float]:
    """Existence probability e and conditional high-salience h -> (N,T,P)."""
    existence = _unit_interval(e, "e")
    high_given_exists = _unit_interval(h, "h")
    return (1.0 - existence, existence * (1.0 - high_given_exists), existence * high_given_exists)


def _shares(n, t, p, a) -> tuple[float, float, float, float]:
    values = tuple(_unit_interval(value, name) for name, value in (("n", n), ("t", t), ("p", p), ("a", a)))
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise ValueError("Truth shares n,t,p,a must sum to one")
    return values


def _negative_log(weight: float, probability: float) -> float:
    # 0*log(0)=0; a positive true mass assigned zero probability costs infinity.
    if weight == 0:
        return 0.0
    if probability == 0:
        return math.inf
    return -weight * math.log(probability)


def _negative_log_complement(weight: float, probability: float) -> float:
    if weight == 0:
        return 0.0
    if probability == 1:
        return math.inf
    return -weight * math.log1p(-probability)


def binary_cross_entropy(target: float, probability: float) -> float:
    """Endpoint-safe probability BCE, without clipping impossible predictions."""
    truth = _unit_interval(target, "target")
    prediction = _unit_interval(probability, "probability")
    return _negative_log(truth, prediction) + _negative_log_complement(1.0 - truth, prediction)


def factorized_loss(e: float, h: float, *, n: float, t: float, p: float, a: float = 0.0) -> float:
    """BCE(t+p+a,e) + (t+p)*BCE(p/(t+p),h), with the second term zero if t+p=0.

    The algebraically expanded form avoids 0/0 and uses only truth shares as
    coefficients; neither predicted e nor a guessed T/P split weights h.
    Input sum tolerance is numerical only: no shares/probabilities are clipped
    or renormalized. A contradictory endpoint prediction returns +infinity.
    """
    existence = _unit_interval(e, "e")
    high_given_exists = _unit_interval(h, "h")
    negative, low, high, unresolved_salience = _shares(n, t, p, a)
    exists = math.fsum((low, high, unresolved_salience))
    return math.fsum((_negative_log_complement(negative, existence),
                      _negative_log(exists, existence),
                      _negative_log_complement(low, high_given_exists),
                      _negative_log(high, high_given_exists)))


def softmax_partial_loss(probabilities, *, n: float, t: float, p: float, a: float = 0.0) -> float:
    """-n log pN -t log pT -p log pP -a log(pT+pP) for a normalized triple.

    A supervises target existence but does not invent a T/P assignment. It is
    not U: n+t+p+a must cover the full technically valid, E-resolved support.
    Predicted sums allow FP32 numerical error up to 1e-6; truth shares retain
    their 1e-12 tolerance. Probabilities are neither clipped nor renormalized.
    Prefer softmax_partial_loss_logits in production to avoid rounded endpoints.
    """
    if len(probabilities) != 3:
        raise ValueError("Exactly three probabilities in N,T,P order are required")
    pred_n, pred_t, pred_p = (_unit_interval(value, name) for name, value in zip(("pN", "pT", "pP"), probabilities))
    if not math.isclose(math.fsum((pred_n, pred_t, pred_p)), 1.0, rel_tol=0.0, abs_tol=PREDICTION_SUM_TOLERANCE):
        raise ValueError("N,T,P probabilities must sum to one")
    negative, low, high, unresolved_salience = _shares(n, t, p, a)
    return math.fsum((_negative_log(negative, pred_n), _negative_log(low, pred_t),
                      _negative_log(high, pred_p), _negative_log(unresolved_salience, math.fsum((pred_t, pred_p)))))


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _logsumexp(values) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def factorized_loss_logits(existence_logit: float, high_logit: float, *, n: float, t: float,
                           p: float, a: float = 0.0) -> float:
    """Stable logits form, including +/-1000; both terms use truth masses.

    This is one per-window joint loss, not two separately averaged head losses.
    No probability conversion, clipping, or t+p division is performed.
    """
    ze = _finite_real(existence_logit, "existence_logit")
    zh = _finite_real(high_logit, "high_logit")
    negative, low, high, unresolved_salience = _shares(n, t, p, a)
    exists = math.fsum((low, high, unresolved_salience))
    terms = ((negative, ze), (exists, -ze), (low, zh), (high, -zh))
    return math.fsum(weight * _softplus(value) for weight, value in terms if weight != 0)


def softmax_partial_loss_logits(logits, *, n: float, t: float, p: float, a: float = 0.0) -> float:
    """Stable logsumexp partial loss from three N,T,P logits, without clipping."""
    if len(logits) != 3:
        raise ValueError("Exactly three logits in N,T,P order are required")
    values = tuple(_finite_real(value, "logit") for value in logits)
    negative, low, high, unresolved_salience = _shares(n, t, p, a)
    maximum = max(values)
    shifted = tuple(value - maximum for value in values)
    log_partition = _logsumexp(shifted)
    terms = [(negative, log_partition - shifted[0]), (low, log_partition - shifted[1]),
             (high, log_partition - shifted[2])]
    if unresolved_salience != 0:
        terms.append((unresolved_salience, log_partition - _logsumexp(shifted[1:])))
    return math.fsum(weight * loss for weight, loss in terms if weight != 0)
