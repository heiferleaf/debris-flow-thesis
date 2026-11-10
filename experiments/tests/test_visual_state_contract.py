"""Reference-contract counterexamples; no model, media, labels, or training."""
import math
import struct
import unittest

from experiments.common.video.visual_state_contract import (
    ONTOLOGY, binary_cross_entropy, classify_visual_state, factorized_loss,
    factorized_loss_logits, factorized_probs, softmax_partial_loss,
    softmax_partial_loss_logits,
)


class VisualStateContractTest(unittest.TestCase):
    def test_existence_gate_and_reliable_occupancy_bounds(self):
        def classify(existence, **fields):
            return classify_visual_state({"ontology": ONTOLOGY, "E": existence, **fields})

        self.assertEqual(classify(0, s_lower=0.9, s_upper=1, visible_evidence_complete=True,
                                  motion_measure=1000), "N")
        self.assertEqual(classify(None, s_lower=0.9, s_upper=1, visible_evidence_complete=True), "U")
        self.assertEqual(classify("unknown"), "U")
        self.assertEqual(classify(1, s_lower=0.01, s_upper=0.04), "A")
        self.assertEqual(classify(1, s_lower=0.01, s_upper=0.04, mask_complete=True,
                                  possible_polygons=[[[0.1, 0.1], [0.2, 0.2]]]), "A")
        self.assertEqual(classify(1, s_lower=0.01, s_upper=0.04, visible_evidence_complete=True), "T")
        self.assertEqual(classify(1, s_lower=0.01, s_upper=0.04, upper_bound_validated=True), "T")
        self.assertEqual(classify(1, s_lower=0.2, s_upper=0.21), "P")
        self.assertEqual(classify(1, s_lower=0.2, s_upper=0.2, visible_evidence_complete=True), "P")
        self.assertEqual(classify(1, s_lower=0.1, s_upper=0.2, visible_evidence_complete=True), "A")
        self.assertEqual(classify(1, s_lower=0.1, s_upper=0.199999, upper_bound_validated=True), "T")
        self.assertEqual(classify(1, s_lower=0.19, s_upper=0.21, upper_bound_validated=True), "A")
        # An unvalidated possible polygon does not contradict a reliable lower bound.
        self.assertEqual(classify(1, s_lower=0.3, s_upper=0.1), "P")
        with self.assertRaisesRegex(ValueError, "contradict"):
            classify(1, s_lower=0.3, s_upper=0.1, upper_bound_validated=True)
        with self.assertRaises(ValueError):
            classify(1, upper_bound_validated="yes")
        with self.assertRaises(ValueError):
            classify(0, s_upper=float("nan"))
        with self.assertRaises(ValueError):
            classify(0.5)

    def test_legacy_transition_requires_explicit_new_ontology(self):
        for record in ({"kind": "transition", "E": 1},
                       {"ontology": "temporal-composition", "kind": "transition", "E": 1}):
            with self.assertRaisesRegex(ValueError, "ontology"):
                classify_visual_state(record)
        current = {"ontology": ONTOLOGY, "kind": "transition", "E": 1, "s_lower": 0.01, "s_upper": 0.02}
        self.assertEqual(classify_visual_state(current), "A")  # Legacy word does not certify an upper bound.
        with self.assertRaisesRegex(ValueError, "E must"):
            classify_visual_state({"ontology": ONTOLOGY})

    def test_factorized_probabilities_and_endpoint_semantics(self):
        self.assertEqual(factorized_probs(0, 1), (1, 0, 0))
        self.assertEqual(factorized_probs(1, 0), (0, 1, 0))
        self.assertEqual(factorized_probs(1, 1), (0, 0, 1))
        for e in (0, 0.1, 0.37, 0.9, 1):
            for h in (0, 0.2, 0.73, 1):
                values = factorized_probs(e, h)
                self.assertAlmostEqual(math.fsum(values), 1)
                self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertEqual(binary_cross_entropy(0, 0), 0)
        self.assertEqual(binary_cross_entropy(1, 1), 0)
        self.assertTrue(math.isinf(binary_cross_entropy(1, 0)))
        self.assertTrue(math.isinf(binary_cross_entropy(0, 1)))
        self.assertEqual(factorized_loss(0, 0, n=1, t=0, p=0), 0)
        self.assertEqual(factorized_loss(1, 1, n=0, t=0, p=1), 0)
        for h in (0, 0.3, 1):
            self.assertAlmostEqual(factorized_loss(0.3, h, n=0, t=0, p=0, a=1), -math.log(0.3))
            self.assertEqual(factorized_loss(1, h, n=0, t=0, p=0, a=1), 0)
        self.assertEqual(softmax_partial_loss((0, 0.4, 0.6), n=0, t=0, p=0, a=1), 0)
        self.assertTrue(math.isinf(softmax_partial_loss((1, 0, 0), n=0, t=0, p=0, a=1)))

    def test_complete_ce_partial_loss_and_true_mass_weight(self):
        truths = [dict(n=0.2, t=0.3, p=0.5, a=0), dict(n=0.2, t=0.3, p=0.1, a=0.4),
                  dict(n=0, t=0, p=0, a=1), dict(n=1, t=0, p=0, a=0)]
        for e in (0.01, 0.3, 0.8, 0.99):
            for h in (0.02, 0.4, 0.9):
                probabilities = factorized_probs(e, h)
                for shares in truths:
                    with self.subTest(e=e, h=h, shares=shares):
                        actual = factorized_loss(e, h, **shares)
                        self.assertAlmostEqual(actual, softmax_partial_loss(probabilities, **shares), places=12)
                        mass = shares["t"] + shares["p"]
                        expected = binary_cross_entropy(shares["t"] + shares["p"] + shares["a"], e)
                        if mass:
                            expected += mass * binary_cross_entropy(shares["p"] / mass, h)
                        self.assertAlmostEqual(actual, expected, places=12)
                        if shares["a"] == 0:
                            ce = -math.fsum(shares[label] * math.log(prediction)
                                           for label, prediction in zip(("n", "t", "p"), probabilities))
                            self.assertAlmostEqual(actual, ce, places=12)
        shares = dict(n=0.2, t=0.3, p=0.1, a=0.4)
        expected_difference = 0.4 * (binary_cross_entropy(0.25, 0.2) - binary_cross_entropy(0.25, 0.7))
        for e in (0.1, 0.9):
            self.assertAlmostEqual(factorized_loss(e, 0.2, **shares) - factorized_loss(e, 0.7, **shares),
                                   expected_difference, places=12)  # h weight is true t+p, never predicted e.

    def test_logits_extremes_are_finite_without_probability_clipping(self):
        shares = dict(n=0.25, t=0.25, p=0.25, a=0.25)
        self.assertEqual(factorized_loss_logits(1000, -1000, **shares), 500)
        self.assertEqual(softmax_partial_loss_logits((-1000, 0, -1000), **shares), 500)
        self.assertEqual(softmax_partial_loss_logits((-1000, 1000, -1000), **shares), 1000)
        self.assertTrue(math.isinf(factorized_loss(1, 0, **shares)))
        for ze in (-1000, 1000):
            for zh in (-1000, 1000):
                self.assertTrue(math.isfinite(factorized_loss_logits(ze, zh, **shares)))
                self.assertEqual(factorized_loss_logits(ze, zh, n=0, t=0, p=0, a=1), 1000 if ze < 0 else 0)
        e, h = 0.3, 0.7
        ze, zh = math.log(e / (1 - e)), math.log(h / (1 - h))
        self.assertAlmostEqual(factorized_loss_logits(ze, zh, **shares), factorized_loss(e, h, **shares), places=12)
        log_probabilities = tuple(math.log(value) for value in factorized_probs(e, h))
        self.assertAlmostEqual(softmax_partial_loss_logits(log_probabilities, **shares),
                               softmax_partial_loss(factorized_probs(e, h), **shares), places=12)

    def test_invalid_mass_finiteness_and_predictions_are_rejected(self):
        # Real FP32 values converted to Python floats need not sum to exactly 1.
        fp32 = tuple(struct.unpack("f", struct.pack("f", value))[0] for value in (0.1, 0.2, 0.7))
        self.assertEqual(math.fsum(fp32), 0.9999999925494194)
        self.assertEqual(softmax_partial_loss(fp32, n=1, t=0, p=0), -math.log(fp32[0]))
        self.assertEqual(softmax_partial_loss(fp32, n=0, t=0, p=0, a=1), -math.log(fp32[1] + fp32[2]))
        # The wider prediction tolerance must not normalize inputs or relax truth.
        with self.assertRaisesRegex(ValueError, "Truth shares"):
            softmax_partial_loss((0.1, 0.2, 0.7), n=fp32[0], t=fp32[1], p=fp32[2])
        with self.assertRaisesRegex(ValueError, "probabilities must sum"):
            softmax_partial_loss((0.1, 0.2, 0.6), n=1, t=0, p=0)
        for invalid in (-0.001, 1.001, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    factorized_probs(invalid, 0.5)
                with self.assertRaises(ValueError):
                    factorized_loss(0.5, 0.5, n=invalid, t=0, p=0, a=1)
        for shares in (dict(n=0.2, t=0.2, p=0.2, a=0.2), dict(n=0.5, t=0.5, p=0.5, a=0)):
            with self.assertRaisesRegex(ValueError, "sum to one"):
                factorized_loss(0.5, 0.5, **shares)
        with self.assertRaises(ValueError):
            softmax_partial_loss((0.2, 0.2, 0.2), n=1, t=0, p=0)
        with self.assertRaises(ValueError):
            factorized_loss_logits(float("inf"), 0, n=1, t=0, p=0)
        with self.assertRaises(ValueError):
            softmax_partial_loss_logits((0, float("nan"), 0), n=1, t=0, p=0)


if __name__ == "__main__":
    unittest.main()
