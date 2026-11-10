"""Input aliases must not silently change an evaluator's model input contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
from experiments.common.video.dense_evaluation import checkpoint_input_modes


class InputAliases(unittest.TestCase):
    def test_original_modes_unchanged(self):
        c={'rgb_zero':'a.pt','rgb_diff':'b.pt'}
        self.assertEqual(checkpoint_input_modes(c),{k:k for k in c})
        self.assertEqual(checkpoint_input_modes(None),{})

    def test_three_rgb_models_share_input_only_not_identity(self):
        c={'bce':'a.pt','cross':'b.pt','same':'c.pt'}
        mapping={k:'rgb_zero' for k in c}
        self.assertEqual(checkpoint_input_modes(c,mapping),mapping)
        self.assertEqual(len(checkpoint_input_modes(c,mapping)),3)

    def test_missing_extra_and_unknown_inputs_rejected(self):
        for mapping in ({},{'bce':'rgb_zero','extra':'rgb_diff'},{'bce':'same'}):
            with self.assertRaises(ValueError):checkpoint_input_modes({'bce':'a.pt'},mapping)
        with self.assertRaises(ValueError):checkpoint_input_modes({'same':'a.pt'})


if __name__=='__main__':unittest.main()
