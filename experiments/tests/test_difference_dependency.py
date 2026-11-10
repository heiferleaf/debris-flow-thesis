"""Intervention must remove D only and leave the immutable source intact."""
import importlib.util
from pathlib import Path
import unittest
import runpy
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('difference_dependency', ROOT/'experiments/runspecs/EXP-016/diagnose_difference_dependency.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DifferenceDependencyTests(unittest.TestCase):
    def test_relative_cli_source_path_is_normalized(self):
        source = runpy.run_path(str(Path('experiments/runspecs/EXP-016/diagnose_difference_dependency.py')),
                                run_name='diagnostic_path_fixture')
        self.assertTrue(source['SOURCE'].is_absolute())
        self.assertEqual(source['SOURCE'].parent, ROOT/'experiments/runspecs/EXP-016')

    def test_rgb_and_original_input_unchanged(self):
        x = torch.arange(2*6*16*3*3, dtype=torch.float32).reshape(2, 6, 16, 3, 3)
        saved = x.clone()
        zero = diagnostic.zero_difference(x)
        self.assertTrue(torch.equal(saved, x))
        self.assertTrue(torch.equal(zero[:, :3], x[:, :3]))
        self.assertEqual(zero[:, 3:].count_nonzero(), 0)
        self.assertNotEqual(x.data_ptr(), zero.data_ptr())

    def test_invalid_interventions_fail(self):
        for x in [torch.zeros(1, 3, 16, 3, 3), torch.zeros(1, 6, 15, 3, 3),
                  torch.zeros(1, 6, 16, 3, 3, dtype=torch.float64),
                  torch.full((1, 6, 16, 3, 3), float('nan'))]:
            with self.assertRaises(ValueError):
                diagnostic.zero_difference(x)


if __name__ == '__main__':
    unittest.main()
