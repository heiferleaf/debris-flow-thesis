"""Exercise seed propagation and rejection through the real paired controller.

Models, CUDA and upload verification are replaced; no training or decoding runs.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('seed_controller', ROOT/'experiments/runspecs/EXP-016/run.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class SeedControllerTests(unittest.TestCase):
    def test_check_and_full_forward_seed_and_reject_other_preflight(self):
        process_parent = ROOT/'data/process/EXP-016'
        result_parent = ROOT/'data/results/exp016'
        with tempfile.TemporaryDirectory(dir=process_parent, prefix='seed-test-') as cp, \
                tempfile.TemporaryDirectory(dir=result_parent, prefix='seed-test-') as rp:
            check_out, full_out = Path(cp)/'check', Path(rp)/'full'
            manifest = Path(cp)/'manifest.csv'
            manifest.write_text('fixture', encoding='utf-8')
            plan_dir = ROOT/'data/metadata/EXP-016/training-plan-v2'
            plan = json.loads((plan_dir/'plan.json').read_text(encoding='utf-8'))
            common = ['run.py', '--seed', '8', '--pixel-reference',
                      str(ROOT/'data/process/EXP-016/dense-pixel-verification-v3/result.json'),
                      '--ffmpeg', __file__, '--transfer-manifest', str(manifest),
                      '--seconds-limit', '7200', '--budget-yuan', '4']
            with patch.object(controller.subprocess, 'run', return_value=SimpleNamespace(stdout='fixture ffmpeg\n')), \
                    patch.object(controller, 'configure_fp32') as configured, \
                    patch.object(controller, 'load_inputs', return_value=(plan, [], {}, [])), \
                    patch.object(controller, 'verify_uploaded_files', return_value={}), \
                    patch.object(controller.threading, 'Thread'), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(controller, 'train', return_value={'seconds': 1}) as trained, \
                    patch.object(controller, 'evaluate', return_value={
                        'videos': {str(i): {'windows': 32537 if i == 0 else 1} for i in range(31)},
                        'metrics': {'rgb_zero': {'status': 'complete'}, 'rgb_diff': {'status': 'complete'}}}):
                with patch('sys.argv', common+['--stage', 'check', '--out', str(check_out)]):
                    controller.main()
                preflight = json.loads((check_out/'preflight.json').read_text(encoding='utf-8'))
                self.assertEqual(preflight['seed'], 8)
                self.assertIn('docs/process/experiments/EXP-016-seed8.md', preflight['code_sha256'])
                full = common+['--stage', 'full', '--preflight-result', str(check_out/'preflight.json'),
                               '--out', str(full_out)]
                with patch('sys.argv', full):
                    controller.main()
                self.assertEqual([(c.kwargs['mode'], c.kwargs['seed'], c.kwargs['epochs'])
                                  for c in trained.call_args_list],
                                 [('rgb_zero', 8, 1), ('rgb_zero', 8, 20), ('rgb_diff', 8, 20)])
                self.assertTrue(all(c.args == (8,) for c in configured.call_args_list))
                self.assertEqual(json.loads((full_out/'launch.json').read_text())['seed'], 8)
                self.assertEqual(json.loads((full_out/'pipeline-result.json').read_text())['seed'], 8)
                trained.reset_mock()
                preflight['seed'] = 7
                (check_out/'preflight.json').write_text(json.dumps(preflight), encoding='utf-8')
                full[-1] = str(Path(rp)/'refused')
                with patch('sys.argv', full), self.assertRaisesRegex(ValueError, 'no longer matches'):
                    controller.main()
                trained.assert_not_called()


if __name__ == '__main__':
    unittest.main()
