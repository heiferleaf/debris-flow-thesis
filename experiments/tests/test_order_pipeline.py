"""Pure budget/CLI fixtures only; never starts the pipeline, cloud or shutdown."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('order_pipeline_fixture',ROOT/'experiments/runspecs/EXP-019/pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


class PipelineBudgetTests(unittest.TestCase):
    def budget(self,**changes):
        args=dict(now=1000.,gpu_start=900.,platform_deadline=8100.,price=1.48,
                  prior_gpu_seconds=0.,prior_stage_yuan=0.,prior_total_yuan=1.59,seconds_limit=6600.)
        args.update(changes);return pipeline.budget_deadline(**args)

    def test_elapsed_boot_and_prior_gpu_time_are_not_reset(self):
        first=self.budget();self.assertEqual(first['remaining_seconds'],6540.)
        continued=self.budget(prior_gpu_seconds=1800.,platform_deadline=6300.)
        self.assertEqual(continued['deadline_unix'],6240.)
        self.assertLessEqual(continued['deadline_unix']+60-900+1800,7200.)
        with self.assertRaises(ValueError):self.budget(gpu_start=-7000.)

    def test_stage_and_total_money_ceiling_and_platform_timer(self):
        b=self.budget(prior_stage_yuan=3.5,prior_total_yuan=19.6)
        spend=(b['deadline_unix']+60-900)*1.48/3600
        self.assertLessEqual(3.5+spend+.05+.10,4.+1e-9)
        self.assertLessEqual(19.6+spend+.05+.10,20.+1e-9)
        for kwargs in (dict(price=1.49),dict(prior_stage_yuan=4),dict(prior_total_yuan=20),
                       dict(platform_deadline=8102),dict(price=float('nan')),dict(gpu_start=1010)):
            with self.assertRaises(ValueError):self.budget(**kwargs)

    def test_runtime_forecast_does_not_select_by_scores(self):
        probes={mode:dict(status='runtime_probe_complete',completed_epochs=1,optimizer_steps=21,
                          seconds=30.,wall_seconds=40.,best_validation_bce=1e9)
                for mode in pipeline.MODES}
        forecast=pipeline.forecast(probes)
        self.assertEqual(forecast['remaining_forecast_seconds'],4320.)
        self.assertGreaterEqual(forecast['dense_reserve_seconds'],1.25*pipeline.DENSE_OBSERVED_THREE_MODELS)
        for row in probes.values():row['best_validation_bce']=0.
        self.assertEqual(forecast,pipeline.forecast(probes))
        probes[pipeline.MODES[0]]['optimizer_steps']=20
        with self.assertRaises(ValueError):pipeline.forecast(probes)

    def test_deployment_interface_lists_new_execution_sources(self):
        paths={p.relative_to(ROOT).as_posix() for p in pipeline.code_paths()}
        self.assertIn('experiments/runspecs/EXP-019/pipeline.py',paths)
        self.assertIn('experiments/runspecs/EXP-019/run.sh',paths)
        self.assertIn('experiments/common/video/order_cache_acceptance.py',paths)
        self.assertIn('experiments/common/video/paired_plan.py',paths)
        argv=['--plan','p','--out','o','--cache-acceptance','c','--pixel-reference','r',
              '--ffmpeg','f','--transfer-manifest','t']
        args=pipeline._parser().parse_args(argv)
        self.assertEqual(args.stage,'check')
        self.assertIsNone(args.gpu_start_unix)

    def test_hard_deadline_leaves_failure_evidence_without_a_real_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(pipeline,'_request_shutdown',return_value={'returncode':0}) as shutdown,patch.object(pipeline.os,'_exit') as terminate:
                pipeline._hard_stop(0.,out,threading.Event())
            shutdown.assert_called_once_with(out)
            terminate.assert_called_once_with(124)
            evidence=json.loads((out/'hard-time-limit.json').read_text(encoding='utf-8'))
            self.assertEqual(evidence['status'],'time_limit_reached')
            self.assertTrue(evidence['metrics_withheld'])
            stopped=threading.Event();stopped.set()
            with patch.object(pipeline,'_request_shutdown') as shutdown,patch.object(pipeline.os,'_exit') as terminate:
                pipeline._hard_stop(0.,out,stopped)
            terminate.assert_not_called()
            shutdown.assert_not_called()

    def test_shutdown_timeout_and_unwritable_evidence_still_exit124(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(pipeline,'_request_shutdown',side_effect=TimeoutError('fixture')) as shutdown,patch.object(pipeline.os,'_exit') as terminate:
                pipeline._hard_stop(0.,out,threading.Event())
            shutdown.assert_called_once_with(out);terminate.assert_called_once_with(124)
            evidence=json.loads((out/'hard-stop-shutdown.json').read_text(encoding='utf-8'))
            self.assertEqual(evidence['status'],'shutdown_command_failed')
            self.assertEqual(evidence['error_type'],'TimeoutError')
            with patch.object(pipeline,'write_json',side_effect=OSError('fixture disk full')),patch.object(pipeline,'_request_shutdown',return_value={'returncode':0}) as shutdown,patch.object(pipeline.os,'_exit') as terminate:
                pipeline._hard_stop(0.,out,threading.Event())
            shutdown.assert_called_once_with(out);terminate.assert_called_once_with(124)


if __name__=='__main__':unittest.main()
