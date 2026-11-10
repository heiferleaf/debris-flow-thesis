"""Three narrow in-memory transport tests; never connect or read weights."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('collect_exp022_fixture', ROOT / 'experiments/runspecs/EXP-022/collect_results.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Channel:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class SFTP:
    def __init__(self):
        self.channel = Channel()

    def get_channel(self):
        return self.channel

    def native_nonblocking_operation(self):
        self.get_channel().settimeout(0)


class TransportTests(unittest.TestCase):
    def test_helper_wait_expands_to_thirty_without_changing_native_channel(self):
        raw = SFTP(); view = M.HelperSFTP(raw, 200., clock=lambda: 100.)
        view.get_channel().settimeout(5.)
        self.assertEqual(raw.channel.timeouts, [30.])
        self.assertIs(raw.get_channel(), raw.channel)
        view.native_nonblocking_operation()
        self.assertEqual(raw.channel.timeouts, [30., 0])

    def test_remaining_global_deadline_tightens_wait_and_expiry_refuses_read(self):
        channel = Channel(); now = [198.]
        view = M.HelperChannel(channel, 200., clock=lambda: now[0])
        view.settimeout(2.)
        self.assertEqual(channel.timeouts[-1], 2.)
        now[0] = 199.9; view.settimeout(.1)
        self.assertAlmostEqual(channel.timeouts[-1], .1)
        now[0] = 200.
        with self.assertRaises(TimeoutError):
            view.settimeout(.001)
        self.assertEqual(len(channel.timeouts), 2)

    def test_collect_only_and_temporary_hook_leave_original_source_unchanged(self):
        before = M.verify_operator()  # Small source SHA only, no operator execution.
        for action in ('launch', 'sync', 'shutdown', 'status'):
            with self.subTest(action=action), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    M.parse_args([action, '--config', 'fixture.json'])
        args = M.parse_args(['collect', '--config', 'fixture.json', '--once'])
        raw = SFTP(); calls = []; original_argv = sys.argv
        original_open = lambda client, end: raw
        operator = SimpleNamespace(COLLECT=SimpleNamespace(bounded_sftp=original_open))
        def main():
            self.assertEqual(sys.argv[1], 'collect')
            self.assertIn('--once', sys.argv)
            proxy = operator.COLLECT.bounded_sftp(object(), M.time.time() + 100)
            proxy.get_channel().settimeout(5)
            raise RuntimeError('original collector exit fixture')
        operator.main = main
        with self.assertRaisesRegex(RuntimeError, 'original collector exit'):
            M.run_collect(operator, args, calls.append)
        self.assertEqual(raw.channel.timeouts, [30.])
        self.assertIs(operator.COLLECT.bounded_sftp, original_open)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(calls[0]['helper_read_wait_cap_seconds'], 30.)
        self.assertEqual(M.verify_operator(), before)


if __name__ == '__main__':
    unittest.main()
