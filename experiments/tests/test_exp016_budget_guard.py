"""The budget timeout must stop even when its log cannot be written."""
import importlib.util
from pathlib import Path
import unittest
import csv
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('exp016_launcher', ROOT / 'experiments/runspecs/EXP-016/run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class BudgetGuardTests(unittest.TestCase):
    def test_static_upload_check_hashes_bytes_without_decoding(self):
        parent = ROOT / '.codex/docs/.temp/static-preflight-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            folder = Path(folder)
            media = folder / 'sample.jpg'
            media.write_bytes(b'not an image; static check must not decode')
            manifest = folder / 'files.csv'
            row = dict(path=media.name, size_bytes=media.stat().st_size, sha256=launcher.digest(media))
            with manifest.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            with patch.object(launcher, 'evaluate', side_effect=AssertionError('must not decode')):
                proof = launcher.verify_uploaded_files(folder, manifest, {media.name: row['sha256']})
            self.assertEqual(proof['decoded_frames'], 0)
            self.assertEqual(proof['files_sha_verified'], 1)
            with self.assertRaisesRegex(ValueError, 'cover frozen input'):
                launcher.verify_uploaded_files(folder, manifest, {'missing.jpg': row['sha256']})
            media.write_bytes(b'x' * media.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'Uploaded file changed'):
                launcher.verify_uploaded_files(folder, manifest, {media.name: row['sha256']})

    def test_exit_is_unconditional_when_logging_fails(self):
        with patch.object(launcher.time, 'sleep'), patch.object(Path, 'write_text', side_effect=OSError('disk full')):
            with patch.object(launcher.os, '_exit') as forced_exit:
                with self.assertRaises(OSError):
                    launcher.hard_stop(1, ROOT)
                forced_exit.assert_called_once_with(124)

    def test_normal_timeout_also_exits(self):
        with patch.object(launcher.time, 'sleep'), patch.object(Path, 'write_text'):
            with patch.object(launcher.os, '_exit') as forced_exit:
                launcher.hard_stop(1, ROOT)
                forced_exit.assert_called_once_with(124)


if __name__ == '__main__':
    unittest.main()
