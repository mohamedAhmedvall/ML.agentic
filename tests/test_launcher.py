import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bootstrap', ROOT / 'start.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class LauncherTests(unittest.TestCase):
    def test_venv_paths_support_windows_and_unix(self):
        root = Path('project with spaces')
        self.assertEqual(bootstrap.environment_python(root, True), root / '.venv/Scripts/python.exe')
        self.assertEqual(bootstrap.environment_python(root, False), root / '.venv/bin/python')

    def test_install_failure_is_not_marked_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'pyproject.toml').write_text('[project]')
            python = bootstrap.environment_python(root); python.parent.mkdir(parents=True); python.touch()
            with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)), patch.object(bootstrap, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')):
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.prepare(root)
            self.assertFalse((root / '.venv/.ml-agentic-bootstrap').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows command launcher')
    def test_windows_launcher_in_directory_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='ml agentic ') as tmp:
            for name in ['start.cmd', 'start.py']:
                shutil.copy2(ROOT / name, Path(tmp) / name)
            result = subprocess.run(['cmd.exe', '/c', str(Path(tmp) / 'start.cmd'), '--help'], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--update', result.stdout)
