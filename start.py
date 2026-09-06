"""Bootstrap the local dashboard without shell activation (Windows/macOS/Linux)."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def run(command):
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)


def environment_python(root, windows=None):
    windows = os.name == 'nt' if windows is None else windows
    return root / '.venv' / ('Scripts/python.exe' if windows else 'bin/python')


def prepare(root=ROOT):
    python = environment_python(root)
    if not python.exists():
        print('Creation de l’environnement Python...', flush=True)
        run([sys.executable, '-m', 'venv', root / '.venv'])
    fingerprint = hashlib.sha256((root / 'pyproject.toml').read_bytes()).hexdigest()
    marker = root / '.venv' / '.ml-agentic-bootstrap'
    installed = subprocess.run([str(python), '-c', 'import agentic_data.web_app, uvicorn'],
                               cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not installed or not marker.exists() or marker.read_text() != fingerprint:
        print('Installation de ML.agentic (connexion Internet requise)...', flush=True)
        run([python, '-m', 'pip', 'install', '-e', '.[web]'])
        marker.write_text(fingerprint)
    return python


def main():
    parser = argparse.ArgumentParser(description='Demarrer ML.agentic')
    parser.add_argument('--update', action='store_true', help='Mettre a jour la branche courante via git pull --ff-only')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 ou plus recent est requis.')
    if not 1024 <= args.port <= 65535:
        raise RuntimeError('Le port doit etre compris entre 1024 et 65535.')
    if args.update:
        if not shutil.which('git'):
            raise RuntimeError('Git est introuvable.')
        # Git keeps local changes and refuses divergent histories; never reset/stash.
        run(['git', 'pull', '--ff-only'])
        forwarded = [x for x in sys.argv[1:] if x != '--update']
        return subprocess.call([sys.executable, str(ROOT / 'start.py'), *forwarded], cwd=ROOT)
    python = prepare()
    missing = [name for name in ('codex', 'claude', 'ollama', 'copilot') if not shutil.which(name)]
    if len(missing) == 4:
        print('Aucun provider CLI detecte. Le dashboard peut ouvrir vos projets; configurez un provider pour executer les agents.')
    if not shutil.which('docker'):
        print('Docker non detecte : les outils Python du dashboard seront indisponibles.')
    url = f'http://127.0.0.1:{args.port}'
    # Fail before launch if this address is already occupied.
    import socket
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', args.port))
        except OSError as exc:
            raise RuntimeError(f'Port {args.port} occupe. Essayez --port {args.port + 1}.') from exc
    server = subprocess.Popen([str(python), '-m', 'uvicorn', 'agentic_data.web_app:app',
                               '--host', '127.0.0.1', '--port', str(args.port)], cwd=ROOT)
    try:
        for _ in range(150):
            if server.poll() is not None:
                raise RuntimeError('Le serveur s’est arrete. Consultez le message ci-dessus.')
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.2)
        else:
            raise RuntimeError('Le serveur ne repond pas apres 30 secondes.')
        print(f'ML.agentic est pret : {url}\nGardez ce terminal ouvert. Ctrl+C pour arreter.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return server.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Demarrage impossible : {exc}', file=sys.stderr)
        sys.exit(1)
