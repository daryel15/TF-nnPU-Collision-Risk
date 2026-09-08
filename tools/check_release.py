"""Check notebook startup from its own directory and compile active Python code."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits

    print('Hashes:', verify_manifest(), flush=True)
    print('Data/splits/masks:', validate_data_and_splits(), flush=True)
    for p in ROOT.rglob('*.py'):
        if set(p.relative_to(ROOT).parts) & {'outputs', '.venv', '__pycache__'}:
            continue
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    print('Python syntax: PASS', flush=True)
    # Test the actual Jupyter execution path; notebooks remain unchanged on disk.
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jupyter-check-', dir=ROOT / 'outputs') as profile:
        previous_profile = os.environ.get('IPYTHONDIR')
        os.environ['IPYTHONDIR'] = profile
        try:
            for p in sorted((ROOT / 'notebooks').glob('*.ipynb')):
                started = time.perf_counter()
                notebook = nbformat.read(p, as_version=4)
                NotebookClient(
                    notebook, timeout=180, kernel_name='python3',
                    resources={'metadata': {'path': str(p.parent)}},
                ).execute()
                print(f'Notebook PASS: {p.name} ({time.perf_counter()-started:.1f}s)', flush=True)
        finally:
            if previous_profile is None:
                os.environ.pop('IPYTHONDIR', None)
            else:
                os.environ['IPYTHONDIR'] = previous_profile
    print('Default notebooks passed. Full training is tested separately with --train --smoke.', flush=True)

if __name__ == '__main__':
    main()
