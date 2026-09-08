"""Regenerate tracked release hashes after intentional edits."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
excluded={'outputs','.git','.venv','__pycache__','.ipynb_checkpoints','.pytest_cache'}
lines=[]
for p in sorted(root.rglob('*')):
    relative=p.relative_to(root)
    if not p.is_file() or set(relative.parts)&excluded or p.name=='MANIFEST_SHA256.txt' or p.suffix=='.pyc': continue
    lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+relative.as_posix())
(root/'MANIFEST_SHA256.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Manifest entries:',len(lines))
