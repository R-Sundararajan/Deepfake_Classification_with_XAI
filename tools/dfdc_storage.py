"""Conservative 30 GiB phase budget. No cleanup or deletion of existing data."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
LIMIT = 30 * GIB
HEADROOM = 5 * GIB
PHASE_DIRS = ('data/dfdc', 'results', 'tools', 'tests', 'docs', 'configs')
RESEARCH = ROOT.parent / 'research'


def tree_bytes(folder):
    total = 0
    for parent, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d != 'node_modules' and not Path(parent, d).is_symlink()]
        for name in files:
            try:
                total += Path(parent, name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def phase_bytes():
    # These phase directories were absent at the initial audit. Count complete
    # directories, including .part, PDFs rendered for verification, and caches.
    extra = sum(tree_bytes(ROOT / d) for d in PHASE_DIRS) + tree_bytes(RESEARCH)
    # Conservatively include code and its caches as well as all new model weights,
    # media, extracted crops, temporary files and outputs. Existing environment,
    # old image dataset/checkpoint are baseline storage and are never duplicated.
    for d in ['training', 'evaluation', 'models', 'inference', 'explainability']:
        extra += tree_bytes(ROOT/d)
    extra += sum(p.stat().st_size for p in [ROOT/'.gitignore', ROOT/'app.py'] if p.exists())
    dependency_ledger=ROOT/'results/dependency_storage.json'
    if dependency_ledger.exists():
        ledger=json.loads(dependency_ledger.read_text(encoding='utf-8'))
        extra += ledger.get('environment_added_bytes',ledger.get('environment_reserved_bytes',0))
    # Later phases retain the original ledger and record only their incremental
    # package footprint, so newly installed encoders still count toward 30 GiB.
    for ledger_path in (ROOT/'results').glob('*/dependency_storage.json'):
        ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
        extra+=max(0,ledger.get('added_since_prior_phase',0))
    return extra


def owned_path(path):
    p = Path(path).resolve()
    if p.drive.upper() != 'D:':
        raise ValueError('DFDC phase outputs must remain on D:')
    if not (any(p.is_relative_to((ROOT / d).resolve()) for d in PHASE_DIRS) or p.is_relative_to(RESEARCH.resolve())):
        raise ValueError('Output is outside the phase-owned project directories')
    return p


def check_budget(additional=0, used=None, free=None):
    if additional < 0:
        raise ValueError('Reservation must be non-negative')
    used = phase_bytes() if used is None else used
    free = shutil.disk_usage(ROOT).free if free is None else free
    if used + additional > LIMIT:
        raise RuntimeError(f'30 GiB phase budget would be exceeded: {used + additional} bytes')
    if free - additional < HEADROOM:
        raise RuntimeError('Insufficient disk space to retain 5 GiB free headroom')
    return {'phase_bytes': used, 'free_bytes': free, 'limit_bytes': LIMIT, 'headroom_bytes': HEADROOM}


def atomic_json(path, data):
    p = owned_path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
    check_budget(len(payload) + 4096)
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_suffix(p.suffix + '.part')
    if part.exists():
        raise FileExistsError(f'Existing unfinished output preserved: {part}')
    with part.open('xb') as f:
        f.write(payload)
    part.replace(p)


if __name__ == '__main__':
    print(json.dumps(check_budget(), indent=2))
