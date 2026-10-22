"""Snapshot the demo's current files without changing baseline artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/demo_accuracy_20261021_040741'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    backup = RUN / 'baseline/files'
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = [p for p in paths if p and (ROOT / p).is_file()]
    paths.append('models/best_resnet18.pth')
    files = []
    for relative in paths:
        source, target = ROOT / relative, backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        original_hash = digest(source)
        assert original_hash == digest(target), relative
        files.append({'path': relative, 'sha256': original_hash, 'bytes': source.stat().st_size})
    checkpoints = []
    for parent in ['models', 'results', 'data/dfdc/runtime/unified_encoders']:
        for source in (ROOT / parent).rglob('*'):
            if source.is_file() and source.suffix.lower() in {'.pth', '.pt', '.ckpt'} and not source.is_relative_to(RUN):
                checkpoints.append({'path': str(source.relative_to(ROOT)), 'sha256': digest(source)})
    manifest = {
        'project_root': str(ROOT), 'run_root': str(RUN), 'start_utc': '2026-10-21T04:07:41+00:00',
        'deadlines_utc': {'new_experiments': '2026-10-21T04:42:41+00:00', 'training': '2026-10-21T04:47:41+00:00',
                          'finish': '2026-10-21T04:57:41+00:00', 'demo': '2026-10-21T05:07:41+00:00'},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'initial_git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).decode(),
        'files': files, 'baseline_checkpoints': checkpoints,
        'restore_files': ['app.py', 'inference/catalog.py', 'models/best_resnet18.pth'],
        'absent_paths': ['configs/runtime/active_image_model.json'],
        'preprocessing': 'RGB; full frame; PIL bilinear resize 224x224; ImageNet mean/std; FAKE=0, REAL=1; threshold 0.5',
        'baseline_model': 'models/best_resnet18.pth',
    }
    (RUN / 'rollback_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (RUN / 'processes.json').write_text('[]', encoding='utf-8')
    (RUN / 'ROLLBACK.md').write_text('Restore the baseline with:\n\n'
        f'`powershell -ExecutionPolicy Bypass -File "{ROOT / "rollback_demo.ps1"}"`\n\n'
        'The script stops only this run’s registered experiment processes, restores the explicitly listed app/catalog/checkpoint files, '
        'removes only the candidate activation pointer if it was absent at baseline, and restarts this project’s local demo. '
        'Other working files, datasets, video models and experiment results are preserved. The files directory contains a verified '
        'snapshot of the original source and versioned metrics, plus the ignored original image checkpoint. All baseline checkpoint hashes '
        'are recorded. Use -VerifyOnly to validate backup integrity without restoring or stopping anything.\n', encoding='utf-8')
    print(json.dumps({'run': str(RUN), 'verified_backup_files': len(files), 'baseline_checkpoints': len(checkpoints)}, indent=2))


if __name__ == '__main__':
    main()
