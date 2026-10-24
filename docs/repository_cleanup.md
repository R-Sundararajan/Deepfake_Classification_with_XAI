# Repository cleanup — 24 October 2026

The final cleanup keeps the deployed image/video application, trained checkpoints, local data, frozen experiment contracts, research documents and useful regression/reproduction tools.

## Removed

- `tools/smoke_app.py`, `tools/verify_demo.py`, `tools/verify_app_refinement.py`: superseded checks for removed method/family/target controls and older metric layouts. Use `tools.verify_automatic_gallery`.
- `tools/install_streamlit.py`: completed one-time installation workflow; `requirements.txt` remains the environment specification.
- `tools/demo_deadline_watchdog.ps1`, `tools/prepare_demo_rollback.py`, `tools/restart_demo_verified.ps1`, `rollback_demo.ps1`: expired 21 October deadline/deployment helpers. They could restore an older app or checkpoint; they are not current deployment tools. Historical scripts and their matching manifests remain recoverable through Git/local backups.
- One-off audit control/report-generation scripts: their measured records and the static audit report are retained. The broader `tools.audit_xai_end_to_end` runner remains available.
- Disposable historical console logs, empty/temporary test work directories, failed partial audit exports, and duplicate post-fix media/arrays already proven identical by the saved parity record.

No trained weights, datasets, frozen JSON/CSV contracts, active model pointers, research PDFs/workbooks or regression tests were removed. Existing hash-bound result paths were preserved.

## Organization

`results/` now has a directory index. Root-level setup snapshots moved to ignored `results/local_archive/setup_20261019/`; the dependency/storage ledgers retain their original root locations. Existing benchmark run directories retain their paths because model loading and provenance checks depend on them.

`results/xai_audit_20261024/` keeps compact, versioned numerical evidence at its root. Local browser snapshots live in `ui/`, diagnostic plots/contact sheets in `figures/`, raw IG diagnostic arrays in `arrays/`, and selected execution logs in `logs/`. The original per-case artifacts remain in their named case directories. `verified/` retains post-fix numerical records and checkpoint-integrity results; identical duplicate media were discarded.

Face screenshots, uploads, arrays, logs, weights and runtime state remain excluded from Git. The [audit artifact index](../results/xai_audit_20261024/README.md) distinguishes portable evidence from optional local files. Older reports can reference local outputs or retired scripts; their historical numerical evidence was not rewritten.

## Verification

Before cleanup, SHA-256 identities were recorded for all tracked result artifacts, all existing result/model checkpoints, and runtime model pointers. The final cleanup verification checks these identities, the image/video catalog, Python source imports and documentation links, then reruns the 74-test suite and current gallery/concurrency verifiers. Executed results and removal/move counts are recorded in [cleanup verification](../results/xai_audit_20261024/cleanup_verification.json).

The post-cleanup run passed all 74 regression tests and 16 gallery integration cases covering 63 maps. Concurrent/serial comparison passed for the current image champion and S8 video system (14 maps total), with predictions unchanged. The maintained verifiers now use held-out dataset fixtures and support both image and video predictor module layouts.

The XAI change fixes score-space labeling and exposes occlusion effect size. Predictions, attribution calculations and model weights remain as audited. The requested commit title does not imply that unusual but mathematically valid maps were cosmetically changed.
