# Versioned experiment evidence

## Directory index

| Location | Purpose / retention |
|---|---|
| `current_image_models/` | Current image checkpoints, bound metrics and training history; aggregate evidence is versioned. |
| `dfdc_unified_20261019/` | Frozen eight-frame video protocol, model heads, calibration and evaluation; preserve paths and hashes. |
| `execution_20261019/` | Historical image/transfer experiments still referenced by the catalog. |
| `debug_refinement_20261019/` | Historical diagnostics and the image-transfer CSV checked by the catalog. |
| `demo_accuracy_20261021_040741/` | Historical continuation evidence, checkpoints and local rollback backups. |
| `demo_presentation_20261020/` | Earlier automatic-gallery verification records. |
| [`xai_audit_20261024/`](xai_audit_20261024/README.md) | Final numerical audit, regression/integration records and organized local UI evidence. |
| `local_archive/setup_20261019/` | Ignored, relocated acquisition/environment snapshots; historical filenames and bytes retained. |
| `champion.json` | Preserved original champion contract. |
| `dependency_storage.json`, `storage_ledger.json` | Local storage accounting; the dependency ledger is read by runtime budgeting. |

Disposable historical logs and temporary smoke/work directories were removed. Repeated post-fix XAI media were removed after exact parity was recorded; the original maps and both numerical summaries remain. See [the cleanup record](../docs/repository_cleanup.md). New logs and regenerated experiment outputs remain ignored.

Git contains the source code, run configurations, documentation and compact evidence needed to review the reported experiments:

- Frozen DFDC protocol, split manifest, checkpoint/summary hashes, metrics, per-video predictions and aggregate charts.
- Completed image-run configurations, summaries, validation scores and aggregate charts.
- Refinement reports, aggregate attribution checks, pilot comparison and the image-transfer prediction CSV used by the model catalog's integrity check.
- Final automatic-gallery UI checks, concurrent/serial attribution comparison and regression summary.
- The 21 October image continuation's aggregate validation/test metrics, epoch history and executed report. Portable PNG/SVG figures and their compact evidence snapshot are in `docs/figures/demo_accuracy_20261021/`.
- Current ResNet/E02 alias updates use `results/current_image_models/` for aggregate metrics, trustworthy epoch histories and the executed report, with PNG/SVG figures in `docs/figures/current_image_models/`. Historical experiment summaries retain their original checkpoint hashes. Per-image scores, rollback backups, latest optimizer states, selected weights and runtime pointers stay local.
- The final XAI audit's compact JSON records and [report](../docs/xai_end_to_end_audit.md), including original/repeated numerical experiments, controls, exact parity, verified catalog/checkpoint identities, and UI/regression summaries. Browser images, prepared face media, raw arrays and console logs remain local.

`.gitignore` allows these files explicitly. New experiment output is ignored by default until its reviewable evidence is deliberately included.

Datasets, original uploads, individual face crops/overlays, attribution arrays, model checkpoints, temporary work directories, smoke/demo media, server state, logs, dependency caches and generated spreadsheets remain local. Some historical report links refer to those local artifacts; they are not part of the Git checkout. Their parent reports and numerical summaries are retained where listed above.

The demo requires separately provisioned checkpoints and local dataset assets; Git does not contain trained weights or bundled data. Existing frozen JSON/CSV files retain their recorded hashes and local paths without rewriting the original experiment evidence. `.gitattributes` disables line-ending conversion for results so checkout does not invalidate their SHA-256 checks.
