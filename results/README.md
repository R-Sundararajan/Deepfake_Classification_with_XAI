# Versioned experiment evidence

Git contains the source code, run configurations, documentation and compact evidence needed to review the reported experiments:

- Frozen DFDC protocol, split manifest, checkpoint/summary hashes, metrics, per-video predictions and aggregate charts.
- Completed image-run configurations, summaries, validation scores and aggregate charts.
- Refinement reports, aggregate attribution checks, pilot comparison and the image-transfer prediction CSV used by the model catalog's integrity check.
- Final automatic-gallery UI checks, concurrent/serial attribution comparison and regression summary.
- The 21 October image continuation's aggregate validation/test metrics, epoch history and executed report. Portable PNG/SVG figures and their compact evidence snapshot are in `docs/figures/demo_accuracy_20261021/`.
- Current ResNet/E02 alias updates use `results/current_image_models/` for aggregate metrics, trustworthy epoch histories and the executed report, with PNG/SVG figures in `docs/figures/current_image_models/`. Historical experiment summaries retain their original checkpoint hashes. Per-image scores, rollback backups, latest optimizer states, selected weights and runtime pointers stay local.

`.gitignore` allows these files explicitly. New experiment output is ignored by default until its reviewable evidence is deliberately included.

Datasets, original uploads, individual face crops/overlays, attribution arrays, model checkpoints, temporary work directories, smoke/demo media, server state, logs, dependency caches and generated spreadsheets remain local. Some historical report links refer to those local artifacts; they are not part of the Git checkout. Their parent reports and numerical summaries are retained where listed above.

The demo requires separately provisioned checkpoints and local dataset assets; Git does not contain trained weights or bundled data. Existing frozen JSON/CSV files retain their recorded hashes and local paths without rewriting the original experiment evidence. `.gitattributes` disables line-ending conversion for results so checkout does not invalidate their SHA-256 checks.
