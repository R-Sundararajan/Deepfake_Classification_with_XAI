# Final XAI audit evidence — 24 October 2026

Read [the audit report](../../docs/xai_end_to_end_audit.md) for methods, observations, fixes and limitations. No weights were trained or replaced.

## Portable evidence (versioned)

| File | Contents |
|---|---|
| `environment.json`, `shared_assets.json` | Software/device details, verified model catalog/configurations, encoder/detector identities. |
| `inputs.json`, `saved_image_label_search.json` | Original input hashes, sources and the subsequently verified test-label match for fake.jpg. |
| `experiments.json` | Original 14 input/model cases and 87 methods, raw attribution statistics, perturbations, completeness, LIME fidelity and stability. |
| `verified/experiments.json` | Repeated cases after diagnostics changes. |
| `controls.json` | Independent formula references, original label-defect reproduction, controlled masks and video/attention correspondence. |
| `before_after_parity.json` | Exact unchanged predictions and attribution arrays across both passes. |
| `checkpoint_integrity_after.json`, `verified/checkpoint_integrity_after.json` | Checkpoint identity checks after both passes. |
| `verification.json`, `app_gallery_verified.json` | Recorded audit regression and integration outcomes. |
| `ui_evidence_index.json`, `browser_errors.json` | Browser evidence inventory and observed frontend warnings/errors; these are separate from AppTest records. |
| `upload_retry_control.json` | Hash-identical preserved Goodies.jpeg copy used after its original path disappeared. |
| `cleanup_verification.json` | Final cleanup counts, protected-artifact integrity and post-cleanup checks. |

The files describe a small, deliberately selected audit cohort, not a new accuracy benchmark. Four supplied-file labels remain unverified. Personal source paths in records are historical provenance and are not required application configuration.

## Local evidence (ignored)

- `00_*` through `13_*`: original prepared inputs, full per-case JSON, maps and overlays. Their prediction JSON retains its original artifact paths.
- `ui/`: real browser DOM snapshots and screenshots. `ui_evidence_index.json` lists the complete gallery captures.
- `figures/`: method contact sheets and signed-versus-magnitude IG diagnostic plot.
- `arrays/`: raw signed/channel-magnitude IG arrays.
- `logs/`: selected initial/final test, integration, numerical-run and failed-intermediate-export logs.
- `upload_controls/`: the exact-byte preserved image used for the input-availability control.

These media/arrays/logs are intentionally absent from a fresh Git checkout. Links to them in the report are optional local evidence; the compact numerical evidence above is portable. Identical duplicate post-fix media and incomplete exports were discarded during cleanup; the original media and parity records remain. File names inside older snapshots describe their capture-time layout.

## Rerun

From the project root with local assets provisioned:

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m tools.verify_automatic_gallery --output results/gallery_rerun/app_qa.json
.\dfenv\Scripts\python.exe -m tools.verify_parallel_attribution --output results/gallery_rerun/parallel_parity.json
.\dfenv\Scripts\python.exe -m tools.audit_xai_end_to_end --output results/xai_audit_new_run
```

The gallery/concurrency checks use existing held-out dataset fixtures. `--samples PATH` adds optional media to the broader audit; filenames are never treated as ground truth. Use new output names to preserve recorded evidence. One-off control/report writers were retired; the report is maintained as a static document.
