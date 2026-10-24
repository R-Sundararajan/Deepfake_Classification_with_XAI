# Tools index

Run modules from the repository root with the existing `dfenv`. Use a new output directory when a command records evidence. Local datasets and checkpoints must already be provisioned.

## Demo and verification

| Tool | Purpose |
|---|---|
| `start_demo.ps1` | Start the current Streamlit application on localhost. |
| `verify_automatic_gallery.py` | Current UI integration: uploads, all compatible methods, caching, opacity, frame selection and errors. Uses local held-out fixtures. |
| `verify_parallel_attribution.py` | Compare concurrent and serial maps, targets, hooks and unchanged model scores. |
| `audit_xai_end_to_end.py` | Optional image/video checkpoint audit, perturbations and attribution diagnostics; accepts `--output` and `--samples`. |
| `verify_unified.py`, `verify_refinement.py`, `verify_demo_accuracy.py` | Check retained video/image experiment contracts and recorded deployment evidence. |
| `smoke_transfer.py`, `smoke_dfdc_gpu.py` | Optional architecture/device feasibility checks. Synthetic smoke outputs are not benchmark results. |

The maintained regression suite is `python -m unittest discover -s tests -v`. Keep the tests: they protect class mappings, preprocessing, signed attribution, saturation handling, video context and model-state integrity.

## Runtime dependencies

`runtime.py`, `dfdc_storage.py` and the shared hashing functions in `acquire_dfdc.py` are imported by inference/training. `acquire_dfdc_cli.py` also supplies parsing checks used by tests. Their filenames do not mean they are disposable download scripts.

## Research and reproduction

- `run_unified.py`, `audit_unified_cache.py`, `benchmark_unified.py`, `report_unified.py`: frozen video protocol and reporting.
- `plot_demo_accuracy.py`, `report_demo_accuracy.py`, `finalize_execution.py`, `report_transfer.py`, `export_experiments.py`: retained image metrics, figures and reports. Reporting commands may update derived Markdown; inspect their arguments before running them.
- `diagnose_refinement.py`, `audit_refinement.py`, `profile_refinement.py`, `report_refinement.py`: optional diagnostic/research workflows.
- `acquire_*`, `download_*`, `fetch_yunet.py`, source audits, inventories and acquisition plans: optional data provisioning. No download is needed for an already provisioned demo.
- Workbook/literature builders and validators operate on the parent `research/` workspace. They are separate from the Streamlit application and retain their existing dependencies and source material.

## Retired utilities

The obsolete manual-selector UI verifiers, one-time Streamlit installer, expired deadline watchdog and experiment-specific rollback/restart scripts were removed on 24 October 2026. The current automatic-gallery verifier replaces their UI checks. Historical versions remain recoverable through Git. One-off XAI report/control-generation scripts were also removed after their numerical evidence was retained; the audit report is now a maintained static document.

See [the cleanup record](../docs/repository_cleanup.md) and [results index](../results/README.md) for artifact retention and locations.
