# DFDC phase: preparation and baseline

This is the historical baseline-only workflow. Use [TRANSFER_LEARNING.md](TRANSFER_LEARNING.md) for the active pipeline, source-safe validation protocol, training, current face detection, application setup and experiment workbook. Keep the final test untouched during model development.

Run commands from `D:\sundarCollege\Deepfake_Detect\Deepfakereserch` in PowerShell. Reuse the existing `dfenv`; no package installation, new environment, architecture implementation or DFDC training is required. The original checkpoint and image-data pipeline are unchanged.

## Official access and download

The official [Kaggle competition](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) currently lists the sample as **400 individual MP4 files plus metadata.json**, totaling **1,870,973,085 bytes (1.7425 GiB)**. Direct acquisition avoids an archive/extracted duplicate. The implementation has no full-dataset download option.

The local token could list files, but the first metadata download returned **HTTP 403**. Sign in to the account associated with that token and accept/join through the [competition rules](https://www.kaggle.com/competitions/deepfake-detection-challenge/rules), if Kaggle permits it. If entry/download access is unavailable, obtain permission from Kaggle/Meta; this pipeline does not use mirrors or bypass access controls. Credentials stay in your environment (`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`); never add them to Git or reports.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force data/dfdc/tmp | Out-Null
$env:TEMP = (Resolve-Path data/dfdc/tmp).Path
$env:TMP = $env:TEMP
# List/estimate only, then explicitly acquire the selected official sample:
.\dfenv\Scripts\python.exe -m tools.acquire_dfdc
.\dfenv\Scripts\python.exe -m tools.acquire_dfdc --download
```

The downloader preflights total project usage, selected remote sizes, a 256 MiB work allowance, the **30 GiB additional phase cap**, and **5 GiB remaining D: headroom**. It checks storage during transfers and records original URLs, sizes and SHA-256 digests in `data/dfdc/provenance.json`; cross-host redirects strip credentials. It reuses existing sample files only when their size and digest match recorded provenance. Unverified files and interrupted `.part` files are preserved and cause a safe stop; inspect them manually before retrying. No automatic cleanup touches existing data.

`python -m tools.dfdc_storage` reports conservative phase usage, including `../research/working`, temporary files and caches. Counted phase directories are documented in that module; keep phase outputs inside those directories. No archives, decoded-frame caches or additional model checkpoints are retained by this workflow.

## Prepare deterministic source-group splits

```powershell
.\dfenv\Scripts\python.exe -m training.dfdc_data
```

This reads official sample metadata, checks REAL/FAKE labels and filenames, hashes/probes each video, and writes `data/dfdc/prepared_seed42/manifest.csv` and `preparation.json`. Missing/corrupt videos are recorded without deletion. Every fake and its original, including siblings whose original is outside the sample, stay in one component. Whole components are allocated approximately 70/15/15 to train/validation/test using seed 42 and a deterministic class-balance objective. Splits are generated from labeled training data, **not official competition test splits**. A missing FAKE→original relationship blocks splitting. Too few components or single-class splits produce an error or explicit limitation; source groups are never broken to force balance.

Source-video separation cannot establish identity separation: the sample metadata lacks reliable actor IDs. The manifest preserves original metadata split fields separately. Preparation refuses to overwrite an existing prepared directory; use `--output data/dfdc/prepared_seed43 --seed 43` for an intentional new manifest.

## Evaluate the existing checkpoint

```powershell
# Initial cross-dataset assessment across the complete labeled sample:
.\dfenv\Scripts\python.exe -m evaluation.dfdc --split all --faces full_frame --run-id resnet18_dfdc_sample_all
# Alternatively use only the generated held-out partition:
.\dfenv\Scripts\python.exe -m evaluation.dfdc --split test --faces full_frame --run-id resnet18_dfdc_sample_test
```

Using all partitions is appropriate only for the original image-trained checkpoint, which has seen no DFDC data. Do not treat an all-partitions run as held-out evidence after any DFDC adaptation. This phase does not fine-tune or select thresholds on DFDC.

Defaults: deterministic one-second sampling, at most 16 uniformly retained sample positions per video, one frame in GPU memory at a time, RGB resize to 224×224 and the existing ImageNet normalization. **FAKE is checkpoint output index 0**; video score is the arithmetic mean of frame FAKE probabilities, classified FAKE at ≥0.5. Metrics use FAKE as positive; the confusion matrix uses REAL/FAKE rows and columns. Undefined AUC is JSON `null`; zero-denominator precision/recall/F1 use zero. Counts, class distribution, source groups, failure coverage, hashes, timings, device and peak allocated GPU memory accompany every real run. Failed videos are excluded as whole videos; partial frame rows remain auditable.

The installed OpenCV 5.0.0 build can decode videos but lacks `CascadeClassifier`. The working default therefore uses **full frames, without face detection/alignment**. This adds background/scale shift relative to face-image training and must be reported. Optional `--faces haar` works only in a compatible OpenCV build with its bundled cascade; it selects the largest face, margin 1.3, and explicitly counts full-frame fallbacks when no face is found. It does not track identities. No existing package is changed to add this option.

Results go to a new `results/dfdc_runs/<run-id>/` directory: `summary.json`, video predictions and frame predictions. Official provenance must verify before a real run can start. The model is loaded with `weights_only=True` and no pretrained-weight download. The checkpoint digest is checked again after evaluation. Measured GPU memory is PyTorch peak allocated memory, not the entire driver's reserved footprint.

## Validate and rebuild the workbook

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m tools.smoke_dfdc_gpu
```

The GPU smoke test uses synthetic video and the real checkpoint; it **does not measure DFDC performance** and is excluded from the local-results worksheet.

Workbook authoring uses the existing bundled artifact tool rather than changing project dependencies:

```powershell
$bundle = 'C:\Users\Rangarajan\.cache\codex-runtimes\codex-primary-runtime\dependencies'
$env:DFDC_NODE_MODULES = "$bundle\node\node_modules"
# Refresh transcription JSON from the curated original-paper evidence:
& "$bundle\python\python.exe" tools/curate_dfdc_literature.py
& "$bundle\node\bin\node.exe" tools/build_dfdc_workbook.mjs
& "$bundle\python\python.exe" tools/validate_dfdc_workbook.py
```

The builder automatically adds only completed real official-data runs to **Local DFDC Baseline**. `../research/analyses/dfdc_literature.json` is the auditable transcription, not a scrape of the old workbook's unverified claims. **Source Audit** includes all 112 distinct original PDFs and exclusion reasons; `pdf_###` keys in result source cells identify the exact audited document. **Reading Notes** defines units, missing values and deduplication. Main metrics use accuracy 0–100 and F1/AUC 0–1. Preview, frame, video, validation and cross-dataset protocols must be compared separately.

For a fresh source-text extraction, run `tools/audit_dfdc_sources.py` with bundled Python first; it reads the original workbook, papers and local ZIP archives without modifying them. Keep its `.work` outputs ignored. Do not overwrite the original literature workbook. Paper-specific architectures remain literature references only.
