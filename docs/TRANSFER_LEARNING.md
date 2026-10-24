# DFDC transfer-learning implementation

This is the historical initial transfer-stage record. Its acquisition status, four-frame protocol and deferred-method statements describe that stage. For the deployed eight-frame systems and automatic XAI gallery, use the [current README](../README.md), [demo guide](automatic_gallery.md), and [24 October audit](xai_end_to_end_audit.md).

The pipeline is implemented and validated with synthetic software fixtures. **Official Kaggle sample download returns HTTP 403. No DFDC model has been trained, no DFDC score measured, and no champion selected.** An ImageNet backbone is not automatically a deepfake detector. Software tests and memory checks are excluded from the research leaderboard.

## Repository and research locations

`Deepfakereserch` is the implementation Git root. `Deepfake_Detect` is its parent research workspace, not a Git root. The original uncommitted DFDC modules and tools were preserved and extended. No commits, resets or checkpoint replacements were made.

The authoritative master stays at `../Literature_Analysis.xlsx`. Research workbooks are in `../research/workbooks/`, including `Existing_Six_Papers_Detailed_Analysis.xlsx`, `DFDC_Research_Comparison.xlsx`, and `DFDC_Transfer_Learning_Experiments.xlsx`. Reports and supporting PDF material are in the parent research tree. `../research/analyses/relocation_audit.json` records every original/destination hash. Checkpoints, dataset bytes, caches, uploads and model downloads are ignored by Git.

## Research decisions

The six-paper analysis and original PDF fingerprints were checked before selecting components. Paper 2 motivates lightweight attention and video probability averaging. Paper 35 motivates an economical CNN/frequency fusion ablation. Papers 9 and 45 use FF++ and supply methodological ideas, not DFDC baseline scores. Papers 15 and 16 are surveys, not independent experiments. Published scores do not populate the local leaderboard.

Three initial backbones are implemented: torchvision ResNet-18 (`IMAGENET1K_V1`), EfficientNet-B0 (`IMAGENET1K_V1`), and MobileNetV3-Large (`IMAGENET1K_V2`). Weights and versioned source URLs are recorded with each real checkpoint. Architectures use a binary head with **FAKE index 0, REAL index 1**, preserving the legacy model's class mapping. Evaluation converts FAKE to positive label 1 independently of output index.

ResNet-18 ECA and CBAM variants add attention after the last spatial feature map. EfficientNet/MobileNet native squeeze-and-excitation remains part of those baseline backbones. The frequency hybrid concatenates 16 normalized radial FFT log-power bands with the CNN embedding. No hybrid benefit is assumed. Compact ViT, two-CNN fusion, trainable attention pooling, prototypes, LIME/SHAP and expensive perturbation sweeps are deferred pending baseline evidence and available memory.

## Environment and storage

Use the existing `dfenv`. Streamlit was listed in requirements but missing; Streamlit 1.66.0 and its missing dependencies were installed with every pre-existing package version constrained and verified unchanged. The measured environment increase is recorded in `results/dependency_storage.json`; downloads and pip caches stayed on D:. CUDA PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128 work on the RTX 4060 Laptop GPU. OpenCV 5 lacks Haar cascades but supports YuNet. The official 232,589-byte YuNet detector was downloaded to D: and successfully exercised on a pre-existing local face image; its immutable source commit and SHA-256 are saved beside it.

`tools.runtime` directs PyTorch downloads, temporary files and caches to `data/dfdc/runtime` on D:. `tools.dfdc_storage` conservatively counts research files, controlled output/cache directories, implementation modules and the measured environment increase, including the old baseline checkpoint, against 30 GiB. It retains at least 5 GiB disk headroom. This overcounts some existing files; it is an upper bound, not an exact additional-byte estimate. The Streamlit installation used a 1 GiB reservation and its measured installed bytes plus D: cache are included.

The hardware audit found approximately 15.3 GiB physical RAM but only 0.7 GiB available at its snapshot. Free memory fluctuated during testing. Close memory-heavy applications before real training. The run uses zero loader workers, batch 4, accumulation 8, mixed precision, frozen BatchNorm statistics, sequential model loading and a four-frame crop cache. GPU smoke tests use batch 2 and do not guarantee a larger training configuration fits. Source-independent implementation references: [torchvision models/weights](https://docs.pytorch.org/vision/stable/models.html), [Streamlit's published package](https://pypi.org/project/streamlit/1.66.0/), and [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing).

## Acquire the official sample

Run commands from `D:\sundarCollege\Deepfake_Detect\Deepfakereserch`:

```powershell
.\dfenv\Scripts\python.exe -m tools.runtime
.\dfenv\Scripts\python.exe -m tools.acquire_dfdc_cli
.\dfenv\Scripts\python.exe -m tools.acquire_dfdc_cli --download
```

Only the official DFDC competition is requested. The current CLI inventory exposes 400 sample MP4s plus `metadata.json`, totaling 1,870,973,085 bytes (1.7425 GiB), rather than `train_sample_videos.zip`. Individual official acquisition avoids simultaneous archive/extracted copies. The CLI checks declared size, SHA-256 and provenance, preserves unverified existing files, and refuses an entire competition download. If Kaggle later exposes only a ZIP, the tool stops before downloading until expanded storage is inspected.

Authentication and file listing work, but `DownloadDataFile` returns **HTTP 403 Forbidden** for `train_sample_videos/metadata.json`. Review access for the account tied to the local token at [the competition rules page](https://www.kaggle.com/competitions/deepfake-detection-challenge/rules). Signing in does not establish competition authorization. No credentials are printed or stored in reports. No other dataset or mirror was used. Additional chunks cannot yet be justified because the sample itself is unavailable.

## Prepare the shared benchmark

```powershell
.\dfenv\Scripts\python.exe -m training.dfdc_data
.\dfenv\Scripts\python.exe -m tools.fetch_yunet
.\dfenv\Scripts\python.exe -m training.dfdc_cache
```

Preparation verifies metadata, labels, file hashes and video decoding, and probes frame rate, frame count and resolution. Duration is frame count / FPS. Approximately 70/15/15 training/validation/test partitions are formed with seed 42, grouping each original and its derivatives, including siblings whose original is absent. Optional official actor IDs are grouped if supplied. The sample normally lacks actor IDs; identity disjointness is not established. Manifest loading rejects duplicate videos crossing splits, source leakage, inconsistent derivatives, class-map errors and altered hashes.

The cache deterministically retains four evenly spread positions on a one-second grid. YuNet detects on an image with maximum side 640; the largest face is selected, with optional eye/nose landmark affine alignment and margin 1.3. A missed face explicitly falls back to the full frame and is counted. Multiple identities are not tracked. RGB crops are saved losslessly at 224×224, then ImageNet-normalized during loading. No full-resolution frame cache is retained. Validation/test transforms are deterministic. An incomplete cache blocks training rather than silently changing evaluation coverage.

An explicit full-frame alternative is available, with its changed protocol recorded:

```powershell
.\dfenv\Scripts\python.exe -m training.dfdc_cache --faces full_frame --no-align --output data/dfdc/cache_full_frame_4
```

All experiments must use the intended shared cache. Do not regenerate random splits or mix crops from different manifests. Cache/preparation commands refuse to overwrite existing directories.

## Baseline, then one training run

```powershell
# Initial cross-dataset assessment of the unchanged image-trained checkpoint.
# Validation only: the final test remains untouched for the future champion.
.\dfenv\Scripts\python.exe -m evaluation.experiments baseline

# First controlled transfer-learning run.
.\dfenv\Scripts\python.exe -m training.experiment --config configs/resnet18_frozen.json

# Resume an interrupted run from its last completed epoch.
.\dfenv\Scripts\python.exe -m training.experiment --config configs/resnet18_frozen.json --resume
```

Defaults: seed 42, input 224, four frames/video, batch 4, accumulation 8 (effective 32 for full batches), three frozen-head epochs, AdamW, head LR 0.001, weight decay 0.0001, class-weighted cross-entropy, AMP, early-stopping patience 2 and a 30-minute limit per invocation. Final incomplete accumulation groups use their actual step count. Basic augmentation is horizontal flip 0.5 and moderate color jitter. The JPEG ablation adds quality 50..95 at probability 0.5; the noise ablation adds Gaussian RGB sigma 0.01..0.03 at probability 0.5. These are separate ablations against basic augmentation.

`resnet18_selective.json` starts from A1's best frozen checkpoint, then trains layer4 for three additional epochs at backbone LR 0.00001. Extra epochs and cost are explicitly reported. It is not a matched-total-epoch comparison. BatchNorm running statistics stay fixed. Checkpoints preserve model/head configuration, class mapping, preprocessing, manifest hash, optimizer, scaler and RNG state. Resume preserves the original configuration and total recorded time. A completed experiment cannot be overwritten.

Results are written to a unique ignored `results/experiments/<experiment_id>/`: `config.json`, `last.pt`, `best.pt`, `summary.json`, validation frame/video predictions and confusion/ROC/PR plots. Best checkpoints and stopping use validation video AUC, balanced accuracy and F1. Parameters, duration, peak allocated GPU memory, peak process RAM, checkpoint bytes and timing are recorded. Timing includes a model warm-up and clearly distinguishes cached-crop model latency from end-to-end decoding/detection/XAI.

There are nine planned configurations, not an automatic queue: three frozen baselines, a selective-tuning comparison, ECA/CBAM, JPEG/noise, and frequency fusion. Run the other baselines individually after reviewing A1. Run later stages only when earlier validation evidence and resource use justify them. Adjust the reference CNN before attention/augmentation experiments if another backbone is the stronger practical choice.

## Select and test once

```powershell
.\dfenv\Scripts\python.exe -m evaluation.experiments select
.\dfenv\Scripts\python.exe -m evaluation.experiments test
```

Selection verifies checkpoint and prediction hashes and accepts only completed official-data training runs with identical manifest and preprocessing. It ranks validation video ROC-AUC, balanced accuracy and F1, rounded to 0.001 for ties, then lower cached inference latency. Synthetic checks and the old cross-dataset baseline are ineligible. `results/champion.json` seals the choice. The final test is source-group held out, not Kaggle's hidden competition test.

The final-test directory is claimed before predictions start. Existing selection/test directories prevent silent reruns, including after an interrupted final run; review the failure before any manually authorized recovery. Final results include frame/video metrics, confusion matrix, ROC/PR plots and 500-draw source-group bootstrap intervals. Small-sample intervals condition on this fixed trained model and sample and do not establish external generalization.

## Rebuild the research workbook

```powershell
.\dfenv\Scripts\python.exe -m tools.export_experiments
& 'C:\Users\Rangarajan\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' tools/build_transfer_workbook.mjs
```

The builder uses Codex's bundled artifact tool through the ignored `tools/node_modules` junction. The export independently recomputes metrics from hashed predictions and excludes synthetic runs. Output: `../research/workbooks/DFDC_Transfer_Learning_Experiments.xlsx`. Its five sheets separate validation, ablations, final test, interpretation and charts. Until real experiments complete, metric cells are blank and planned configurations say **Not evaluated**. No literature scores or synthetic-fixture scores fill missing observations.

## Image/video application and explanations

```powershell
.\dfenv\Scripts\python.exe -m streamlit run app.py
.\dfenv\Scripts\python.exe -m inference.media --image D:\path\face.jpg --output results/inference/image_run
.\dfenv\Scripts\python.exe -m inference.media --video D:\path\clip.mp4 --output results/inference/video_run
```

The application automatically loads the verified selected champion when available. Otherwise it clearly labels the preserved existing image-trained baseline, which has not been validated on DFDC. Both media paths use checkpoint-bound cropping, normalization and class mapping. The video verdict is the arithmetic mean of frame FAKE probabilities, threshold 0.5. Four frames/video is the default transfer protocol; explicitly changing sampling is labeled a protocol change. Representative explanations use frames most supportive of the aggregate class, which is a selection rule, not whole-video coverage.

Scores are **uncalibrated model outputs**, not calibrated probabilities of forensic authenticity. Grad-CAM is available for all CNN variants. The Grad-CAM++ option uses the common first-gradient power approximation, not exact higher-order derivatives or a reproduction of Paper 45's all-layer averaging. Spatial CAMs for the frequency hybrid cover only the CNN branch. Heatmaps show class-score influence and do not diagnose manipulation artifacts.

Optional economical validation attribution checks compare 15%-area salient masking with five equal-area random masks and measure horizontal-flip map correlation:

```powershell
.\dfenv\Scripts\python.exe -m explainability.evaluate_cam --checkpoint results/experiments/A1_resnet18_frozen/best.pt
```

Masked mean-color inputs can be out of distribution. Negative deletion drops and undefined stability are retained. These checks are separate from classifier accuracy.

## Verification

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m tools.smoke_transfer
.\dfenv\Scripts\python.exe -m tools.verify_automatic_gallery --output results/gallery_rerun/app_qa.json
```

Unit tests use synthetic fixtures. The sequential smoke includes frozen and selective-backbone forward/backward operations at batch 2, 224×224, both CAM methods, legacy image/video inference, no-face fallback, and a detector/alignment check on a pre-existing face image. It does not train on or measure DFDC. The legacy `models/best_resnet18.pth` hash is checked before/after. Test logs, software-check JSON and the implementation report record executed evidence and remaining limits.
