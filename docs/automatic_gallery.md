# Automatic XAI demonstration

## Current image checkpoints

The existing ResNet and E02 selector entries now use their validation-selected full-dataset checkpoints. The selector shows full 20,000-image validation accuracy; model details separately expose the common 2,000-image ranking cohort and repeated 20,000-image test result. Both retain all six CNN attribution methods.

Checkpoint hashes are included in prediction and explanation cache keys. Current training provenance and metrics are in [the execution report](../results/current_image_models/REPORT.md). Historical video results remain unchanged.

Upload an image or video, keep the recommended detection model or choose another, and scroll through the explanation gallery. Every compatible method runs automatically. The default video explanation uses the sampled frame whose spatial score most supports the video prediction; the optional frame panel can select up to three frames.

## What REAL and FAKE meant in the old control

The old selector chose the model output to explain. Choosing FAKE asked “which regions support the FAKE output?” even when the prediction was REAL. It did not relabel the input or change the prediction. The gallery now selects the predicted class automatically. Attention rollout is explicitly class agnostic.

## How to present a heatmap

Say: “The model predicted FAKE. This method attributes support for that decision to these regions. We can inspect the eye or mouth if it is highlighted.” For occlusion, explain that the score changed when a patch was hidden. For signed maps, red supports the explained class and blue opposes it. Yellow denotes stronger relative support in CNN maps or stronger attention in the attention views.

Do not describe a classification heatmap as a verified manipulation mask. These checkpoints were trained for image/video classification; they do not provide ground-truth edit boundaries or an eye/mouth artifact diagnosis. Establishing such a claim needs pixel/region annotations and separately measured localization performance. A REAL decision can depend on a small set of cues: there is no requirement that the whole face light up. Colors are normalized per map, so area and brightness are not confidence scores.

The UI names each plot's purpose and contents, states the model branch it covers, and labels numerically inconclusive IG, weak LIME fits and negligible maps. Full warnings stay in the collapsed Notes and warnings panel at the bottom. Agreement among methods can guide inspection; it does not validate an edit boundary.

## Names and checkpoint identities

| Display name | Internal checkpoint |
|---|---|
| ResNet-18 [v, img100k] | S0 |
| ResNet-18 [a, img12k] | E01_resnet18 |
| EfficientNet-B0 [ft, a, img100k] | E02_efficientnet_b0 |
| EfficientNet-B0 + ECA [a, img12k] | E03_attention |
| EfficientNet-B0 [a, j, img12k] | E04_jpeg |
| EfficientNet-B0 [v] | S1 |
| EfficientNet-B0 [f] | S1p |
| EfficientNet-B0 [f, b] | S1b |
| EfficientNet-B0 [f, b, a] | S2 |
| DeiT-Tiny [v] | S3 |
| DeiT-Tiny [f, b, a] | S4 |
| EfficientNet-B0 + DeiT-Tiny [f, b, a] | S5 |
| EfficientNet-B0 + LSTM [f, b, a] | S6 |
| EfficientNet-B0 + LSTM (score fusion) [f, b, a] | S7 |
| EfficientNet-B0 + DeiT-Tiny + LSTM (score fusion) [f, b, a] | S8 |

Tags: `v` = baseline preprocessing/training pipeline; `f` = YuNet face detection and alignment; `b` = inverse-class-frequency balanced sampling; `a` = training augmentation; `j` = JPEG augmentation; `img12k` / `img100k` = number of face images used in image training. Tags come from the saved run configuration and appear in a legend below the selector. `a` denotes the augmentation registered for each run, rather than implying identical augmentation across datasets.

Each selector option shows its validation balanced accuracy for the current input mode, for example `EfficientNet-B0 + LSTM [f, b, a] · 94.82% validation balanced accuracy`. The metric matches the balanced-accuracy ranking. Experimental image transfers without an image benchmark say `validation score not measured` instead of borrowing a video score.

S6 has a temporal LSTM decision head. S7 combines the S2 spatial classifier's probability with the S6 LSTM probability; the selected weights are 25% spatial and 75% temporal. S8 combines the S5 CNN/transformer spatial classifier with S6; its LSTM consumes EfficientNet features only. The selected model caption spells out these branches and weights. S5 instead concatenates encoder features before one spatial classifier.

Architecture, training dataset, evaluation protocol, checkpoint hash and internal ID remain available in Model and evaluation details. Simplified names do not alter models or benchmark numbers. Motion models offered experimentally in image mode still use their explicitly documented spatial fallback.

## Execution and verification

`explainability/suite.py` schedules all compatible methods through two worker threads. Each worker copies the loaded PyTorch module graph, preserving the attribution layer's reference to its copied encoder. The original classifier, other worker and their hooks do not share model tensors. Workers reuse prepared input arrays and video context rather than rerunning face detection/classification. Separate CUDA streams avoid a device-wide timing barrier in each method. Explicit LSTM gates differentiate the same saved weights without changing global cuDNN flags; normal inference keeps its existing path.

The suite cache holds at most four requests for 30 minutes, keyed by preserved input, checkpoint identity/hash, modality, predicted target, frame positions, methods, parameters and UI version. Each worker retains only its latest model. An individual failed method does not discard successful views. The opacity control changes presentation only.

Use a new output namespace when rerunning:

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m tools.verify_automatic_gallery --output results/gallery_rerun/app_qa.json
.\dfenv\Scripts\python.exe -m tools.verify_parallel_attribution --output results/gallery_rerun/parallel_parity.json
```

The first tool checks actual Streamlit uploads, automatic generation, target consistency, cache reuse, extra video frames, CNN/transformer/hybrid compatibility, bottom warnings and corrupt input. The second compares concurrent and serial maps from real checkpoints, checks hook cleanup and verifies unchanged predictions. AppTest needs Windows loopback socket access for its local event loop.

Both verifiers use the existing local held-out image/video fixtures rather than files in a personal Pictures folder. Optional broader experiments are available through `python -m tools.audit_xai_end_to_end --output results/xai_audit_new_run`; `--samples` selects an additional local media directory.

The [24 October audit](xai_end_to_end_audit.md) records actual browser uploads and independent numerical checks. Occlusion now shows its largest absolute patch effect in percentage points beside the relative-color map. Its diagnostics retain separate target-versus-alternative margin changes for saturated probabilities. LIME and occlusion score-space labels identify probabilities; raw attention remains class agnostic. These presentation changes preserve predictions and attribution values.
