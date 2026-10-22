# Automatic XAI demonstration

## Current image checkpoint — 21 October 2026

The recommended image model is **ResNet-18 [ft, a, img100k]**, where `ft` means continued fine-tuning of the original checkpoint. Its selector shows **98.655% full validation accuracy** on 20,000 images. The caption separately reports **98.695% held-out test accuracy** on 20,000 images. Ranking still uses the shared 2,000-image validation cohort; the full validation score is explicitly labeled to distinguish these scopes. Internal ID: `R18_FT_20261021`.

The final checkpoint passed reload, repeated inference, all six CNN explanation methods and actual Streamlit uploads. Epoch selection used validation only. The test set was evaluated once after selection. These are within-dataset image results; heatmaps retain the interpretation limits below. The original image checkpoint remains available, and the video models retain their existing results. See [the local measured report](../results/demo_accuracy_20261021_040741/REPORT.md) for precision, recall, F1, confusion matrices and rollback.

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
| EfficientNet-B0 [a, img12k] | E02_efficientnet_b0 |
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

Each selector option also shows its validation accuracy for the current input mode, for example `EfficientNet-B0 + LSTM [f, b, a] · 90.29% validation accuracy`. This is ordinary accuracy; balanced accuracy remains separately labeled in the metrics caption and determines the ranking. Experimental image transfers without an image benchmark say `accuracy not measured` instead of borrowing a video score.

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
