# Automatic explanation gallery — 20 October 2026

Implemented in the existing local Streamlit demo. Detection models now have plain names in one selector; checkpoint IDs, hashes, architectures and evaluation records remain in the technical details. The recommended image model remains S0 and the recommended video model remains S6.

Uploading starts all compatible methods automatically. CNNs show six views, transformers five, and combined CNN/transformer models eight. Two isolated model copies run methods concurrently on separate CUDA streams. Completed views fill their gallery cards as they finish, with no method selector, Generate button or REAL/FAKE explanation control. Every class-sensitive method explains the prediction; native attention rollout is explicitly class agnostic. Video methods retain the complete selected decision with the other sampled frame embeddings fixed.

Each plot describes why it is included, what it depicts and how to read its colors. This also covers the video frame score timeline and mean-replacement sensitivity chart. Opacity is a display-only setting; completed maps and predictions are reused. Notes and warnings remain in a collapsed panel at the bottom, including experimental spatial fallbacks. Weak, negligible and numerically inconclusive maps have quality labels beside their views.

## Verification

| Check | Result | Evidence |
|---|---|---|
| Full regression suite | 59 passed | [regression.log](regression.log) |
| Actual Streamlit upload/widget cases | 16 passed | [app_qa_final.json](app_qa_final.json) |
| Generated maps in those UI cases | 63 finite, correctly sized, matched predicted target | [app_qa_final.json](app_qa_final.json) |
| Real checkpoint concurrent/serial comparison | All 14 maps matched exactly in this run; predictions unchanged | [parallel_parity.json](parallel_parity.json) |
| Live browser preview | Six views appeared automatically after upload; plot descriptions and collapsed bottom notes verified | [gallery_live.png](gallery_live.png) |

The UI checks cover supplied synthetic and real images, a second image model, an experimental hybrid image fallback, REAL/FAKE validation clips, additional video frames, hybrid video, transformer video, cache reuse, opacity changes and corrupt input. Cache rerenders and opacity changes made zero new prediction or attribution calls. Method call intervals confirm concurrent worker activity.

The real-checkpoint comparison covers all six S0 image methods and all eight S8 complete-video methods. Maximum absolute concurrent/serial map difference was 0 for every case. S0 FAKE probability stayed 0.8578057885 and S8 video FAKE probability stayed 0.9646026492. Hooks were cleaned up and transformer fused-attention settings restored. The additional LSTM gate regression compares outputs and input gradients; the CUDA regression compares explanation and ordinary inference decisions.

The parallel check's peak allocated CUDA memory was about 1.29 GiB for the combined model, with about 3.31 GiB reserved by PyTorch. These are process allocator measurements, not total device use. Automatic all-method runs can take seconds to tens of seconds on a cold model; parallel scheduling does not guarantee faster total execution on one GPU. The saved timings include loading, warmup or cloning where stated and should not be interpreted as a controlled parallel-versus-serial speed benchmark. Final AppTest opacity rerender took about 185 ms, including UI serialization and excluding browser paint.

## Interpretation for judges

The previous REAL/FAKE selector chose which model output to explain. It could explain the FAKE output of an image predicted REAL; it never changed the prediction. Automatic predicted-class attribution removes that ambiguity.

A highlighted eye, mouth or other region is evidence of model influence, not verified evidence that those pixels were edited. The live Layer-CAM view emphasizes the mouth/lower face in the supplied synthetic image. Other methods can emphasize different regions. A classification model may also use background cues. No heatmap is modified to force anatomical localization or whole-face coverage for REAL. Establishing edit locations requires region/pixel labels and a separately evaluated localization model.

The [demonstration guide](../../docs/automatic_gallery.md) provides presentation wording and the full name-to-checkpoint mapping. No retraining, threshold change or benchmark re-selection was performed for this UI change. Original checkpoints and sealed test outputs are preserved.

## Reproduce

Run from the repository root, choosing new output paths to preserve existing evidence:

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m tools.verify_automatic_gallery --output results/gallery_rerun/app_qa.json
.\dfenv\Scripts\python.exe -m tools.verify_parallel_attribution --output results/gallery_rerun/parallel_parity.json
```

Windows AppTest requires local loopback socket access. The browser file chooser was verified through the visible Upload button; direct hidden-input automation did not expose a chooser event in the in-app browser.

![Verified live automatic gallery card](gallery_live.png)
