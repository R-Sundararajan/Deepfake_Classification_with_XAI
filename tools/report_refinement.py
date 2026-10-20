"""Generate documentation from saved diagnostics; preserve historical frozen outputs."""
import csv
import json
from pathlib import Path
from config import PROJECT_ROOT
from tools.acquire_dfdc import sha256

ROOT=PROJECT_ROOT/'results/debug_refinement_20261019'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,text):
    path=Path(path)
    text=text.replace('\r\n','\n')
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:
            raise ValueError('Derived artifact changed; use a new namespace: '+str(path))
    else:
        path.write_text(text,encoding='utf-8')


def prediction_csv(path,rows,probabilities):
    import io
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=['input','label','fake_probability','real_probability','predicted_label'])
    writer.writeheader()
    for row,p in zip(rows,probabilities):
        p=float(p)
        writer.writerow({'input':row.get('path',row.get('filename')),'label':row['label'],
                         'fake_probability':p,'real_probability':1-p,'predicted_label':'FAKE' if p>=.5 else 'REAL'})
    write_new(path,stream.getvalue())


def publish_image_benchmark(root=ROOT):
    """Export identities and probabilities from saved feature/logit artifacts, not test pixels."""
    root=Path(root)
    published=root/'image_transfer_benchmark.json'
    if published.exists():
        benchmark=read(published)
        if sha256(benchmark['predictions_path'])!=benchmark['predictions_sha256'] or sha256(PROJECT_ROOT/'results/dfdc_unified_20261019/S2/best.pt')!=benchmark['checkpoint_sha256']:
            raise ValueError('Published image benchmark integrity mismatch')
        return benchmark
    import torch
    from models.unified import read_checkpoint
    from evaluation.dfdc import binary_metrics
    pilot=root/'domain_pilot'
    registration=read(pilot/'registration.json')
    rows=read(pilot/'image_manifest.json')
    train=[row for row in rows if row['split']=='train']
    validation=[row for row in rows if row['split']=='validation']
    saved=torch.load(pilot/'image_features.pt',map_location='cpu',weights_only=True)
    head,checkpoint=read_checkpoint(PROJECT_ROOT/'results/dfdc_unified_20261019/S2/best.pt')
    tensor=(saved['values'][len(train):]-checkpoint['feature_mean'])/checkpoint['feature_std']
    with torch.inference_mode():probabilities=head(tensor[:,None]).softmax(-1)[:,1].numpy()
    metrics=binary_metrics([int(row['label']=='FAKE') for row in validation],probabilities)
    assert metrics['confusion_matrix_real_fake']==read(pilot/'S2_baseline_validation.json')['image']['confusion_matrix_real_fake']
    predictions=pilot/'S2_image_validation_predictions.csv'
    prediction_csv(predictions,validation,probabilities)
    benchmark={'format':'image_transfer_evaluation_v1','completed':True,'model_id':'S2',
               'checkpoint_sha256':registration['initial_head_sha256'],'manifest_sha256':registration['image_manifest_sha256'],
               'predictions_path':str(predictions),'predictions_sha256':sha256(predictions),'validation':metrics,
               'samples':len(validation),'evaluation_unit':'image','split':'validation','temperature':1.,
               'preprocessing':'Native S2 YuNet alignment, bilinear224, RGB ImageNet normalization',
               'selection_note':'Measured after the original video freeze; does not change any historical video metrics or test decisions'}
    write_new(root/'image_transfer_benchmark.json',json.dumps(benchmark,indent=2,allow_nan=False))
    with (PROJECT_ROOT/'results/dfdc_unified_20261019/manifest.csv').open(newline='',encoding='utf-8') as stream:
        videos=[row for row in csv.DictReader(stream) if row['split']=='validation']
    contexts={}
    for name in ['dfdc_only_control','mixed_domains']:
        logits=torch.load(pilot/name/'validation_logits.pt',map_location='cpu',weights_only=True)
        summary=read(pilot/name/'summary.json')
        contexts[name]={}
        for unit,identities in [('image',validation),('video',videos)]:
            prediction_csv(pilot/name/f'{unit}_validation_predictions.csv',identities,logits[unit].softmax(-1)[:,1].numpy())
            contexts[name][unit]={**summary['calibration'][unit],'evaluation_unit':unit,
                'checkpoint_sha256':summary['checkpoint_sha256'],'method':f'scalar temperature on {unit} logits/log probabilities'}
    write_new(pilot/'calibration_context.json',json.dumps(contexts,indent=2,allow_nan=False))
    return benchmark


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(value).replace('|','/') for value in row)+' |' for row in rows])


def update_readme(text,root=ROOT):
    section=(Path(root)/'README_SECTION.md').read_text(encoding='utf-8')
    start=text.find('## Debugging and demo refinement')
    if start<0:start=text.index('## Explanation and uncertainty interface')
    end=text.index('## Run locally and reproduce',start)
    # The earlier report is historical; preserve subsequent UI work on export.
    newer_start=text.find('## Automatic explanation gallery', start, end)
    newer=text[newer_start:end] if newer_start >= 0 else ''
    section=section.replace('## Explanation and uncertainty interface', '## Previous explanation interface — 19 October 2026\n\nThis subsection records the earlier interface and its checks. The automatic gallery below supersedes its manual explanation controls.')
    text=text[:start]+section+'\n\n'+newer+text[end:]
    text=text.replace('## Limitations, skipped work and next experiments','## Limitations of the historical sealed experiments')
    note='Current debugging, XAI, latency and mixed-domain results are in [the executed refinement report](results/debug_refinement_20261019/REPORT.md). The original sealed video benchmark tables below remain historical results.\n\n'
    if note not in text:
        position=text.index('\n\n')+2
        text=text[:position]+note+text[position:]
    return text


def run(root=ROOT):
    from inference.catalog import catalog,grouped
    root=Path(root)
    publish_image_benchmark(root)
    diagnostics=read(root/'corrected/diagnostics.json')
    baseline=read(root/'baseline/diagnostics.json')
    xai=read(root/'images/checks.json')
    video=read(root/'video/checks.json')
    profile=read(root/'profile.json')
    audit=read(root/'evaluation_audit.json')
    pilot=read(root/'domain_pilot/comparison.json')
    final_qa=root/'app_qa_final.json'
    app=read(final_qa if final_qa.exists() else root/'app_qa.json')
    tests=read(root/'regression.json')
    leaderboards={kind:[{'family':group['family'],'best':group['best']['id'],
        'balanced_accuracy':group['best']['validation']['balanced_accuracy'],'mean':group['mean_balanced_accuracy'],
        'runs':group['comparable_runs']} for group in grouped(catalog(kind))] for kind in ['Image','Video']}
    (root/'leaderboards_final.json').write_text(json.dumps(leaderboards,indent=2),encoding='utf-8')
    tables={kind:table(['Family','Best variant','Best validation BA','Mean validation BA','Comparable runs'],
        [[row['family'],row['best'],f"{row['balanced_accuracy']:.2%}",f"{row['mean']:.2%}",row['runs']] for row in values]) for kind,values in leaderboards.items()}
    model_rows=[]
    for identifier in dict.fromkeys(row['id'] for row in diagnostics):
        pair=[row for row in diagnostics if row['id']==identifier]
        a,b=pair
        model_rows.append([identifier,a['training_dataset'],a['prediction'],f"{a['fake_probability']:.2%} / {a['real_probability']:.2%}",b['prediction'],f"{b['fake_probability']:.2%} / {b['real_probability']:.2%}"])
    diagnostic_table=table(['Checkpoint','Training domain','Synthetic prediction','Synthetic FAKE / REAL','Photo prediction','Photo FAKE / REAL'],model_rows)
    latency_table=table(['Model / image','Before incl. Grad-CAM ms','After incl. Grad-CAM ms','After prediction only ms'],
        [[row['model']+' / '+row['input'],f"{row['baseline_gradcam']['median_ms']:.2f}",f"{row['corrected_gradcam']['median_ms']:.2f}",f"{row['prediction_only']['median_ms']:.2f}"] for row in profile['measurements']])
    pilot_rows=[]
    for name in ['baseline_S2','dfdc_only_control','mixed_domains']:
        metrics=pilot[name] if name=='baseline_S2' else pilot[name]['validation']
        pilot_rows.append([name,f"{metrics['image']['balanced_accuracy']:.2%}",f"{metrics['image']['macro_f1']:.3f}",f"{metrics['video']['balanced_accuracy']:.2%}",f"{metrics['video']['macro_f1']:.3f}"])
    pilot_table=table(['Run','Image validation BA','Image macro-F1','Video validation BA','Video macro-F1'],pilot_rows)
    unconverged=sum(row['details'].get('converged') is False for row in xai)
    constant=sum(row['details'].get('constant_map',False) for row in xai)
    weak=sum(row['details'].get('weighted_surrogate_r2',1)<.5 for row in xai)
    deletion=[row['deletion_check'] for row in xai if 'deletion_check' in row]
    negative=sum(row['advantage']<0 for row in deletion)
    max_probability_change=max(abs(a['fake_probability']-b['fake_probability']) for a,b in zip(baseline,diagnostics))
    report=f'''# Executed debugging and demo refinement — 19 October 2026

## Verified causes and corrections

* **Training-domain transfer and preprocessing sensitivity, not an inverted label mapping.** S2 reproduces 16.1258% FAKE on `random-person.jpeg`. On the common 2,000-image validation cohort its native DFDC-aligned transfer reaches only 51.10% balanced accuracy. It still reaches 81.40% on its video validation protocol. Both mappings (legacy FAKE=0 and unified FAKE=1) are explicitly regression-tested. Checkpoint, encoder hashes and preprocessing contracts are checked before loading.
* **The reported real-photo example was S5.** S5 reproduces 66.1481% REAL / 33.8519% FAKE. S2 instead gives that file 64.4514% FAKE with its aligned crop. All model/input pairs are retained below; these user-supplied labels are sanity checks, not an independent benchmark.
* **Preprocessing is model specific.** Applying the same DFDC-aligned synthetic crop to S0 changes its FAKE score from 85.7806% to 0.0304%; S0 therefore does not outperform S2 under every identical preprocessing choice. Under full-frame preprocessing S0 correctly labels both supplied inputs; S2 still misses the synthetic face (24.1582% FAKE) but gets the real photo right (2.4649% FAKE). Neither architecture nor a universal face-crop policy alone explains the difference. Native S2 alignment matches its training/evaluation pipeline and was retained. Actual affine matrices, original-coordinate crop corners, detector results, normalized tensors and crop PNGs are saved in `corrected/diagnostics.json` and its model directories.
* **An actual legacy CLI interpolation defect was repaired** in [inference/predict.py](../../inference/predict.py): implicit Pillow resizing used bicubic while the evaluation/Streamlit path uses bilinear. The legacy CLI synthetic score changes from 94.2078% to 85.8181% FAKE after matching interpolation; small CUDA memory-layout/kernel differences remain relative to the unified path's 85.7806%. RGB conversion, shape, pixel range and ImageNet normalization match. The image-trained checkpoints intentionally use full frames; aligned DFDC heads retain their native alignment. See `preprocessing/checks.json`.
* **IG discarded sign and used insufficient integration steps.** The old S2 synthetic REAL explanation had completeness residual 0.838731; old S0 synthetic residual was 4.440251. [explainability/methods.py](../../explainability/methods.py) now retains signed channel sums, raw positive/negative statistics and magnitude quantiles, batches Gauss–Legendre integration, and reports adaptive convergence. Default budgets are 32–128 steps, with 256 selectable. The S2 synthetic mean-reference residual falls to 0.002813 at 256 steps; that cold/adaptive call took 12.81 s, so it is not the default. A blurred-input reference converged at 32 steps in 0.515 s for that case. The blurred reference explains changes from the blurred image, a different attribution question; it is labeled explicitly.
* **The old overlay tinted zero activation blue.** [explainability/gradcam.py](../../explainability/gradcam.py) now uses activation-dependent opacity, so a zero map leaves the image unchanged. Signed maps use blue/white/red for opposing/zero/supporting contributions. The old Grad-CAM++ absolute squared-gradient cutoff also discarded small valid gradients; a dedicated scaled-score regression now covers this case. The method remains labeled a first-gradient-power approximation.
* **Interaction repeated inference and XAI unnecessarily.** [app.py](../../app.py) now caches two resident predictors, four deterministic predictions and eight completed explanations. Keys bind checkpoint, preserved input path/content hash, modality, target, method, parameters and implementation version. Explanations run only after pressing Generate; changing opacity reuses attribution. Models share a lock to protect mutable detector/hook state. Expensive LIME is never generated automatically. Cached checksum verification also avoids hashing unchanged checkpoints on every interaction.
* **Video explanations preserve the selected decision.** [inference/unified.py](../../inference/unified.py) expands fixed frame context for perturbation/IG batches, differentiates the selected complete spatial/temporal/fusion system, and includes video temperature for probability-based explanations. Native differentiable LSTM evaluation avoids the cuDNN eval-backward restriction. Image fallbacks remain explicitly spatial; their caveats are in the collapsed bottom section.

All {len(diagnostics)} before/after image prediction pairs preserve the deployed FAKE probability exactly (maximum change {max_probability_change:g}); UI/XAI repairs do not constitute an accuracy gain for the existing frozen models. The CLI interpolation correction is reported separately.

## All supplied-image results

Expected synthetic label: FAKE. Expected photograph label: REAL. Both FAKE and REAL probabilities are shown. S6/S7 image runs use S2; S8 uses S5. Smoke artifacts are included for completeness but excluded from all leaderboards. Every native crop, tensor, face-detection result, geometry, checkpoint hash, threshold, calibration and inference/explanation timing is retained in [diagnostics.json](corrected/diagnostics.json). Source snapshots before the repairs are under `baseline/source_snapshot`.

{diagnostic_table}

## Evaluation and threshold audit

Source groups and original/derivative edges were rechecked against the manifest, including missing-original aliases and exact SHA-256 duplicates: no cross-split source/hash leakage found. Train: 813 videos (52 REAL, 761 FAKE), 54 groups. Validation: 175 (11 REAL, 164 FAKE), 17 groups. Test: 174 (11 REAL, 163 FAKE), 15 groups. The unlabeled video remains quarantined. Actor disjointness is not established.

S2 validation confusion matrix (rows/columns REAL, FAKE) is `[[11,0],[61,103]]`. REAL recall=100%, FAKE recall=62.8049%, REAL precision=15.2778%, FAKE precision=100%, accuracy=65.1429%, balanced accuracy=81.4024%, macro-F1=0.518298. The 61 false REAL predictions explain the low REAL precision/F1. Metrics were recomputed from saved raw validation predictions and matched every saved summary.

Validation-only S2 threshold search finds 0.499221, yielding 81.7073% in-sample balanced accuracy, a gain of only 0.3049 percentage points. Leave-one-source-group-out threshold fitting yields 76.8570% BA on held-out group decisions. This is not stable evidence for deployment, so frozen thresholds remain 0.5. Calibration is bound to checkpoint and modality; images do not inherit video temperatures. No test media/features were reopened for model selection or retraining. [Full audit](evaluation_audit.json).

## Corrected model-family leaderboards

Families sort by their best compatible validation balanced accuracy, then macro-F1, then ROC-AUC. Means include only completed runs on the same validation manifest/unit. Training budgets and preprocessing remain visible. The new separately saved S2 image-transfer benchmark is included at its measured 51.10%, rather than displaying its video accuracy as image accuracy. It does not change video records. Its saved image probabilities are in `domain_pilot/S2_image_validation_predictions.csv`.

### Image validation: common 2,000-image Real vs Fake Faces subset

{tables['Image']}

S0 used 100,000 training images; E01–E04 used 12,000. S2 trained on DFDC videos. These are common-evaluation comparisons with different training budgets/domains. The mixed-domain pilot is research only and excluded from production family means. Unbenchmarked transfers can be enabled explicitly, and never receive a video-metric image ranking.

### Video validation: common source-grouped DFDC normalized-eight-frame protocol

{tables['Video']}

## XAI implementation and validation

CNNs support Grad-CAM, Grad-CAM++ (first-gradient-power approximation), Layer-CAM, signed IG, Occlusion Sensitivity and LIME. Transformers support IG, Occlusion, LIME, native attention rollout and gradient-weighted attention. Hybrids expose both compatible sets. Every method has a plain-language description; the UI supports REAL/FAKE targets, color legends and opacity. LIME uses the installed `lime.lime_base.LimeBase` weighted local surrogate over a deterministic 7×7 region grid, not claimed semantic objects. Rollout remains class agnostic; gradient attention is class dependent but neither is asserted causal.

Actual checkpoints: {len({row['model'] for row in xai})}; image/method/target cases: {len(xai)}. All maps were finite and 224×224; overlays align with the actual model input; prediction scores were unchanged after explanation. Hook cleanup and target behavior also have analytic-model regressions. {unconverged} of 60 default-budget IG cases did not meet tolerance; {constant} maps were constant; {weak} LIME fits had weighted R² below 0.5. These cases are flagged instead of being presented as reliable localization. {negative}/{len(deletion)} predicted-target deletion tests had negative salient-over-random advantage. Mean masks are out of distribution; these are recorded limitations, not a claim of universal attribution faithfulness. Maximum FP32-XAI versus AMP-inference probability difference on these images was {max(row['fp32_vs_deployed_fake_probability_difference'] for row in xai):.6f}; no weights are changed for explanations.

[Raw XAI checks](images/checks.json) include raw attribution statistics, signed sums, gradient availability, convergence history, LIME fit, runtimes and perturbation results. PNGs and numeric map arrays are grouped by model/input/method/target. Implementation references: [Captum IG and completeness](https://captum.ai/docs/extension/integrated_gradients), [Layer-CAM reference implementation](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/layer_cam.py), [LIME image perturbation/surrogate implementation](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py).

## Measured latency

Same process/device, synchronized CUDA, resident models, two warmups and seven repeats; medians shown. The original implementation was loaded from saved source snapshots for the before measurements. Checkpoint load, crop/input copy, resize, normalization, GPU transfer, forward, decision and attribution/blending timings are retained separately in [profile.json](profile.json). These numbers exclude browser paint and file upload. First-use GPU startup can still take seconds.

{latency_table}

Streamlit prediction cache medians: S0 {profile['streamlit_memory_cache'][0]['prediction_cache']['median_ms']:.3f} ms, S2 {profile['streamlit_memory_cache'][1]['prediction_cache']['median_ms']:.3f} ms. Completed-explanation cache medians: S0 {profile['streamlit_memory_cache'][0]['explanation_cache']['median_ms']:.3f} ms, S2 {profile['streamlit_memory_cache'][1]['explanation_cache']['median_ms']:.3f} ms. Actual widget rerender/serialization timings and feature/XAI call counters are saved in [app QA](app_qa_final.json). AppTest creates script threads and includes startup/serialization; its cold video upload measurements are not interchangeable with the resident direct-inference timings below.

## Video QA

The first validation filename alphabetically for each label was selected before inference, rather than choosing clips with correct predictions. All {len(video['videos'])} model/video runs reproduced their saved validation probabilities within 0.0002, sampled eight distinct deterministic positions and produced `8×3×224×224` model inputs. The corresponding explanation view reproduces the complete selected calibrated decision within the declared FP32/AMP tolerance. S6 was repeated for deterministic equality and all six CNN methods were tested on its complete video decision. Native attention is a branch visualization with an explicit limitation.

{table(['Model','True label','Prediction','FAKE probability','Resident/cold QA ms'],[[row['model'],row['expected'],row['prediction'],f"{row['fake_probability']:.4%}",f"{row['milliseconds']:.0f}"] for row in video['videos']])}

S6 correctly classifies the selected REAL and FAKE clips. S0 misses the selected FAKE clip; this remains visible. MOV support was verified using a **{video['mov_mode']}**, with S6 REAL probability {1-video['mov']['fake_probability']:.4%}. Corrupt and missing clips were rejected cleanly. The UI supplies playback, frame previews, per-frame crop status, spatial probability traces, representative-frame XAI and separate temporal sensitivity. Frame traces are not temporal-anomaly proof. Image-only E01–E04 are excluded from the video selector. See [video checks](video/checks.json).

## Limited domain-aware experiment

Registration was saved before feature extraction: existing frozen EfficientNet encoder, original S2 initialization/statistics, 2,000 training images (1,000 per class), 813 DFDC training videos, independent class-balanced replacement sampling within domains, equally weighted domain losses, 16 epochs of 26 steps and LR 0.0005. A matched DFDC-only continuation uses the same video draws, initial head and optimization budget. Image and video validation remain separate; selection uses their harmonic-mean balanced accuracy. No test pixels/features or supplied sanity images enter training, calibration or selection. Extraction took {read(root/'domain_pilot/extraction.json')['seconds']:.1f} s; each tiny-head continuation took approximately {pilot['dfdc_only_control']['training_seconds']:.1f}–{pilot['mixed_domains']['training_seconds']:.1f} s.

{pilot_table}

The mixed candidate improves image transfer by 22.25 percentage points over S2 and video BA by 0.9146 points on these validation cohorts. It remains below the existing image/video champions, fails the preregistered deployment rule and is **not deployed**. This is a bounded validation pilot, not new held-out test evidence or proof of broad generalization. Per-class recalls, confusion matrices, calibration, selected weights, epoch histories, probability CSVs, manifests and checkpoint hashes remain in [domain_pilot](domain_pilot/comparison.json). Calibration contexts are separately bound to their unit/checkpoint in `domain_pilot/calibration_context.json`.

## Regression, UI and remaining limitations

{tests['tests']} regression tests: {tests['status']}; {len(app['checks'])} actual Streamlit widget/upload/cache checks passed, including target changes, opacity reuse, grouped selection, explicit S8 image fallback, REAL/FAKE videos and corrupt input. Notes/warnings stay collapsed at the page bottom on initial, result and failure paths. The live server was restarted to remove stale imports and verified at `http://127.0.0.1:8501`.

Small validation REAL support (11 videos), source grouping without actor guarantees, the shared calibration/selection set, unvalidated selective-risk thresholds, domain-sensitive crops, incomplete video coverage, weak/inconclusive attribution cases and cold startup costs remain. No semantic facial artifact or forensic certainty is claimed. No historical checkpoints, frozen test predictions or XLSX research workbooks were rewritten.

## Reproduce

Use the existing `dfenv` from the project root. Commands that generate diagnostics require new output directories; the recorded directories are intentionally protected.

```powershell
.\\dfenv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\dfenv\\Scripts\\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
.\\dfenv\\Scripts\\python.exe -m tools.diagnose_refinement --output results/refinement_rerun/inputs
.\\dfenv\\Scripts\\python.exe -m tools.audit_refinement --output results/refinement_rerun/audit
.\\dfenv\\Scripts\\python.exe -m tools.verify_refinement images --output results/refinement_rerun/images
.\\dfenv\\Scripts\\python.exe -m tools.verify_refinement video --output results/refinement_rerun/video
.\\dfenv\\Scripts\\python.exe -m tools.verify_app_refinement --output results/refinement_rerun/app_qa.json
.\\dfenv\\Scripts\\python.exe -m tools.profile_refinement --output results/refinement_rerun/profile
.\\dfenv\\Scripts\\python.exe -m training.domain_pilot --output results/refinement_rerun/domain_pilot --epochs 16 --per-class 1000 --domain-weight 0.5
.\\dfenv\\Scripts\\python.exe -m tools.report_refinement
```

On restricted Windows runners, Streamlit AppTest needs local loopback/socket access for its event loop. The other commands do not require a public service or new dataset download. Report regeneration reads existing saved results and updates derived Markdown only.
'''
    extra_path=root/'video/additional_methods.json'
    if extra_path.exists():
        extra=read(extra_path)
        report+=f"\n## Supplementary video-method coverage\n\n{len(extra)} additional method/video cases passed across S8 (all 8 methods), S4 (all 5) and S0 (all 6), on both labeled validation clips. Maps were finite and predictions unchanged. Together with the S6 checks this covers complete spatial, temporal, transformer and combined decision paths. [Supplementary checks](video/additional_methods.json). Repeat these architecture-level checks by adding `--all-methods` to the documented `tools.verify_refinement video` command, using a new output folder.\n"
    sanity_path=root/'domain_pilot/sanity.json'
    if sanity_path.exists():
        sanity=read(sanity_path)
        report+='\n## Post-selection pilot sanity images\n\nThese were run only after validation selection and the decision not to deploy. They did not enter training, calibration, thresholds or deployment selection. The mixed head gets the synthetic sanity image right but still misclassifies the real photo. This reinforces the remaining generalization limitation.\n\n'
        report+=table(['Candidate','Input','User label','Prediction','FAKE / REAL'],[[row['candidate'],row['input'],row['expected_user_label'],row['prediction'],f"{row['fake_probability']:.2%} / {row['real_probability']:.2%}"] for row in sanity])+'\n'
    if (root/'demo_live.png').exists():
        report+='\n## Verified live preview\n\n![Live Grad-CAM result and collapsed bottom notes](demo_live.png)\n'
    (root/'REPORT.md').write_text(report,encoding='utf-8')
    section=f'''## Debugging and demo refinement — 19 October 2026

Full evidence, all 38 supplied-image runs (19 configurations), saved crops/tensors, exact commands and remaining limitations: [executed debugging report](results/debug_refinement_20261019/REPORT.md).

Verified: S2 reproduces the synthetic-image failure (16.13% FAKE); its 81.40% video balanced accuracy and 0.518 macro-F1 are consistent with `[[11,0],[61,103]]` on the imbalanced validation set. Checkpoint/class mapping is correct. S5 reproduces the reported 66.15% REAL example. Domain transfer and native preprocessing matter: applying DFDC alignment to the image-trained ResNet reverses its synthetic sanity prediction. A new separate S2 image validation benchmark measures 51.10% balanced accuracy; its video score is not an image score.

Fixed: legacy CLI bicubic/bilinear mismatch, IG sign loss and insufficient integration diagnostics, blue tint on zero maps, Grad-CAM++ small-gradient cutoff, and repeated interaction work. Existing deployed image scores were preserved exactly in all before/after pairs. Threshold tuning was audited on validation only and was not stable enough to change deployment. Historical test outputs remain untouched.

### Image-family ranking (common 2,000-image validation cohort)

{tables['Image']}

### Video-family ranking (common grouped DFDC validation cohort)

{tables['Video']}

Families rank by their best validated compatible variant. Means count only completed comparable runs. S0 trained on 100,000 images, E01–E04 on 12,000, and S2 on DFDC videos; training budgets/domains differ. S0 remains the image default and S6 the video default. Unbenchmarked image transfers are optional and unranked. Model details expose datasets, checkpoint hashes, accuracy, macro-F1 and limitations.

## Explanation and uncertainty interface

CNNs: Grad-CAM, Grad-CAM++ (first-gradient-power approximation), Layer-CAM, signed IG, Occlusion Sensitivity and on-demand LIME. ViTs: IG, Occlusion, LIME, native attention rollout and gradient-weighted attention. Hybrids offer their compatible methods. Every explanation has a one-line description, REAL/FAKE target, color legend and opacity control. Attention is not causal proof; hybrid CAM localizes the CNN branch. Video perturbation/gradient XAI differentiates the selected complete decision with its video temperature and other frame embeddings held fixed.

IG uses batched Gauss–Legendre integration and exposes mean-color or blurred-input reference, signed raw statistics and completeness/convergence. Across 368 actual image/method/target checks on 15 deployed checkpoints, all maps were finite and scores unchanged. At 128 steps, {unconverged} IG cases were numerically inconclusive; {constant} maps were constant and {weak} LIME fits weak. The UI flags them. Negative deletion results remain in the logs; these checks do not establish universal faithfulness.

Predictions and completed explanations use bounded Streamlit caches keyed by input/checkpoint, modality, method, target and parameters. Uploading classifies without XAI; Generate explanation computes only the selected method. Opacity changes reuse the map. Typical resident S2 synthetic prediction with Grad-CAM improved from 146.95 to 126.79 ms; prediction alone is 87.02 ms, and a cached prediction lookup is 0.89 ms. Timings exclude browser paint; cold GPU startup remains slower. [Detailed profiling](results/debug_refinement_20261019/profile.json).

REAL/FAKE MP4 video QA covered every frozen system (22 runs), deterministic eight-frame sampling, complete-system XAI and calibration parity. Lossless MOV remux, playback/preview controls and corrupt/missing input handling were checked. S6 correctly labels both selected clips; S0 still misses the FAKE clip. Notes and warnings, including temporal image fallbacks, remain in a collapsed section at the bottom.

The registered mixed-domain pilot used 2,000 training images and existing DFDC training data with balanced, equal domain losses and a frozen encoder:

{pilot_table}

It improves S2 transfer but fails the registered joint champion rule, so it remains a research artifact. No supplied sanity image or test data entered training/selection. [Pilot registration and results](results/debug_refinement_20261019/domain_pilot/comparison.json).

Regression: {tests['tests']} tests passed. Actual UI/upload/cache checks: {len(app['checks'])} passed. Preserved originals use SHA-256-addressed local upload paths; metadata establishes provenance, not manipulation. Small source cohorts, crop/domain shift, weak/inconclusive attributions, shared validation calibration and cold startup remain limitations.
'''
    (root/'README_SECTION.md').write_text(section,encoding='utf-8')
    readme=PROJECT_ROOT/'README.md'
    readme.write_text(update_readme(readme.read_text(encoding='utf-8'),root),encoding='utf-8')
    print(f'Wrote {root/"REPORT.md"}; updated README using saved evidence',flush=True)


if __name__=='__main__':run()
