"""Regenerate the main README and CSV/plots from observed immutable run logs."""
import csv
import json
import sys
import importlib.metadata
from collections import Counter
from pathlib import Path
import numpy as np
from tools.dfdc_storage import ROOT,atomic_json,check_budget
from training.video_protocol import RUN,load_rows,write_csv


def percent(value):return '—' if value is None else f'{100*value:.2f}%'


def build():
    audit=json.loads((RUN/'audit.json').read_text(encoding='utf-8'));freeze=json.loads((RUN/'freeze.json').read_text(encoding='utf-8'))
    summaries=[json.loads((RUN/id/'summary.json').read_text(encoding='utf-8')) for id in freeze['experiments']]
    tests={s['id']:json.loads((RUN/s['id']/'test.json').read_text(encoding='utf-8')) for s in summaries}
    timing_path=RUN/'runtime_benchmark.json'
    timings={r['id']:r for r in json.loads(timing_path.read_text(encoding='utf-8'))['records']} if timing_path.exists() else {}
    rows=load_rows();folder=RUN/'figures';folder.mkdir(exist_ok=True)
    table=[];full=[];testtable=[]
    for s in summaries:
        c=s['config'];v=s['validation'];t=tests[s['id']]['calibrated']
        attention='native self-attention' if any('deit' in name for name in c.get('backbones',[])) else 'native SE' if s['id']!='S0' else 'none'
        temporal='LSTM + late fusion' if s['id'] in {'S7','S8'} else 'one-layer LSTM' if c['kind']=='temporal' else 'none'
        balance='video-balanced' if c.get('balance') else 'natural'
        balance+=' + augmentation' if c.get('augment') else ''
        runtime=timings.get(s['id'],{});latency=runtime.get('forward_ms_per_video',s['inference_ms_per_video'])
        table.append(f"| {s['id']} | {c['model']} | {c['preprocess']} | {balance} | {attention} | {temporal} | {percent(v['balanced_accuracy'])} | {percent(v['accuracy'])} | {v['macro_f1']:.4f} | {v['roc_auc']:.4f} | {s['training_seconds']:.1f} s | {latency:.2f} ms |")
        testtable.append(f"| {s['id']} | {percent(t['balanced_accuracy'])} | {percent(t['accuracy'])} | {percent(t['real_recall'])} | {percent(t['fake_recall'])} | {t['precision']:.4f} | {t['macro_f1']:.4f} | {t['roc_auc']:.4f} | {t['pr_auc']:.4f} | {t['log_loss']:.4f} | {t['brier_score']:.4f} | {t['ece']:.4f} |")
        full.append({'id':s['id'],'model':c['model'],'preprocessing':c['preprocess'],'balance':balance,'attention':attention,'temporal':temporal,
                     **{'validation_'+k:v[k] for k in ['accuracy','balanced_accuracy','real_recall','fake_recall','precision','macro_f1','roc_auc','pr_auc','log_loss','brier_score']},
                     **{'test_'+k:t[k] for k in ['accuracy','balanced_accuracy','real_recall','fake_recall','precision','macro_f1','roc_auc','pr_auc','log_loss','brier_score','ece']},
                     'training_seconds':s['training_seconds'],'inference_ms_per_video':latency,'peak_head_vram_bytes':s['peak_vram_bytes'],
                     'peak_inference_vram_bytes':runtime.get('peak_vram_bytes'),'model_artifact_bytes':runtime.get('model_artifact_bytes'),
                     'checkpoint_bytes':Path(s['checkpoint']).stat().st_size,'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256']})
    reference=json.loads((RUN/'all_fake_reference.json').read_text(encoding='utf-8'))
    vr=reference['validation'];tr=reference['test']
    table.append(f"| Ref | All-FAKE constant rule | — | — | — | — | {percent(vr['balanced_accuracy'])} | {percent(vr['accuracy'])} | {vr['macro_f1']:.4f} | {vr['roc_auc']:.4f} | no training | not benchmarked |")
    testtable.append(f"| Ref | {percent(tr['balanced_accuracy'])} | {percent(tr['accuracy'])} | {percent(tr['real_recall'])} | {percent(tr['fake_recall'])} | {tr['precision']:.4f} | {tr['macro_f1']:.4f} | {tr['roc_auc']:.4f} | {tr['pr_auc']:.4f} | {tr['log_loss']:.4f} | {tr['brier_score']:.4f} | {1-tr['accuracy']:.4f} |")
    write_csv(RUN/'comparison.csv',full)
    atomic_json(RUN/'comparison.json',{'audit':audit,'freeze':freeze,'validation':summaries,'test':tests,'common_runtime_benchmark':timings,'storage':check_budget()})
    make_figures(summaries,tests,freeze['champion'])
    duration=[]
    for label,values in audit['duration_by_class'].items():
        duration.append('| '+label+' | '+' | '.join(str(values[key]) if key=='count' else f"{values[key]:.3f}" for key in ['count','total_seconds','minimum','p10','p25','median','mean','p75','p90','maximum'])+' |')
    splits=[]
    for name,values in audit['split_counts'].items():splits.append(f"| {name} | {values['videos']} | {values['labels'].get('REAL',0)} | {values['labels'].get('FAKE',0)} | {values['source_groups']} |")
    champion=next(s for s in summaries if s['id']==freeze['champion']);ct=tests[champion['id']]['calibrated']
    observed={s['id']:s['validation'] for s in summaries}
    properties={label:{'fps_counts':dict(Counter(r['fps'] for r in rows if r['label']==label)),
        'resolution_counts':dict(Counter(r['width']+'×'+r['height'] for r in rows if r['label']==label)),
        'frames':{key:float(function([int(r['frame_count']) for r in rows if r['label']==label])) for key,function in [('minimum',np.min),('median',np.median),('mean',np.mean),('maximum',np.max)]}} for label in ['REAL','FAKE']}
    atomic_json(RUN/'media_properties.json',properties)
    resolution_table='\n'.join(f"| {label} | {properties[label]['resolution_counts'].get('1920×1080',0)} | {properties[label]['resolution_counts'].get('1080×1920',0)} |" for label in ['REAL','FAKE'])
    extract_logs=list(RUN.glob('features_*.json'));extraction_seconds=sum(json.loads(p.read_text(encoding='utf-8'))['forward_seconds'] for p in extract_logs)
    extraction_vram=max(json.loads(p.read_text(encoding='utf-8'))['peak_vram_bytes'] for p in extract_logs)
    cache=json.loads((RUN/'cache.json').read_text(encoding='utf-8'))
    cache_integrity=json.loads((RUN/'cache_integrity.json').read_text(encoding='utf-8'))
    fallback_table='\n'.join(f"| {name} | {counts.get('yunet_landmark_aligned',0)} | {counts.get('no_face_full_frame_fallback',0)} | {percent(counts.get('no_face_full_frame_fallback',0)/sum(counts.values()))} |" for name,counts in sorted(cache_integrity['face_methods_by_split_class'].items()))
    longer=audit['duration_by_class']['REAL']['mean']>audit['duration_by_class']['FAKE']['mean']
    duration_description='Both class means are effectively 10.01 seconds; the microsecond-scale difference does not support a meaningful REAL-versus-FAKE duration shortcut in this cohort.' if abs(audit['duration_by_class']['REAL']['mean']-audit['duration_by_class']['FAKE']['mean'])<.01 else f"REAL videos have a {'higher' if longer else 'lower or equal'} mean duration in this cohort."
    image=json.loads((ROOT/'results/execution_20261019/comparison.json').read_text(encoding='utf-8'))
    # All image values come from existing logs; no retraining or test repetition.
    image_rows=[]
    for id in ['E00','E01_resnet18','E02_efficientnet_b0','E03_attention','E04_jpeg']:
        path=ROOT/'results/execution_20261019'/('E00_validation/summary.json' if id=='E00' else id+'/summary.json')
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'));image_rows.append({'id':id,'log':str(path.relative_to(ROOT)),'metrics':data})
    atomic_json(RUN/'image_results_references.json',image_rows)
    old_test=json.loads((ROOT/'results/execution_20261019/E00_test/metrics.json').read_text(encoding='utf-8'))
    old_values={r['id']:r['metrics']['validation']['accuracy'] for r in image_rows}
    qa_path=RUN/'verification.json';qa=json.loads(qa_path.read_text(encoding='utf-8')) if qa_path.exists() else {'status':'not yet run'}
    sanity=qa.get('xai_sanity',[])
    if sanity:
        write_csv(RUN/'xai_sanity.csv',[{k:r.get(k) for k in ['id','method','clean_class_probability','salient_deletion_drop','random_deletion_drop','advantage','flip_correlation','nonconstant','finite']} for r in sanity])
    sanity_description=(f"Observed {len(sanity)} model/method sanity cases: {sum(r['finite'] for r in sanity)} finite maps, {sum(r['nonconstant'] for r in sanity)} nonconstant maps, and {sum(r['advantage']>0 for r in sanity)} positive salient-over-random deletion advantages. These reuse one image and include repeated spatial fallbacks, so they are not independent samples or a method ranking. Full signed values and flip correlations are in [xai_sanity.csv](results/dfdc_unified_20261019/xai_sanity.csv)." if sanity else 'Final XAI verification has not yet completed.')
    software={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','scikit-learn','streamlit','timm']}
    atomic_json(RUN/'software_versions.json',{'python':sys.version,'packages':software})
    body=f'''# Explainable Deepfake Classification: unified image/video experiments

This repository preserves the image-trained ResNet-18 prototype and adds controlled transfer experiments on the **existing local DFDC subset**. The objective is to compare spatial CNN/ViT features, preprocessing, balancing, augmentation, a learned hybrid, and a lightweight temporal branch on one source-grouped video protocol. Results below are generated by `python -m tools.report_unified` from JSON/CSV logs, not manually supplied targets.

The validation-selected video champion is **{champion['id']} — {champion['config']['model']}**, with validation balanced accuracy **{percent(champion['validation']['balanced_accuracy'])}** and one-time held-out test balanced accuracy **{percent(ct['balanced_accuracy'])}**. Test accuracy is **{percent(ct['accuracy'])}**; the all-FAKE test reference scores {percent(tr['accuracy'])} accuracy with 0% REAL recall. These measurements apply to this local subset, not the complete DFDC benchmark or arbitrary images. Predicting FAKE for every local labeled video gives **{percent(audit['all_fake_accuracy'])} accuracy and 50% balanced accuracy**.

## Data and reproducible split

No additional dataset was downloaded. Local media: **{audit['local_videos']} MP4 files**, **{audit['labeled_videos']} labeled** ({audit['label_counts']['REAL']} REAL, {audit['label_counts']['FAKE']} FAKE). `elyvkuldpz.mp4` has no matching label and is logically quarantined in [quarantine.json](results/dfdc_unified_20261019/quarantine.json); it remains on disk but never enters training, calibration, validation, or test. The metadata contains {audit['metadata_entries']} entries, including files absent locally. Filename labels come from supplied local metadata of the Kaggle community mirror; this is not certified official distribution provenance. No label was inferred from a filename or prediction.

Original-to-derivative edges form connected components, including missing-original aliases. All videos in a component share a split; seed 42 searches deterministic group assignments near 70/15/15 without splitting a component. No frame-level split is used. Optional identity IDs are only grouped when supplied; this metadata does not establish actor-disjoint evaluation. Exact-file SHA-256 duplicates are checked across splits. Manifest SHA-256: `{audit['manifest_sha256']}`.

| Split | Videos | REAL | FAKE | Source groups |
|---|---:|---:|---:|---:|
{chr(10).join(splits)}

The common [manifest](results/dfdc_unified_20261019/manifest.csv) records every labeled file's hash, source, split, FPS, frame count, resolution, duration and decode status. [Audit JSON](results/dfdc_unified_20261019/audit.json), [media properties](results/dfdc_unified_20261019/media_properties.json). Decode failures: **{len(audit['decode_errors'])}**. Probe scope: container properties and eight normalized sampled frames per labeled video, not exhaustive decoding of every frame. Failed clips, if any, are excluded uniformly before feature extraction.

| Class | Landscape 1920×1080 | Portrait 1080×1920 |
|---|---:|---:|
{resolution_table}

Orientation proportions differ by class. Raw resizing can retain layout/background cues; cropping is a controlled comparison, not proof that a model learned manipulation traces. FPS and frame-count distributions, including per-class frame-count minimum/median/mean/maximum, are retained in the media-properties JSON. The cache-integrity report gives face fallback counts by split and class.

## Duration analysis and fixed video protocol

| Class | Videos | Total seconds | Min | P10 | P25 | Median | Mean | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(duration)}

{duration_description} Duration is reported as a potential shortcut and is never an input feature. This is descriptive analysis, not a significance claim. The classifier always receives exactly **eight distinct frames**, sampled at bin centers `(i+0.5)/8` of the frame-count timeline. Clips with fewer than eight frames or a failed sampled decode are rejected, never padded or replicated. All sequences have shape `8 × 3 × 224 × 224`, independent of class/duration.

![Duration by class](results/dfdc_unified_20261019/figures/duration.png)
![Duration and natural split distribution](results/dfdc_unified_20261019/figures/distribution.png)

Raw preprocessing decodes, converts BGR to RGB, resizes bilinearly to 224×224, then applies ImageNet normalization. Processed input applies the same YuNet largest-face detector and three-landmark affine alignment (margin 1.3) to both classes, then the same resizing/normalization. Detection failure explicitly falls back to the full frame; it is recorded per frame and is not a model input feature. Cache statuses: `{json.dumps(cache['crop_methods'])}`. No identity tracking or guaranteed face correspondence is claimed. Raw/aligned arrays are stored losslessly in a bounded shared uint8 cache; the same tensors are used by every compatible experiment. Val/test/inference never use stochastic augmentation.

| Split/class | Aligned frames | Full-frame fallbacks | Fallback rate |
|---|---:|---:|---:|
{fallback_table}

Fallback rates differ across splits/classes, especially training FAKE versus validation FAKE. This is a preprocessing/distribution limitation: although detection status is not fed to the head, fallback pixels can retain different background/layout cues. [Cache integrity](results/dfdc_unified_20261019/cache_integrity.json) verifies original hashes, shape, sampled positions and the full metadata source graph. Decoder equivalence on two validation clips is recorded separately; all eight optimized samples were pixel-identical to repeated seeking.

## Transfer training and controlled progression

EfficientNet-B0 and DeiT-Tiny encoders use ImageNet-1K weights and remain frozen/eval, including normalization statistics. Each video is represented by eight 1280-D CNN or 192-D ViT embeddings; training-only feature mean/std standardize embeddings. A small linear head trains using the video mean of frame probabilities. This is feature-based transfer, **not full backbone fine-tuning**. DeiT-Tiny has native transformer self-attention; EfficientNet has native squeeze/excitation. No added ECA/CBAM improvement is claimed.

`S1 → S1p → S1b → S2` changes one factor at a time: raw → aligned face → video-balanced training → augmentation. Balanced training samples whole videos with inverse-class-frequency probabilities, replacement, one training-set length per epoch, and an unweighted loss. There is no simultaneous aggressive loss weighting. Natural validation/test distributions are retained. Augmentation chooses the original or one reproducible extra view per training video, with a shared horizontal flip/brightness/contrast/saturation transform across its eight frames. It does not duplicate a frame to make a sequence. A bounded extra-view feature cache limits augmentation diversity; it is not a permanent expanded labeled dataset.

New heads use seed 42, AdamW (LR .002, weight decay .01), 32-video batches, AMP, up to 24 epochs and patience 5. Checkpoint selection maximizes validation balanced accuracy, then macro-F1, then ROC-AUC. Unique `best.pt`, resumable `last.pt` (optimizer/scaler/epoch state), epoch history and config are retained. Shared frozen feature extraction is separately timed. Real CUDA forward/backward smoke checks precede each budget. No new image-dataset training was run.

The hybrid S5 concatenates frozen CNN and compact ViT embeddings and learns a small common classifier. S6 projects CNN embeddings to 64 dimensions and trains a one-layer 32-hidden LSTM with a video label. S7 is transparent late probability fusion of S2 spatial and S6 temporal scores; its spatial weight is {json.loads((RUN/'S7/config.json').read_text(encoding='utf-8'))['spatial_weight']}, selected from .25/.5/.75 on validation. S8 combines the learned CNN–ViT S5 model with S6, using spatial weight {json.loads((RUN/'S8/config.json').read_text(encoding='utf-8'))['spatial_weight']}; it represents the complete proposed spatial-hybrid/temporal system. S6 is temporal-head-only on spatial features; it still depends on a CNN encoder. Images bypass temporal computation and explicitly use S2 for S6/S7 and S5 for S8. These image fallbacks have no DFDC image-level accuracy claim.

## Validation comparisons (selection data)

Metrics use video labels REAL=0/FAKE=1 and decision threshold .5. The legacy checkpoint's FAKE=0 model output is explicitly remapped. Precision is FAKE precision; macro-F1 includes both classes. ROC-AUC and average-precision PR-AUC use FAKE as positive. Confusion matrices use rows/columns REAL, FAKE. Raw validation scores below precede temperature calibration.

| ID | Model/system | Preprocessing | Balance/augmentation | Attention | Temporal | VAL balanced acc | VAL accuracy | VAL macro-F1 | VAL ROC-AUC | Head training | Forward/video |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Observed progression: face preprocessing changed EfficientNet validation balanced accuracy from {percent(observed['S1']['balanced_accuracy'])} (S1) to {percent(observed['S1p']['balanced_accuracy'])} (S1p), and balancing reached {percent(observed['S1b']['balanced_accuracy'])}. Augmented S2 reached {percent(observed['S2']['balanced_accuracy'])}; its held-out balanced accuracy ({percent(tests['S2']['calibrated']['balanced_accuracy'])}) was lower than S1b ({percent(tests['S1b']['calibrated']['balanced_accuracy'])}). The learned spatial hybrid S5 ({percent(observed['S5']['balanced_accuracy'])} validation) did not outperform processed ViT S4 ({percent(observed['S4']['balanced_accuracy'])}). S6 was selected before test release and retained the strongest measured balanced accuracy; S7/S8 fusion did not improve it. Under this raw-frame protocol, the preserved image-trained S0 predicted REAL for every test video. These are local observations, not statistical superiority claims; no tuning followed test results.

Training times exclude the shared media audit/cache and encoder extraction. S0 has zero DFDC training cost; its historical image training is separate. S7/S8 reuse trained branches and have zero additional training. The forward column uses the [common runtime benchmark](results/dfdc_unified_20261019/runtime_benchmark.json): the same resident eight-frame validation input through each complete deployed encoder/head/fusion path, five warmups and 30 timed repeats. The legacy ResNet runs FP32; new frozen encoders use FP16 autocast and heads use FP32. Decode, face processing, CPU/GPU transfer, XAI and scalar calibration are excluded. Original summaries preserve earlier encoder-plus-head/branch estimates used before test release; retrospective timing does not change the frozen champion. Shared encoder forward time across all extracted views/splits: **{extraction_seconds:.1f} s**, observed peak encoder VRAM **{extraction_vram/1024**2:.1f} MiB**. Per-run head/inference VRAM, complete model artifact sizes, hashes and exact metrics are in [comparison.csv](results/dfdc_unified_20261019/comparison.csv) and [comparison.json](results/dfdc_unified_20261019/comparison.json). Shared encoder files are needed in addition to each small head checkpoint. Final resumed cache pass: {cache['seconds']:.1f} s (earlier partial passes excluded); audit: {audit['audit_seconds']:.1f} s. Cache preparation uses two bounded CPU workers; model experiments run sequentially.

VRAM values use `torch.cuda.max_memory_allocated`, excluding driver/context memory. Timing differences include laptop clock/thermal/runtime variation; a single 30-repeat pass is not an exhaustive latency study. A fusion's slightly lower measured latency does not establish a reduction in computational work. With only 11 REAL validation and 11 REAL test videos, one REAL error shifts REAL recall by 9.09 percentage points and balanced accuracy by 4.55 points.

![Observed validation comparisons](results/dfdc_unified_20261019/figures/comparison.png)
![Executed head training curves](results/dfdc_unified_20261019/figures/training.png)

## Frozen held-out test and calibration

[freeze.json](results/dfdc_unified_20261019/freeze.json) seals checkpoint hashes, calibration temperatures, fusion weight, manifest and champion **before** test feature extraction/prediction. Every completed frozen configuration is evaluated once on the same natural test split; `test.claim` prevents silent reruns. Test metrics did not select a model or change a configuration. The deployment champion is chosen on validation with both class recalls nonzero, then balanced accuracy, macro-F1, AUC and cost. A local champion is not an external benchmark result.

Temperature scaling fits one positive scalar to **validation video** log probabilities/logits by minimizing log loss; no test labels enter calibration. It leaves the .5 class decision unchanged. Below, log loss, Brier and ECE use calibrated video scores; ranking metrics and class decisions remain unchanged. ECE uses ten equal bins of predicted-class confidence on [.5,1]. Small REAL counts make all calibration and recall estimates uncertain. No inconclusive threshold is deployed because this validation cohort cannot establish a reliable selective-risk guarantee. REAL/FAKE probabilities remain visible. Images use uncalibrated spatial scores rather than transferring a video calibration claim.

| ID | TEST balanced acc | Accuracy | REAL recall | FAKE recall | FAKE precision | Macro-F1 | ROC-AUC | PR-AUC/AP | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(testtable)}

![Champion validation and held-out test confusion matrices](results/dfdc_unified_20261019/figures/champion_confusion.png)
![Calibration](results/dfdc_unified_20261019/figures/calibration.png)

Each run directory has `validation_predictions.csv`, `test_predictions.csv`, `summary.json`, `test.json`, full confusion matrices, calibration before/after values and checkpoint hashes. [All-FAKE reference](results/dfdc_unified_20261019/all_fake_reference.json) provides natural-distribution validation/test baselines; its clipped log loss for deterministic errors is not a calibrated model.

## Prior image-only results (kept separate)

The preserved original ResNet-18 reports 97.115% historically; its earlier independently rerun full {old_test['samples']:,}-image test yielded **{percent(old_test['accuracy'])}**, balanced accuracy {percent(old_test['balanced_accuracy'])}, ROC-AUC {old_test['roc_auc']:.7f}. The one-image difference from the historical result was not isolated. Those image labels/splits differ from the new video benchmark. The earlier 12,000-train/2,000-validation image runs measured validation accuracy: E01 ResNet-18 {percent(old_values['E01_resnet18'])}, E02 EfficientNet-B0 {percent(old_values['E02_efficientnet_b0'])}, E03 B0+ECA {percent(old_values['E03_attention'])}, E04 B0+JPEG augmentation {percent(old_values['E04_jpeg'])}. The original had 100,000 training images; those values are not a controlled equal-budget comparison with it. Existing source logs are retained under [execution_20261019](results/execution_20261019); no image test was repeated to choose a DFDC configuration. The original `models/best_resnet18.pth` and `results/champion.json` remain preserved. No new XLSX report was created.

## Explanation and uncertainty interface

The existing Streamlit interface lists only completed, integrity-checked models and compatible methods. It displays the checkpoint, architecture, actual training/evaluation datasets and validation metrics, then image/video REAL/FAKE outputs, both class scores, original/crop/heatmap/overlay, and per-frame crop status.

Image mode defaults to the preserved, image-benchmarked S0; video mode defaults to validation-selected S6. Users can select every available configuration explicitly. S6/S7 image input uses the named S2 spatial fallback, and S8 uses S5; temporal computation is bypassed and the actual fallback checkpoint hash is returned.

* CNN: Grad-CAM, Grad-CAM++ **first-gradient-power approximation**, Integrated Gradients (IG).
* ViT: input-space IG and native class-token attention rollout. Rollout is class-agnostic attention visualization, not proven faithful attribution. CNN CAM is not applied to arbitrary ViT layers.
* Learned CNN–ViT hybrid: IG passes through both selected branches; CAM localizes its CNN branch, and rollout shows its ViT attention. Native attention, added trainable attention and post-hoc explanations are separate concepts.
* Video: eight-frame spatial probability timeline and representative heatmaps. For selected video predictions, gradient methods replace one frame's embedding while holding the other seven fixed and differentiate the complete selected decision. S6/S7 also show spatial/temporal branch probabilities, fusion weight, per-position feature-mean replacement sensitivity and reversed-order temporal sensitivity. The timeline is a prediction trace, not temporal anomaly proof. S6/S7 images clearly identify the S2 spatial fallback and its checkpoint hash.

IG uses a zero normalized (ImageNet-mean RGB) baseline, 24 trapezoid steps, absolute channel-summed maps, and reports a numerical completeness residual. Explanations use the actual selected checkpoint; they describe model behavior and never prove manipulation. The validation-only XAI checks retain nonfinite/constant-map checks, flip stability and salient versus random mean-color deletion when available; mean masking is out of distribution and negative results remain visible. Small software/sanity checks cannot establish population-level faithfulness. [Verification log](results/dfdc_unified_20261019/verification.json): `{json.dumps({k:v for k,v in qa.items() if k in ['passed','model_count','inference_cases','streamlit_cases','cpu_fallback']})}`.

{sanity_description}

Uploads accept JPEG/PNG or MP4/MOV, validate decoding, preserve the original bytes under a SHA-256-addressed local path, and display the actual file hash, size/type and available EXIF/image metadata or video FPS/resolution/duration/frame count/sample count and crop failures. Hash and metadata establish provenance, not fake evidence. Uploaded originals remain locally in `data/dfdc/runtime/uploads`; they count against the storage budget.

## Run locally and reproduce

Hardware used: NVIDIA RTX 4060 Laptop GPU (8 GB), Ryzen 7 7840HS, approximately 16 GB RAM, existing CUDA-enabled `dfenv`. Frozen extraction uses eight-frame AMP batches; training uses 32 cached video sequences and zero data-loader workers. CPU inference fallback is supported; real end-to-end CPU verification is recorded separately. No training on CPU was substituted for the reported CUDA runs. Additional phase storage including downloaded media, caches, model weights and reports is **{check_budget()['phase_bytes']/1024**3:.2f} GiB / 30 GiB** (environment baseline excluded; added dependency reserve included). A 5 GiB free-disk reserve is enforced.

Executed package versions: `{json.dumps(software)}`. [Software/Python provenance](results/dfdc_unified_20261019/software_versions.json), [pretraining weight hashes](results/dfdc_unified_20261019/pretrained_provenance.json), and the local detector's `data/dfdc/runtime/yunet.json` bind the runtime and source weights. Shared encoder snapshots and unique head checkpoints retain independent SHA-256 hashes.

From PowerShell:

```powershell
Set-Location D:\\sundarCollege\\Deepfake_Detect\\Deepfakereserch
.\\dfenv\\Scripts\\Activate.ps1
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
# Or: .\\tools\\start_demo.ps1
python -m unittest discover -s tests -v
# Reproducible selected-model video inference (choose a fresh output directory):
python -m inference.unified --model S6 --video data/dfdc/train_sample_videos/dhjnjkzuhq.mp4 --method gradcam --output results/inference/new_video
```

Research stages are idempotent for completed outputs and resume partial training without changing configs. Existing final results are preserved:

```powershell
python -m training.video_protocol audit
python -m training.video_protocol cache
python -m tools.run_unified
python -m tools.verify_unified
python -m tools.report_unified
```

`run_unified` executes train/validation frozen embeddings → sequential heads → fusion → freeze → test embeddings → one-time test metrics. It never launches a download script. Frozen test claims require explicit investigation after interruption. To run a genuinely new protocol, use a new run/cache namespace rather than editing these sealed results. Pretrained encoder weights and head checkpoints are local ignored artifacts, not committed data. Install requirements in `dfenv` only if recreating the environment; the verified local runtime includes `timm==1.0.22`.

## Limitations, skipped work and next experiments

This is a small, highly imbalanced convenience subset with relatively few independent source groups and REAL validation/test videos. Related derivatives are grouped, but unidentified actors may cross groups. Source-level bootstrap confidence intervals and an external distribution test were not run. There is no claim of complete DFDC, cross-dataset video generalization, 95–97% DFDC accuracy, semantic artifact diagnosis, or monotonic improvement from face processing, balance, augmentation, native attention, hybridization or temporal modeling. Eight samples omit most moments; largest-face fallback and frame-count-based seeking can fail on unusual containers or variable-frame-rate media. Decode probes are limited to the sampled positions.

The first temporal-XAI integration run failed because cuDNN evaluation-mode RNNs cannot backpropagate. The repaired explanation path evaluates the same LSTM weights with the native differentiable backend; normal prediction/training are unchanged. A CUDA regression compares native explanation scores with normal inference and checks input gradients. Initial failed logs remain preserved. UTF-8 model-name decoding on Windows was also corrected without altering numerical results. Constant maps are displayed with an explicit localization-unavailable notice.

Frozen features and a small fixed augmentation-view cache limit adaptation. No added ECA/CBAM, temporal transformer, full end-to-end fine-tuning or independent DFDC image benchmark was completed in this phase; these are explicitly deferred, not reported as successful experiments. S7 reuses branch checkpoints and its cost estimate is not an independent measurement. Temperature and head selection share the validation subset, so calibration estimates there are optimistic. The same held-out test is shared across frozen comparisons, and small counts limit conclusions. Runtime CUDA mixed precision versus FP32 explanation computation may introduce small numerical differences; validation checks compare deployed scores with cached protocol scores. Further work should preregister a new source/actor-aware cohort, increase REAL group coverage, use a separate calibration set, study attribution stability on more clips and benchmark end-to-end latency under representative loads.
'''
    refinement=ROOT/'results/debug_refinement_20261019'
    if (refinement/'README_SECTION.md').exists():
        from tools.report_refinement import update_readme
        body=update_readme(body,refinement)
    (ROOT/'README.md').write_text(body,encoding='utf-8')
    print(f'Generated README and {len(full)} observed comparison rows',flush=True)


def make_figures(summaries,tests,champion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    folder=RUN/'figures'
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(summaries));ax.bar(x-.18,[s['validation']['balanced_accuracy'] for s in summaries],.36,label='Validation balanced accuracy');ax.bar(x+.18,[s['validation']['accuracy'] for s in summaries],.36,label='Validation accuracy')
    ax.axhline(.5,color='black',ls='--',lw=1,label='All-FAKE balanced accuracy');ax.set_xticks(x,[s['id'] for s in summaries]);ax.set(ylim=(0,1.06),ylabel='Video score',title='Observed controlled DFDC validation results');ax.legend();fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for summary in summaries:
        path=RUN/summary['id']/'history.json'
        if not path.exists():continue
        history=json.loads(path.read_text(encoding='utf-8'));epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['loss'] for r in history],label=summary['id'])
        axes[1].plot(epochs,[r['validation']['balanced_accuracy'] for r in history],label=summary['id'])
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='Executed frozen-feature head training')
    axes[1].set(xlabel='Epoch',ylabel='Balanced accuracy',title='Validation, never test selection',ylim=(0,1.02))
    for ax in axes:ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(folder/'training.png',dpi=150);plt.close(fig)
    val=next(s for s in summaries if s['id']==champion)['validation'];test=tests[champion]['calibrated']
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,metrics,title in zip(axes,[val,test],['Validation (selection)','Held-out test (frozen)']):
        cm=np.array(metrics['confusion_matrix_real_fake']);ax.imshow(cm,cmap='Blues')
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['REAL','FAKE'],yticklabels=['REAL','FAKE'],xlabel='Predicted',ylabel='True',title=f'{champion}: {title}')
    fig.tight_layout();fig.savefig(folder/'champion_confusion.png',dpi=150);plt.close(fig)
    with (RUN/champion/'test_predictions.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    y=[int(r['label']=='FAKE') for r in rows];fig,ax=plt.subplots(figsize=(6,4))
    for key,label in [('raw_fake_probability','Raw'),('fake_probability','Validation-temperature')]:
        observed,predicted=calibration_curve(y,[float(r[key]) for r in rows],n_bins=6,strategy='uniform');ax.plot(predicted,observed,'o-',label=label)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Predicted FAKE probability',ylabel='Observed FAKE fraction',title=f'{champion} held-out reliability (small bins)');ax.legend();fig.tight_layout();fig.savefig(folder/'calibration.png',dpi=150);plt.close(fig)


if __name__=='__main__':build()
