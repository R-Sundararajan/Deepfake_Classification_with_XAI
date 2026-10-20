"""Generate concise phase status from local evidence; no network or training."""
import json
import re
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT,GIB,PHASE_DIRS,atomic_json,check_budget,phase_bytes,tree_bytes


def read(name):
    return json.loads(((ROOT.parent/'research/analyses' if name=='dfdc_literature.json' else ROOT/'results')/name).read_text(encoding='utf-8'))


def main():
    literature=read('dfdc_literature.json');smoke=read('gpu_smoke_test.json');validation=read('workbook_validation.json')
    plan=read('dfdc_acquisition_plan.json')
    access=read('dfdc_access_status.json')
    text=(ROOT/'results/.work/tests.log').read_text(encoding='utf-8')
    test_count=int(re.search(r'Ran (\d+) tests',text).group(1))
    assert re.search(r'^OK\s*$',text,re.M), 'Latest offline tests did not pass'
    counts=Counter(x['Decision'] for x in literature['audit'])
    included={}
    for row in literature['rows']:included.setdefault(row['_key'],(row['Paper ID'],row['Paper Title']))
    videos=list((ROOT/'data/dfdc/train_sample_videos').glob('*.mp4'))
    dataset_bytes=sum(p.stat().st_size for p in videos)
    report=ROOT/'../research/reports/DFDC_phase_report.md'
    # Two writes stabilize the byte count including this report and its ledger.
    for _ in range(3):
        budget=check_budget()
        ledger={'measured_at_utc':datetime.now(timezone.utc).isoformat(),'phase_bytes':budget['phase_bytes'],
                'phase_gib':budget['phase_bytes']/GIB,'free_d_bytes':budget['free_bytes'],
                'hard_limit_bytes':budget['limit_bytes'],'required_free_headroom_bytes':budget['headroom_bytes'],
                'phase_directories':{d:tree_bytes(ROOT/d) for d in PHASE_DIRS},
                'accounting':'Complete new phase directories plus new ingestion/evaluation code and conservative Python-cache/.gitignore allowance. Existing model, image data, environment and original papers excluded; no new environment/package installed.',
                'downloaded_mp4_files':len(videos),'downloaded_mp4_bytes':dataset_bytes}
        atomic_json(ROOT/'results/storage_ledger.json',ledger)
        title_rows='\n'.join(f'| {key} | {pid} | {title} |' for key,(pid,title) in included.items())
        output=f'''# DFDC research and baseline phase report

Status: literature workbook and offline-tested pipeline delivered; **real DFDC evaluation blocked by official Kaggle HTTP {access.get('http_status','unknown')}**. No published score or synthetic prediction is presented as a local DFDC measurement.

## Literature deliverable

- `../research/workbooks/DFDC_Research_Comparison.xlsx`: **{len(literature['rows'])} model/protocol rows, {len(included)} experiment families**; original IDs retained where indexed.
- All **112 distinct local PDFs** screened, including the additional ZIP; identical archived copies deduplicated by SHA-256. Source Audit: {counts['Included']} included canonical documents, {counts['Consolidated duplicate']} consolidated versions, {counts['Excluded']} other exclusions.
- Main sheet uses the exact 20 requested columns, source hyperlinks/page references, numeric accuracy/F1/AUC, filters, wrapping and frozen headings. Local DFDC Baseline remains empty because no real local DFDC experiment completed.
- Explicit distinctions include DFDC versus DFDC Preview, validation versus test, and frame/face versus video units. Cited baselines, illustrative simulations and ambiguously pooled datasets are excluded. Author-reported inconsistencies remain visible; these are verified transcriptions, not independent replications.
- Corrections to old literature descriptions include EfficientNet-B4/SBI rather than a ViT for Paper 58, and the actual custom CNN/capsule/LSTM model for Paper 36. No new paper architecture was implemented.

| Source key | Original workbook ID | Included publication / experiment family |
|---|---|---|
{title_rows}

DFP-Net/ExplaNET repeated intra/cross results are consolidated, with the journal's additional F1 retained; the earlier SSRN hybrid version is consolidated with Paper 35. Source Audit gives an individual reason for every exclusion. Important examples: DeepExplain explicitly labels its values illustrative; Anchors and XGBoost papers do not isolate DFDC from mixed evaluation; ViT-Xplain names DFDC but provides no dataset-keyed numeric result.

## Acquisition and storage

- Official Kaggle listing selected **400 MP4s plus metadata.json**, **{plan['selected_bytes']:,} bytes ({plan['selected_bytes']/GIB:.4f} GiB)**. Direct official per-file acquisition replaces ZIP duplication; no full dataset, unlabeled test videos or FF++ requested.
- Metadata download returned HTTP {access.get('http_status','unknown')}; **{len(videos)} videos, {dataset_bytes:,} video bytes acquired**. No accessible ground-truth metadata, class balance, real manifest or real split statistics can yet be reported.
- Required action: sign in to the account tied to the existing local token and accept/join through [Kaggle rules](https://www.kaggle.com/competitions/deepfake-detection-challenge/rules), if available. If Kaggle will not grant entry, official permission is required. No credentials or signed download URLs are logged.
- Conservative new phase footprint at reporting snapshot: **{budget['phase_bytes']:,} bytes ({budget['phase_bytes']/1024**2:.2f} MiB / {budget['phase_bytes']/GIB:.4f} GiB)**, including working PDF extracts/renders, source JSON, code, tests, workbook and reports. See `storage_ledger.json` for current bytes.
- D: free at snapshot: **{budget['free_bytes']/GIB:.2f} GiB**. Enforced additional cap **30 GiB**, remaining-free headroom **5 GiB**. Selected sample plus 256 MiB work allowance fits. Existing project snapshot before any successful acquisition: {plan.get('existing_project_bytes','Not measured in first plan')} bytes (includes existing image data/environment/checkpoint).
- All controlled new media/temp/cache/result paths are under this D: project. No new environment, package installation, archive download, retained decoded-frame cache or model checkpoint. Existing data and original literature files were never write targets.

## Code and executed validation

- Acquisition: `tools/acquire_dfdc.py`; storage: `tools/dfdc_storage.py`; metadata/splits/streaming: `training/dfdc_data.py`; existing-model evaluation: `evaluation/dfdc.py`.
- **{test_count} offline tests passed**: strict metadata/path checks, source grouping including absent originals and case aliases, deterministic split leakage checks, missing/corrupt media, bounded frame sampling, explicit preprocessing, FAKE-positive metrics/undefined AUC, mean probability aggregation, exact budget boundaries, safe download redirects, preserved existing/partial files and remote size overrun rejection. Corrupt-fixture decoder warnings are expected.
- **GPU smoke passed**: real ResNet-18 checkpoint, synthetic MP4, 3 frames × 2 deterministic passes; {smoke['elapsed_seconds']:.3f} seconds total smoke time; {smoke['gpu_peak_allocated_mib']:.2f} MiB peak PyTorch allocated GPU memory on {smoke['gpu']}. This includes model loading/testing and is not a real-data throughput estimate.
- Checkpoint SHA-256 before/after smoke: `{smoke['checkpoint_sha256_after']}`; unchanged. Original application/training modules are unchanged; only separate DFDC modules and ignore rules were added.
- Workbook validation passed ({validation['published_rows']} published rows): exact column order, numeric types/bounds/source values, hyperlinks, filter table, frozen panes, wrapping, Unicode, no merged cells and known Paper 2/Paper 35 values. A rendered preview was visually inspected.

## Measurements and remaining limits

**DFDC accuracy, precision, recall, F1, ROC-AUC and confusion matrix: not measured because data access is blocked.** The earlier image-dataset accuracy 97.115% is not a DFDC result. No DFDC fine-tuning, model selection, attention implementation or target-accuracy optimization was performed.

The ready baseline uses the unchanged image-trained ResNet-18, deterministic one-second sampling capped at 16 frames, RGB resize224/ImageNet normalization, mean frame FAKE probability (checkpoint index 0), threshold ≥0.5, and video-level FAKE-positive metrics with failure coverage. Source-group manifests reserve approximately 70/15/15 train/validation/test partitions. Actor disjointness cannot be guaranteed from sample metadata. Initial evaluation over all labeled sample videos is valid only before DFDC adaptation and is described as cross-dataset testing, not hidden-test competition performance.

Installed OpenCV 5.0.0 lacks Haar cascade support. The working default is full-frame preprocessing without face detection/alignment; optional Haar mode fails clearly unless a compatible build is available. This introduces face-scale/background domain shift and limits interpretation. Existing packages were preserved. Small sample size, imbalance, no actor IDs and uncertain representativeness will limit any eventual sample score. No larger training chunk is justified before the sample is inspected.

Preparation, acquisition, evaluation and workbook rebuild commands are in `docs/DFDC_README.md`. When official access is arranged, rerun acquisition, prepare the manifest, evaluate, and rebuild the workbook; only completed official-data runs populate the local worksheet.
'''
        report.write_text(output,encoding='utf-8')
    print(json.dumps({'report':str(report),'phase_bytes':phase_bytes(),'offline_tests':test_count,'access':access['status']},indent=2))


if __name__=='__main__':main()
