"""Final implementation report from recorded local evidence, never fixture accuracy."""
import ast
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, RESEARCH, atomic_json, check_budget, tree_bytes


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    master=ROOT.parent/'Literature_Analysis.xlsx'
    original=read(RESEARCH/'analyses/relocation_audit.json')
    assert sha256(master)==original['master_sha256']
    relocations=[]
    for row in original['moves']:
        target=Path(row['to']);assert target.is_file(),str(target)
        digest=sha256(target)
        if target.suffix.lower()=='.xlsx':assert digest==row['sha256']
        relocations.append({'path':str(target),'original_sha256':row['sha256'],'current_sha256':digest,'changed_after_relocation':digest!=row['sha256']})
    source_workbooks=[Path(r['to']) for r in original['moves'] if Path(r['to']).suffix.lower()=='.xlsx']
    broken_old_references=[]
    for path in source_workbooks:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith('.rels'):
                    text=archive.read(name).decode('utf-8')
                    if 'Deepfakereserch/results/' in text or 'Deepfakereserch\\results\\' in text:
                        broken_old_references.append({'workbook':str(path),'part':name})
    if broken_old_references:raise ValueError('Moved workbook has an old implementation artifact link: '+str(broken_old_references))
    for folder in ['training','evaluation','models','inference','explainability','tools','tests']:
        for path in (ROOT/folder).glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    assert not list((ROOT/'results').glob('*.xlsx'))
    dependency=read(ROOT/'results/dependency_storage.json')
    dependency['environment_after_bytes']=tree_bytes(ROOT/'dfenv')
    dependency['environment_added_bytes']=max(0,dependency['environment_after_bytes']-dependency['environment_baseline_bytes'])
    atomic_json(ROOT/'results/dependency_storage.json',dependency)
    test_log=(ROOT/'results/transfer_tests.log').read_text(encoding='utf-8')
    tests=int(re.search(r'Ran (\d+) tests',test_log)[1]);assert '\nOK\n' in test_log
    smoke=read(ROOT/'results/smoke/architecture_smoke.json');assert smoke['passed']
    app=read(ROOT/'results/smoke/app_smoke.json');assert app['passed']
    audit=read(ROOT/'results/hardware_audit.json');acquisition=read(ROOT/'results/dfdc_acquisition_plan.json');access=read(ROOT/'results/dfdc_access_status.json')
    assert sha256(ROOT/'models/best_resnet18.pth')==smoke['legacy_checkpoint_sha256']
    videos=list((ROOT/'data/dfdc/train_sample_videos').glob('*.mp4'))
    real=list((ROOT/'results/experiments').glob('*/summary.json'))
    workbook=RESEARCH/'workbooks/DFDC_Transfer_Learning_Experiments.xlsx';assert workbook.exists()
    storage=check_budget()
    # Keep the report/JSON out of implementation; machine resource snapshots stay local.
    evidence={'master_sha256':sha256(master),'legacy_checkpoint_sha256':smoke['legacy_checkpoint_sha256'],
              'relocations_verified':relocations,'workbook_old_path_links':broken_old_references,'tests_passed':tests,'app_check':app,
              'official_videos_obtained':len(videos),'official_video_bytes':sum(p.stat().st_size for p in videos),'access':access,
              'environment_added_bytes':dependency['environment_added_bytes'],'conservative_storage':storage,
              'research_workbook':str(workbook),'research_workbook_sha256':sha256(workbook),'completed_official_training_runs':len(real)}
    atomic_json(RESEARCH/'analyses/transfer_implementation_audit.json',evidence)
    table='\n'.join(f"| {r['backbone']} | {r['added_attention']} | {r['fusion']} | {r['fine_tune_blocks']} | {r['peak_gpu_mib']:.2f} |" for r in smoke['architectures'])
    gib=1024**3
    report=f'''# Transfer-learning implementation report

**DFDC experimentation is blocked by official Kaggle HTTP {access.get('http_status')}.** Authentication and the official CLI inventory succeed; sample metadata download is forbidden. No DFDC videos, real DFDC model training, validation scores, champion or final test result are reported. No classification-accuracy claim is inferred from a synthetic software test.

## Directory structure

```text
Deepfake_Detect/                          research workspace (not a Git root)
  Literature_Analysis.xlsx              authoritative master, unchanged
  papers/ and sources/                  original PDFs/archive, retained
  research/
    workbooks/                          two relocated comparisons and experimental workbook
    reports/                            paper analyses and this implementation report
    analyses/                           source/relocation audits and verified experiment export
    working/                            relocated PDF extracts/renders and workbook previews
  Deepfakereserch/                       implementation Git root
    configs/                            9 controlled experiment configurations
    training/                           metadata/splits, bounded crops and training
    models/                             preserved legacy model plus transfer/attention/fusion
    evaluation/                         video/frame metrics, validation selection and final-test gate
    inference/ and explainability/      image/video prediction, CAM and quantitative checks
    app.py                              local Streamlit image/video application
    data/dfdc/                          ignored media, lossless crop cache, detector and D: caches
    results/                            ignored experiment checkpoints/predictions and software logs
    dfenv/                              existing environment, retained
```

{original['files_moved']} research files were relocated only after size and SHA-256 verification. The two relocated research workbooks remain byte-for-byte identical. Report links were updated after relocation and their before/after hashes are recorded. All earlier uncommitted implementation files were preserved. No commit, reset, dataset upload or Git push was performed.

Master SHA-256: `{evidence['master_sha256']}`.

## Dataset and storage

- Official CLI inventory: {acquisition['selected_files']} files, {acquisition['selected_bytes']:,} bytes ({acquisition['selected_bytes']/gib:.4f} GiB). Kaggle exposes 400 sample MP4s and metadata as expanded official files, without a listed sample ZIP.
- Obtained DFDC videos: **{len(videos)}**. Video payload: **{evidence['official_video_bytes']:,} bytes**. Metadata permission failure occurred before any sample media transfer.
- REAL/FAKE counts, source relationships, actual resolutions/FPS/durations and split sizes remain **unmeasured** because official metadata/media are unavailable.
- Conservative counted project/phase footprint: **{storage['phase_bytes']/gib:.3f} GiB / 30 GiB**. This includes relocated existing research, old baseline checkpoint, controlled code/results, new detector, package downloads/cache and measured environment growth, so it exceeds the true additional footprint.
- Measured environment increase after installing missing Streamlit dependencies and subsequent imports: **{dependency['environment_added_bytes']/1024**2:.2f} MiB**. Every pre-existing package version stayed fixed. No new environment was created.
- D: free at the final audit: **{storage['free_bytes']/gib:.2f} GiB**. A 5 GiB headroom and the 30 GiB ceiling are checked before downloads/checkpoint writes and during processing. Temporary synthetic checkpoints existed only in test-created ignored directories and were removed after verification.
- The official 232,589-byte OpenCV YuNet detector is the only new model payload. New classifier pretrained weights have not been downloaded. No FF++, Celeb-DF, full DFDC download or additional chunk was requested.

## Implemented and tested pipeline

The synthetic vertical slice exercises metadata -> video probes -> deterministic source-group splits -> bounded lossless frame cache -> frozen-feature head training -> checkpoint -> validation frame/video predictions -> recomputed metrics. Provenance checks prevent synthetic fixtures from entering official training or the experiment workbook. Actor disjointness cannot be asserted from sample metadata; optional supplied actor IDs are grouped. Exact duplicate videos crossing splits are rejected.

The shared planned real-data protocol uses four deterministic frames/video, YuNet largest-face selection, optional eye/nose landmark alignment, an explicit counted full-frame no-face fallback, RGB resize 224 and ImageNet normalization. Validation/test receive no augmentation. The primary score averages sampled frame FAKE probabilities. FAKE is output index 0 but positive evaluation label 1. The old checkpoint is preserved.

**{tests} unit tests passed**, covering the original DFDC contracts plus attention/fusion gradients, frozen BatchNorm, class labels, manifest/cache integrity, metrics, duplicate-frame rejection, deterministic transforms, both CAM methods, safe selection/final-test refusal and resumed weights matching an uninterrupted run. Corrupt-video fixture warnings are expected. The installed environment passes `pip check`; implementation Python files parse and `git diff --check` passes.

Streamlit headless tests passed for both upload views and explanation controls. Actual image/video inference and explanation generation passed through the media API using synthetic input and the existing checkpoint. This establishes functionality, not classification quality or a browser-upload performance benchmark. Windows asyncio testing needed local loopback permission outside the execution sandbox.

## Models, experiments and hyperparameters

**Models trained on DFDC: none.** All nine real experiment configurations remain **Not evaluated**:

| Configuration | Controlled purpose | DFDC status |
|---|---|---|
| A1 ResNet-18 frozen | Initial pretrained CNN baseline | Not evaluated |
| A2 EfficientNet-B0 frozen | Compact CNN with native SE | Not evaluated |
| A3 MobileNetV3-Large frozen | Efficient CNN with native SE | Not evaluated |
| A4 ResNet-18 layer4 tuning | Additional selective fine-tuning after A1 | Not evaluated |
| B1 / B2 ResNet-18 ECA / CBAM | Added classification attention | Not evaluated |
| C1 / C2 JPEG / Gaussian noise | Separate augmentation changes from basic augmentation | Not evaluated |
| D1 ResNet-18 + FFT descriptor | Lightweight spatial/frequency fusion | Not evaluated |

Defaults are seed 42, 224 px, 4 frames/video, batch 4, accumulation 8, AdamW, head LR 0.001, backbone LR 0.00001 when unfrozen, weight decay 0.0001, class-weighted loss, AMP, 3 head epochs, patience 2 and a 30-minute limit per invocation. A4 starts from A1 and adds 3 selective epochs; its extra compute is reported rather than claiming equal total training budgets. BatchNorm statistics remain frozen. No uncontrolled model queue exists.

Paper 2 informed attention and frame averaging, and Paper 35 informed lightweight fusion. Papers 9/45 are FF++ methodology references; Papers 15/16 are surveys. Existing source analyses were reused, original PDF fingerprints verified, and paper IDs/worksheet names preserved. No published score is used as a local DFDC measurement. Compact ViT, prototypes and costly LIME/SHAP sweeps remain deferred until measured baseline evidence justifies them.

## Measured hardware feasibility

RTX 4060 Laptop GPU, 8 GB. Audit PyTorch: {audit['torch']}; torchvision: {audit['torchvision']}. Physical RAM: {audit['memory']['total_ram_bytes']/gib:.2f} GiB; available at the initial audit: {audit['memory']['available_ram_bytes']/gib:.2f} GiB. Low host RAM is a practical constraint. Models are processed sequentially with zero loader workers.

The following are **synthetic forward/backward software checks**, 224×224 and batch 2, random initialization. Values are PyTorch peak allocated memory, not total driver/reserved VRAM and not a guarantee for a larger batch. One optimizer step is not a completed research experiment.

| Backbone | Added attention | Fusion | Fine-tuned late blocks | Peak GPU MiB |
|---|---|---|---:|---:|
{table}

Both CAM methods and video averaging passed. YuNet produced `{smoke['yunet_existing_face_status']}` on a pre-existing local face image and correctly recorded a no-face fallback on a blank fixture.

## Measured research results and champion

Validation accuracy, balanced accuracy, precision, recall, F1, AUC, log loss, confusion matrix and real-data inference cost: **not measured**. Attention/augmentation/fusion gains: **not evaluated**. Transfer-learning hypothesis: **untested on DFDC**. Final held-out test: **not evaluated**. Champion: **not selected**. No champion checkpoint exists.

The application currently uses the preserved `Deepfakereserch/models/best_resnet18.pth`, clearly labeled as the existing image-trained baseline without DFDC validation. SHA-256 remains `{smoke['legacy_checkpoint_sha256']}`. It will load a hash-verified `results/champion.json` selection after comparable validation experiments complete.

Scores are labeled uncalibrated. Grad-CAM indicates influential regions, not forensic proof. Grad-CAM++ uses a documented gradient-power approximation; no Paper 45 reproduction is claimed. Optional deletion/random-mask and flip-stability checks are separate from classification metrics and are not reported as DFDC evidence here.

## Workbook and executable commands

Experimental workbook: `research/workbooks/DFDC_Transfer_Learning_Experiments.xlsx`. Five required sheets are present. All unavailable metrics remain blank; all planned configurations say Not evaluated. There are no fabricated data series or placeholder numeric charts. Saved-file structural checks, formula-error scan and rendered-sheet reviews passed. No native Excel application recalculation test was performed; no new metric formulas have evaluated data to calculate yet.

From `D:\\sundarCollege\\Deepfake_Detect\\Deepfakereserch`:

```powershell
# First action after the official account obtains competition-data permission:
.\\dfenv\\Scripts\\python.exe -m tools.acquire_dfdc_cli --download
.\\dfenv\\Scripts\\python.exe -m training.dfdc_data
.\\dfenv\\Scripts\\python.exe -m training.dfdc_cache

# Compare the old checkpoint only on validation, then train A1.
.\\dfenv\\Scripts\\python.exe -m evaluation.experiments baseline
.\\dfenv\\Scripts\\python.exe -m training.experiment --config configs/resnet18_frozen.json

# After individually reviewed experiments, seal selection and test once.
.\\dfenv\\Scripts\\python.exe -m evaluation.experiments select
.\\dfenv\\Scripts\\python.exe -m evaluation.experiments test

# Export verified saved results and rebuild the workbook outside the repo.
.\\dfenv\\Scripts\\python.exe -m tools.export_experiments
& 'C:\\Users\\Rangarajan\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\bin\\node.exe' tools/build_transfer_workbook.mjs

# Current local image/video application, using the preserved fallback model.
.\\dfenv\\Scripts\\python.exe -m streamlit run app.py
```

The first real training command was executed and correctly stopped on missing official cache input before any classifier-weight download or claimed experiment completion. The exact outstanding blocker is Kaggle HTTP {access.get('http_status')} for sample metadata. Review the account's [competition access/rules](https://www.kaggle.com/competitions/deepfake-detection-challenge/rules); authentication alone is insufficient. If access remains unavailable, official locally obtained DFDC files plus auditable provenance are required. No access control was bypassed.

Full component commands, metric definitions, resource limits and remaining methodological limitations are in `Deepfakereserch/docs/TRANSFER_LEARNING.md`. Machine evidence is in `research/analyses/transfer_implementation_audit.json`, the relocation audit, the experiment export, and ignored implementation test/smoke logs.
'''
    target=RESEARCH/'reports/Transfer_Learning_Implementation_Report.md'
    target.write_text(report,encoding='utf-8')
    atomic_json(ROOT/'results/storage_ledger.json',check_budget())
    print(json.dumps({'report':str(target),'tests_passed':tests,'official_videos':len(videos),'phase_gib':check_budget()['phase_bytes']/gib,'preserved_master':True,'preserved_legacy_checkpoint':True},indent=2))


if __name__=='__main__':main()
