"""Build the audited literature register from local original-paper evidence.

Metrics are transcribed published claims, not independently reproduced results.
PDF page numbers below are one-based, including cover pages.
Run audit_dfdc_sources.py first to refresh the machine-readable PDF inventory.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '../research/working'
HEADERS = ['Paper ID', 'Paper Title', 'Year', 'Architecture / Models',
           'Attention Mechanism', 'Fusion Strategy', 'Dataset', 'Dataset Volume',
           'Input Resolution', 'Training Compute', 'Evaluation Protocol',
           'Accuracy (%)', 'F1 Score', 'ROC-AUC', 'Other Metrics', 'Source',
           'Data Balancing', 'Data Augmentation', 'Optimization Techniques',
           'Limitations / Notes']
inventory = json.loads((WORK / 'paper_inventory.json').read_text(encoding='utf-8'))
original = json.loads((WORK / 'workbook_records.json').read_text(encoding='utf-8'))
rows = []


def base(key, title, year, **kw):
    src = next(x for x in inventory if x['key'] == key)
    d = dict.fromkeys(HEADERS, 'Not reported')
    d.update({'Paper ID': ', '.join(map(str, src['literature_ids'])) or 'Not indexed',
              'Paper Title': title, 'Year': year, 'Attention Mechanism': 'None reported',
              'Fusion Strategy': 'None (single model)', 'Dataset': 'DFDC (full-release benchmark)',
              'Accuracy (%)': None, 'F1 Score': None, 'ROC-AUC': None,
              'Other Metrics': 'Not reported', 'Limitations / Notes': '',
              '_key': key, '_source_path': src['source'], '_sha256': src['sha256']})
    d.update(kw)
    return d


def add(b, model, source, acc=None, f1=None, auc=None, **kw):
    d = dict(b)
    d.update({'Architecture / Models': model, 'Source': source,
              'Accuracy (%)': acc, 'F1 Score': f1, 'ROC-AUC': auc})
    d.update(kw)
    rows.append(d)


b = base('pdf_008', 'Deepfake forensics analysis: An explainable hierarchical ensemble of weakly supervised models', 2022,
         **{'Dataset Volume': 'Actual training volume: Not reported. Describes >128,000 videos; tests 5,000 clips; samples 10% of frames.',
            'Input Resolution': '320 × 320 aligned RetinaFace crops',
            'Evaluation Protocol': 'Authors’ DFDC experiment; p. 5 describes frame measures, p. 9 averages frame probabilities for 5,000 clips; Table 2 unit is ambiguous.',
            'Data Augmentation': 'Horizontal flip; Gaussian/ISO noise; motion/Gaussian blur; hue/saturation/brightness/contrast; sharpen, emboss, sepia; guided crop/drop for attention models.',
            'Optimization Techniques': 'Weak supervision; attention models use bilinear attention pooling and center loss. Optimizer, LR, batch size, epochs: Not reported.',
            'Limitations / Notes': 'Table 2 retained over inconsistent abstract headline. Full DFDC naming retained from this original; later papers sometimes relabel its cited results DFDC-P. F1 not reported. No local reproduction.'})
for model, acc, auc in [('Xception',83.88,.9107),('Xception + attention',91.44,.9724),('EfficientNet-B3 + attention',80.23,.9160),('Hierarchical ensemble: Xception + Xception-attention + EfficientNet-B3-attention',92.20,.9752)]:
    add(b, model, 'PDF pp. 5–10; Table 2, PDF p. 10', acc, auc=auc,
        **{'Attention Mechanism': 'None reported' if model=='Xception' else 'Learned convolution/BN attention maps; bilinear attention pooling; attention-guided cropping/dropping (attention components).',
           'Fusion Strategy': 'Weighted final probabilities from 3 separately trained classifiers; weights not reported.' if model.startswith('Hierarchical') else 'None (component evaluation)'})

b = base('pdf_041', 'A Hybrid Deep Learning and Forensic Approach for Robust Deepfake Detection', 2025,
         **{'Input Resolution': '224 × 224 MTCNN-aligned frames; 25 fps',
            'Training Compute': 'NVIDIA GPU, model not specified; mixed precision. T4 16 GB / Xeon / 32 GB describes inference, not confirmed training hardware.',
            'Evaluation Protocol': '80/10/10 stratified split; frame/video split unit, source grouping and aggregation not reported.',
            'Data Augmentation': 'Horizontal flip, color jitter, Gaussian noise; JPEG qualities 50/75/100.',
            'Optimization Techniques': 'Adam, LR 1e-4, cosine annealing, dropout; batch/epochs not reported.',
            'Limitations / Notes': 'DFDC Tables V/VII only; FF++ 97% excluded. Actual DFDC subset volume and ViT variant unspecified. SSRN version pdf_109 repeats these four results and is deduplicated.'})
for model,acc,p,r,f,a in [('PRNU + JPEG + DCT forensic features',67,.65,.60,.62,.68),('ResNet-50',72,.71,.69,.70,.74),('Vision Transformer (variant not specified)',74,.73,.71,.72,.77),('Hybrid: ResNet-50 + ViT + PRNU/JPEG/DCT',79,.78,.76,.77,.82)]:
    add(b,model,'Table V, PDF p. 6; Table VII, PDF p. 7',acc,f,a,
        **{'Other Metrics': f'Precision {p:.2f}; recall {r:.2f}; averaging convention not reported.',
           'Attention Mechanism': 'ViT self-attention; heads/depth not reported.' if 'ViT' in model or 'Transformer' in model else 'None reported',
           'Fusion Strategy': 'Concatenate visual/forensic features; learned fully connected fusion + dropout classifier.' if model.startswith('Hybrid') else 'None (ablation component)'})

b = base('pdf_025', 'X²-DFD: A framework for eXplainable and eXtendable Deepfake Detection', 2025,
         **{'Attention Mechanism': 'Attention inherited from LLaVA transformer; no new attention block specified.',
            'Dataset Volume': 'DFDC / DFDC Preview test subset counts: Not reported in local paper. FF++ c23 used for training.',
            'Training Compute': '1 × RTX 4090; 3 epochs; batch 4; gradient accumulation 1.',
            'Data Augmentation': 'Strong Feature Strengthening / Weak Feature Supplementing annotations; optional SRI synthetic-image training.',
            'Optimization Techniques': 'LLaVA fine-tuning with LoRA r=16, alpha=32; LR projector 2e-5 / other 2e-4; warmup .03.',
            'Data Balancing': 'Not reported. Balanced accuracy is a feature-selection measure, not oversampling.',
            'Fusion Strategy': 'MLLM reasoning strengthened by feature prompts; Specific Feature Detector supplements weak cues.',
            'Limitations / Notes': 'arXiv 2410.06126v4 dated 29 May 2025; preprint under review. Cited methods in Table 1 are not counted as new experiments. Actual evaluation volume/resolution unspecified.'})
for size,frame,video,pframe,pvideo in [('7B',.837,.860,.873,.893),('13B',.834,.857,.903,.910)]:
    for dataset,fa,va in [('DFDC (full-release benchmark)',frame,video),('DFDC Preview (DFDCP)',pframe,pvideo)]:
        for unit,a in [('Frame',fa),('Video',va)]:
            add(b,f'X²-DFD / LLaVA {size}', 'Table 1, PDF p. 7; setup p. 6',auc=a,
                **{'Dataset':dataset,'Evaluation Protocol':f'{unit}-level cross-dataset ROC-AUC; train FF++ c23. Video score aggregation details not reported in local text.'})
for name,a,other,source in [
    ('All features SFS',.778,'AP .819; EER .289','Table 3, PDF p. 8'),
    ('Strong features SFS; no WFS, no SFD',.792,'AP .821; EER .276','Tables 3 and 4, PDF p. 8 (same ablation deduplicated)'),
    ('SFD used only at inference; no WFS',.791,'Not reported','Table 4, PDF p. 8'),
    ('FF++ + SRI training',.839,'Not reported','Table 5, PDF p. 8'),
    ('Human-generated questions',.836,'Not reported','Table 6, PDF p. 9'),
    ('Claude-3.5-Sonnet-generated questions',.835,'Not reported','Table 6, PDF p. 9'),
    ('GPT-4o-generated questions',.835,'Not reported','Table 6, PDF p. 9')]:
    add(b,'X²-DFD / LLaVA 7B — '+name,source,auc=a,
        **{'Evaluation Protocol':'Cross-dataset DFDC ablation, train FF++ unless SRI stated; ablation tables do not explicitly repeat frame/video unit.', 'Other Metrics':other})

b = base('pdf_042', 'Explainable Deepfake Video Detection Using Convolutional Neural Network and CapsuleNet', 2024,
         **{'Dataset Volume':'Two DFDC chunks: 4,948 videos; ~24,740 sampled frames at 1 fps. Stated 3,958 train / 989 test; 792 validation within training.',
            'Input Resolution':'5 × 128 × 128 × 3 full-frame sequences; scale 1/255',
            'Training Compute':'Google Colab GPU, model not reported; batch 4; 30 epochs in setup versus 20 in loss discussion.',
            'Evaluation Protocol':'VALIDATION result, not held-out test accuracy. 80/20 video split then 20% training validation; source/identity grouping not reported.',
            'Other Metrics':'Validation recall .88; validation loss .2893. Table VII instead gives recall .883.',
            'Optimization Techniques':'Capsule routing by agreement; optimizer/LR not reported.',
            'Limitations / Notes':'Use Table VI full-frame validation measures. Training accuracy 99.74% is excluded. Rounded split counts sum to 4,947 rather than 4,948. Grad-CAM is post hoc, not attention.'})
add(b,'ConvLSTM2D-128 → Conv2D-256 → primary/secondary capsules (2 × 16) → LSTM-1024 → dense classifier', 'Table VI, PDF p. 10; model/setup pp. 7–9',88,auc=.951)

b = base('pdf_058','Extracting Local Information from Global Representations for Interpretable Deepfake Detection',2025,
         **{'Dataset Volume':'DFDC test volume: Not reported. Describes >128,000-video release. Training on pristine FF++ and SBI pseudo-fakes.',
            'Evaluation Protocol':'Cross-dataset DFDC ROC-AUC; train FF++. Evaluation unit and video aggregation not explicitly specified.',
            'Fusion Strategy':'Shared pretrained EfficientNet-B4 SBI encoder; local PCA subspaces + separate MLPs for eyes/nose/mouth.',
            'Data Augmentation':'Local RGB/hue/saturation/brightness/contrast; downsampling/sharpening; translation/scale; mask blending; SBI pseudo-fakes.',
            'Optimization Techniques':'Pretrained author SBI EfficientNet-B4 checkpoint; PCA local subspaces; learn per-part MLPs. Optimizer/LR/epochs not reported.',
            'Limitations / Notes':'Backbone is EfficientNet-B4, not ViT as described in the old workbook. No architectural attention reported; GradCAM++ is explanatory. Cited competing-method scores excluded. Local method shares one encoder.'})
for name,a in [('LFD-Eyes',.7129),('LFD-Nose',.7074),('LFD-Mouth',.7112)]:
    add(b,'EfficientNet-B4 (SBI) + PCA + MLP / '+name,'Table 1, PDF p. 7; methods pp. 4–6',auc=a)
for name,a in [('Sum / arithmetic mean',.7198),('Maximum score',.7144),('Minimum score',.7121),('Median score',.7208),('Geometric mean',.7182),('Harmonic mean',.7151),('Majority voting',.6033)]:
    add(b,'LESB: EfficientNet-B4 (SBI) + 3 local PCA/MLP branches', 'Table 2, PDF p. 7; sum also Table 1 (deduplicated)',auc=a,
        **{'Fusion Strategy':name+' of eye/nose/mouth MLP outputs (shared encoder).'})

b = base('pdf_059','Deepfake detection using optimized VGG16-based framework enhanced with LIME for secure digital content',2025,
         **{'Dataset':'DFDC Preview', 'Input Resolution':'256 × 256 RGB',
            'Dataset Volume':'DFDC Preview evaluation volume: Not reported. Initial training dataset: 140,000 real/StyleGAN images, 70/15/15 split.',
            'Evaluation Protocol':'Cross-dataset evaluation from Flickr-Faces-HQ + StyleGAN to DFDC Preview video frames; aggregation not reported.',
            'Training Compute':'Training hardware/time not reported. RTX 3060, 38 ms/image and ~1.2 GB memory describe inference.',
            'Other Metrics':'Precision .909; recall .912.',
            'Data Augmentation':'Random crop, ±15° rotation, 10% masking, horizontal flip, brightness/contrast; Gaussian denoising.',
            'Optimization Techniques':'ImageNet transfer learning; freeze early layers, retrain final layers; batch normalization, early stopping; exact optimizer/LR/batch not reported.',
            'Limitations / Notes':'Only Table 4 DFDC Preview score included. LIME is post hoc. Other 98–99% scores belong to synthetic-image/other datasets.'})
add(b,'VGG16 + fine-tuned classifier; post-hoc LIME','Table 4, PDF p. 8; §§3–4, pp. 3–7',91.7,.910)

b = base('pdf_074','Combating Deepfakes: A Transformer - Obtained and Explainable Frameworks for Robust Video Integrity Verification',2025,
         **{'Dataset':'DFDC Preview', 'Dataset Volume':'DFDC Preview-specific count: Not reported. Generic setup describes 6,000 videos split 4,200/1,800; attribution to each dataset unclear.',
            'Input Resolution':'112 × 112 face crops; 100 frames/video for DFDC Preview result',
            'Training Compute':'NVIDIA RTX 3090; batch, epochs and duration not reported. Generic ≥16 GB RAM is a requirement, not a measurement.',
            'Attention Mechanism':'Spatiotemporal self-attention and audio–visual cross-attention.',
            'Fusion Strategy':'ResNeXt-50 visual embeddings + unspecified audio CNN on Mel-spectrograms; transformer fusion; dense classifier.',
            'Evaluation Protocol':'DFDC Preview, 100 frames/video; train/test relationship to other corpora unspecified; aggregation not reported.',
            'Limitations / Notes':'Table II incorrectly appends % to .95 AUC; prose explicitly says .95–.96, so .95 is retained. Earlier 10–100-frame accuracy curve belongs to FF++ and is excluded. Pruning/quantization described without reproducible parameters.'})
add(b,'ResNeXt-50 + audio CNN + spatiotemporal transformer','Table II, PDF p. 7; setup pp. 4–5',94.28,auc=.95)

b = base('pdf_075','ConvNext-PNet: An interpretable and explainable deep-learning model for deepfakes detection',2024,
         **{'Dataset':'DFDC Preview (DFDC-P)', 'Dataset Volume':'Actual train/test frames/videos: Not reported. Describes 5,000-video Preview release.',
            'Input Resolution':'224 × 224 × 3 face frames',
            'Fusion Strategy':'One modified ConvNeXt encoder + prototype similarity/max pooling + weighted fully connected softmax; not a multi-model ensemble.',
            'Optimization Techniques':'ImageNet initialization; SGD; cross-entropy + clustering/separation losses; sparse final-layer optimization.',
            'Limitations / Notes':'ConvNeXt uses depthwise convolutions, not transformer attention. Exact ConvNeXt size not named; width 96 and stage ratio 1:1:3:1 described. GPU/batch/epochs and source grouping not reported. Cited comparative baselines excluded.'})
model='Modified ConvNeXt (LeakyReLU, width 96) + ProtoPNet'
add(b,model,'Table 1, PDF p. 6; method pp. 2–4',90.87,auc=.9362,**{'Evaluation Protocol':'Intra-dataset train and test on DFDC Preview; split fractions/evaluation unit not reported.'})
for train,a in [('FF++',75.42),('Celeb-DF',60.00)]:
    add(b,model,'Table 3, PDF p. 7',a,**{'Evaluation Protocol':f'Cross-corpus: train {train}, test DFDC Preview. Unit/aggregation not reported.'})

b = base('pdf_077','Deepfakes on Trial: Developing a High-Accuracy, Court-Admissible AI Pipeline for Deepfake Detection in Corporate Fraud Litigation',2025,
         **{'Dataset Volume':'118,409 acquired videos (~469 GB); split 94,726 train / 11,841 test / 11,841 validation; 72 rows dropped during feature merge. Final evaluated count not explicit.',
            'Training Compute':'NVIDIA GeForce RTX 4070 Ti SUPER (environment screenshot, Appendix A.4, PDF p. 88); Python 3.11.7, Torch 2.2.0+cu121. CPU, batch/epochs/training duration not reported.',
            'Input Resolution':'224 × 224 largest MTCNN face; five evenly spaced frames/video; audio 16 kHz',
            'Fusion Strategy':'Average 2,560-d EfficientNet-B7 face embeddings across frames; concatenate classical visual/facial/motion/audio features; classifier.',
            'Data Balancing':'SMOTE within training folds only; original distribution in evaluation.',
            'Optimization Techniques':'Stratified k-fold model selection by AUC; standard scaler fit on train only; MLP two ReLU hidden layers, dropout/L2, early stop. Exact k, sizes/LR not reported.',
            'Limitations / Notes':'Honors thesis, not peer-reviewed paper. Original-source/identity grouping not specified. Counts round to 118,408 before cleaning. Accuracy is a published claim; no legal-admissibility conclusion endorsed.'})
for model,a,f,u,sd in [('LightGBM',83.55,.4709,.8072,'accuracy ±.10 percentage points; AUC ±.0019; F1 ±.0033'),('Logistic Regression',79.83,.5444,.8576,'accuracy ±.25 pp; AUC ±.0038; F1 ±.0034'),('MLP (two hidden layers)',90.54,.7048,.9285,'accuracy ±.69 pp; AUC ±.0025; F1 ±.0125'),('Random Forest',88.35,.5520,.8649,'accuracy ±.17 pp; AUC ±.0038; F1 ±.0057')]:
    add(b,'EfficientNet-B7 + classical features → '+model,'Figure 14, PDF p. 42 (printed p. 33); methods pp. 25–42',a,f,u,
        **{'Evaluation Protocol':'Video-level stratified cross-validation on training split; mean metrics; original-source grouping not specified.', 'Other Metrics':sd})
add(b,'EfficientNet-B7 + classical features → final MLP','§4.2, PDF p. 54 (printed p. 45); pp. 43–46',91.72,.705,.94,
    **{'Evaluation Protocol':'Held-out DFDC video test split; probability threshold .5; not the CV result. F1 averaging/positive class not specified in headline.'})

b = base('pdf_078','DeepFakeShield: AI-Based Detection of Deepfake Videos and Synthetic Media',2025,
         **{'Dataset Volume':'DFDC evaluated subset: Not reported. Generic description >100,000 videos.',
            'Input Resolution':'224 × 224 aligned faces; 32-frame windows; 30 fps',
            'Training Compute':'Dual Intel Xeon processors; GPU model/memory, batch/epochs/time not reported.',
            'Fusion Strategy':'3D-CNN classification probability + transformed autoencoder reconstruction error; weighted score, weights unspecified.',
            'Evaluation Protocol':'FF++ primary training; DFDC robustness/generalization evaluation. DFDC split/unit and score aggregation not fully specified.',
            'Optimization Techniques':'Autoencoder MSE reconstruction loss; cross-dataset fine-tuning mentioned, data allocation unclear.',
            'Limitations / Notes':'3D-CNN/autoencoder layer variants not specified. Grad-CAM/saliency are post hoc. Baseline bars/qualitative ablations have no explicit exact DFDC scores and are not transcribed.'})
add(b,'3D-CNN (unspecified variant) + autoencoder','Figures 3–5 and adjacent text, PDF p. 5; setup p. 4',94.8,auc=.975)

b = base('pdf_090','ExplaNET: A Descriptive Framework for Detecting Deepfakes With Interpretable Prototypes',2024,
         **{'Dataset':'DFDC Preview (DFDC-P)', 'Dataset Volume':'Actual sampled train/test volume: Not reported. Describes >5,000-video Preview release.',
            'Input Resolution':'224 × 224 MTCNN face crops',
            'Fusion Strategy':'Single modified DenseNet-121 encoder + prototype similarity / max-pooling / weighted softmax classifier.',
            'Optimization Techniques':'ImageNet transfer; leaky-ReLU backbone; prototype clustering/separation and cross-entropy; final-layer sparsity.',
            'Limitations / Notes':'Experiment family deduplicated with DFP-Net (IJCB 2023, pdf_083 Tables 2–3 p. 6): same authors, architecture, exact intra/cross results. Journal adds F1 .927. Year from volume/DOI; acceptance date in PDF is inconsistent. GPU/batch/epochs/split fractions not reported.'})
model='ExplaNET / DFP-Net: modified DenseNet-121 + prototype layer'
add(b,model,'Tables III–IV, PDF p. 8 (printed p. 493); DFP-Net Tables 2–3, PDF p. 6',92.53,.927,.93,
    **{'Evaluation Protocol':'Intra-dataset DFDC Preview real/fake classification; evaluation unit not explicit.'})
for train,a in [('FF++',78.6),('Celeb-DF',70.04)]:
    add(b,model,'Table IV, PDF p. 8; DFP-Net Table 3, PDF p. 6',a,**{'Evaluation Protocol':f'Cross-dataset: train {train}; test DFDC Preview; evaluation unit not explicit.'})

b = base('pdf_093','Guardian AI: Synthetic Media Forensics through Multimodal Fusion and Advanced Machine Learning',2024,
         **{'Dataset Volume':'Table 2 p. 6 claims 140,000 real + 140,000 fake train; 20,000 + 20,000 test samples. Unit/source derivation unclear and inconsistent with standard DFDC video count.',
            'Fusion Strategy':'Concatenated facial, ELA/PRNU forensic and optical-flow features; MLP/LSTM/RNN ensemble described without fixed configuration.',
            'Evaluation Protocol':'DFDC accuracy explicitly attributed in §V; split, grouping and evaluation unit not specified.',
            'Optimization Techniques':'SGD with momentum and cross-validation described; actual parameters not reported.',
            'Limitations / Notes':'Low reproducibility: Facenet/ArcFace/VGGFace and MLP/LSTM/RNN are listed as alternatives, not a fixed named implemented backbone. Only explicit DFDC 95% retained. Tables 1/3 and updated 96% prose not separately keyed to DFDC, so F1/AUC left blank. Grad-CAM is not attention.'})
add(b,'Guardian AI: multimodal ensemble; exact backbone/classifier selection not reported','§V, PDF p. 5; dataset Table 2 p. 6',95)

# The large DFDF paper reports multiple internally inconsistent precision levels.
# Transcribe precise prose for main configurations and the printed table for ablations;
# do not recompute metrics from its conflicting confusion matrices.
b = base('pdf_103','Seeing Through the Fake: Explainable AI With Multiple CNNs for Deepfake Detection',2026,
         **{'Dataset Volume':'2,500 DFDC videos (1,500 fake, 1,000 real); 20 frames/video → 50,000 frames → ~27,000 faces (16,200 fake / 10,800 real); 21,600 train / 5,400 test faces.',
            'Input Resolution':'224 × 224 × 3 Haar face crops; manually reject no-face frames',
            'Training Compute':'Intel Core i7-1355U (10 cores, 12 threads); GPU not reported; batch 64, 100 epochs for deep training. Per-model inference values in Table 11 p. 28.',
            'Evaluation Protocol':'80/20 FACE/FRAME split; source-video/identity grouping not specified. Feature normalization described over all 27,000 faces, possible test leakage.',
            'Data Balancing':'60% fake / 40% real faces; no resampling reported.',
            'Data Augmentation':'Random cropping, horizontal flip, rotation, brightness.',
            'Optimization Techniques':'ImageNet/Keras pretrained CNN feature encoders + sklearn classifiers; Adam LR 1e-4 for deep training, cross-entropy; grid search; fixed encoder/learnable classifier preferred.',
            'Fusion Strategy':'Single CNN feature encoder followed by one ML classifier; 5 encoders × 5 classifiers, not an ensemble of all CNNs.',
            'Limitations / Notes':'Published metrics internally conflict: figures, prose, Table 7 and confusion counts disagree; no corrections inferred. Table 7 headers say % but entries are 0–1 fractions; convert accuracy ×100, retain F1 fraction. ResNet-50 identified by Table 11; InceptionResNetV2 by method p. 9. Grad-CAM/SHAP are post hoc.'})
clfs=['Logistic Regression','Decision Tree','Random Forest','KNN','Naive Bayes']
main={
 'DenseNet201':[(97.61,96.18,93.55,99),(94.3,92.89,93.09,92.70),(99.10,99.1,97.18,98.4),(98,98.10,98.41,97.80),(90.05,87.50,90.51,84.70)],
 'InceptionResNetV2':[(94.82,None,None,98.60),(83.17,81.52,77.99,85.40),(93.78,93.17,89.13,97.60),(94.57,93.98,90.62,97.60),(86.91,None,82.57,88.60)],
 'Xception':[(95.13,94.59,91.34,98.10),(89.91,88.40,88.40,88.40),(93.69,None,90.29,96.28),(95.48,94.00,91.40,98.80),(87.22,84.58,88.40,86.45)],
 'VGG16':[(95.43,94.84,93.2,96.4),(90.43,88.29,88.01,88.57),(95.52,95.01,92.11,98.1),(95.22,94.66,91.90,97.60),(79.48,74.03,82.27,67.30)],
 'ResNet-50':[(86,82.24,88.18,78.30),(90.09,88.99,88.68,89.30),(92.52,92.54,92.19,92.90),(93.30,92.30,92.05,92.60),(69.70,55.91,76.08,44.20)]}
without={
 'DenseNet201':[(.97,.95,.93,.98),(.94,.92,.92,.92),(.98,.97,.96,.98),(.97,.97,.98,.97),(.89,.87,.90,.84)],
 'InceptionResNetV2':[(.94,.94,.90,.98),(.82,.81,.77,.85),(.93,.92,.88,.97),(.94,.93,.90,.97),(.86,.85,.82,.88)],
 'Xception':[(.94,.94,.91,.97),(.89,.88,.88,.88),(.93,.92,.89,.95),(.95,.94,.91,.98),(.86,.87,.88,.86)],
 'VGG16':[(.95,.94,.92,.96),(.90,.88,.87,.88),(.95,.94,.91,.97),(.95,.94,.91,.97),(.79,.73,.81,.66)],
 'ResNet-50':[(.85,.82,.87,.78),(.89,.88,.88,.89),(.92,.92,.91,.92),(.93,.92,.91,.92),(.69,.55,.75,.43)]}
inference={
 'DenseNet201':[(142,240),(143,250),(150,390),(165,530),(140.5,250)],
 'InceptionResNetV2':[(182,610),(183,650),(190,810),(205,910),(180.5,650)],
 'Xception':[(132,310),(133,326.8),(140,486),(155,639),(130.5,365.5)],
 'VGG16':[(152,1472.05),(153,1370),(160,1980.8),(175,1876),(150.5,1783.6)],
 'ResNet-50':[(137,310),(138,305),(145,480),(160,610),(135.5,316.8)]}
for model,vals in main.items():
    page={'DenseNet201':'15','InceptionResNetV2':'16','Xception':'17','VGG16':'17–18','ResNet-50':'18'}[model]
    for j,(acc,f,p,r) in enumerate(vals):
        ms,mem=inference[model][j]
        table_f1={('InceptionResNetV2',0):.94,('InceptionResNetV2',4):.85,('Xception',2):.93}
        reported_f1=table_f1[(model,j)] if f is None else f/100
        add(b,model+' → '+clfs[j]+' (with preprocessing)',f'§IV.C / Figures 6–10, PDF pp. {page}; Table 7 p. 25; Table 11 p. 28',acc,reported_f1,
            **{'Other Metrics':f'Precision {p/100 if p is not None else "Table 7: .90 (coarse)"}; recall {r/100}; inference {ms} ms, {mem} MB (Table 11; unit per input not explicit). F1 from '+('Table 7 (prose absent).' if f is None else 'prose; Table 7 may differ.')})
        wa,wf,wp,wr=without[model][j]
        add(b,model+' → '+clfs[j]+' (without preprocessing)','Table 7, PDF p. 25 (printed p. 155)',wa*100,wf,
            **{'Other Metrics':f'Precision {wp}; recall {wr}. Coarse printed table precision retained; exact prose may differ.'})
for mode,model,clf,acc in [('LL','ResNet-50','Random Forest',98),('LL','DenseNet201','Naive Bayes',68),('LF','ResNet-50','Random Forest',77),('LF','DenseNet201','Naive Bayes',57)]:
    add(b,model+' → '+clf+' / '+mode,'Table 3, PDF p. 13',acc,
        **{'Evaluation Protocol':b['Evaluation Protocol']+' Learning ablation: '+('learnable encoder and classifier' if mode=='LL' else 'learnable encoder, fixed classifier')+'.'})
for eps,a in [(.01,97),(.05,95),(.1,91)]:
    add(b,f'DenseNet201 → Random Forest / FGSM ε={eps}','Table 5, PDF p. 21',a,
        **{'Evaluation Protocol':b['Evaluation Protocol']+f' Adversarial FGSM evaluation at ε={eps}; perturbation domain not fully specified.'})

b = base('pdf_104','Semantic Forensics with Explainable AI for Multimodal Deepfake and Media Manipulation Detection',2025,
         **{'Dataset Volume':'Not reported', 'Evaluation Protocol':'DFDC dataset-specific ROC-AUC in Table 3; train/test split, frame/video unit, source grouping and aggregation not reported.',
            'Fusion Strategy':'Artifact logits + semantic risks via logistic meta-classifier; temperature calibration; FaceNet identity, audio/physiology signals; GPT-4V reasoning.',
            'Limitations / Notes':'Exact artifact CNN configuration and LMM version/prompts not fixed: paper lists XceptionNet/ConvNeXt and GPT-4V/LLaVA. Training compute, sampling, image dimensions and optimization absent. Explainability/provenance modules are not evidence of architectural attention.'})
for model,a in [('Baseline CNN (backbone not specified)',.893),('ViT-B/16',.906),('Semantic-forensics framework: artifact CNN/ViT + FaceNet + GPT-4V + logistic fusion',.927)]:
    add(b,model,'Table 3 and §IV.A, PDF p. 4',auc=a,
        **{'Attention Mechanism':'Transformer self-attention (ViT-B/16 comparator)' if model=='ViT-B/16' else 'None reported' if model.startswith('Baseline') else 'Transformer-based vision/LMM components described; specific attention/fusion block not specified.'})

EXCLUDED = {
 'pdf_004':'PDF p. 6 explicitly calls numerical results a simulation / illustrative values; not measured DFDC evidence.',
 'pdf_010':'X-FACTS Table 2 p. 13 not isolated by dataset; method switches between DFDC/WildDF/FF++; simulation setup. No unambiguous DFDC-specific result.',
 'pdf_032':'FF4ALL overview; DFDC/CelebDF/WildDF pooled/cited project results, no isolated new DFDC metric.',
 'pdf_034':'DFDC mentioned in dataset/background discussion; reported experiment not unambiguously isolated to DFDC.',
 'pdf_045':'DFDC is pretrained detector provenance; attack experiments do not isolate DFDC evaluation.',
 'pdf_055':'ExDDV is a new mixed annotation benchmark containing DFDC-derived videos; no isolated DFDC result.',
 'pdf_063':'UK Home Office 2024 Deepfake Detection Challenge, not Meta/Kaggle DFDC; DFDC otherwise cited/background.',
 'pdf_065':'Generic deepfake detection challenges rather than an identified DFDC dataset evaluation.',
 'pdf_067':'Cross-dataset F1 .89 pooled/ambiguous between Celeb-DF and DFDC Preview; exact DFDC-specific metric absent (p. 5).',
 'pdf_071':'Custom mixed FF++ + DFDC benchmark; 91.92% / .931 AUC not isolated to DFDC (pp. 3–4).',
 'pdf_083':'Same experiment family as ExplaNET pdf_090; identical DFDC Preview intra/cross scores consolidated into journal rows. Original conference pp. 5–6 retained as supporting source.',
 'pdf_088':'DFDC explicitly future work; no completed DFDC result.',
 'pdf_091':'Case-study summary of 2024–25 research (p. 4), no dataset-keyed independent DFDC result.',
 'pdf_109':'Earlier SSRN version of literature Paper 35 / pdf_041; identical four DFDC variant results deduplicated.',
 'pdf_111':'DFDC named as complementary testbed (p. 3) but no DFDC-keyed numeric result; principal results concern FF++.',
 'pdf_112':'DFDC in related-work comparative table, no independently isolated DFDC experiment.',
}
survey_keys={'pdf_001','pdf_002','pdf_011','pdf_013','pdf_016','pdf_021','pdf_022','pdf_029','pdf_039','pdf_052','pdf_053','pdf_070','pdf_084','pdf_098','pdf_106','pdf_107','pdf_108'}
included={r['_key'] for r in rows}
audit=[]
for x in inventory:
    key=x['key']
    if key in included:
        decision='Included'; reason='Original dataset-specific experiment; see result rows and methodological caveats.'
    elif key in EXCLUDED:
        decision='Consolidated duplicate' if key in {'pdf_083','pdf_109'} else 'Excluded';reason=EXCLUDED[key]
    elif key in survey_keys:
        decision='Excluded';reason='Survey, conceptual/legal discussion, or investigation framework; no independent dataset-specific DFDC result.'
    else:
        decision='Excluded';reason='No DFDC evaluation found in extracted full text; '+('DFDC occurrences are related work / references or other-dataset context.' if x['dfdc_pages'] else 'No DFDC name/acronym occurrence.')
    audit.append({'PDF Key':key,'Paper IDs':', '.join(map(str,x['literature_ids'])) or 'Not indexed',
                  'Title / Filename':x['matched_title'] if x.get('match_score',0)>=.82 else x['filename'],
                  'Decision':decision,'Reason':reason,'DFDC mention pages':', '.join(map(str,x['dfdc_pages'])) or 'None',
                  'Source':x['source'],'SHA256':x['sha256'],
                  'Duplicate physical copies':len(x.get('duplicate_locations',[]))})

payload={'headers':HEADERS,'rows':rows,'audit':audit,'conventions':[
    'Accuracy uses percentage points (0–100); F1 and ROC-AUC use fractions (0–1). Blank numeric cells mean not reported or not unambiguously attributable.',
    'Not reported is used for missing text. None reported means no architectural attention identified. Grad-CAM/LIME/SHAP are post-hoc explanations.',
    'Paper ID is the original Literature_Analysis.xlsx ID. Not indexed means a newly found local original has no ID in that workbook; no synthetic IDs substituted.',
    'DFDC Preview is a different release. Frame/face, video, validation and cross-dataset scores are not interchangeable; do not rank them as matched experiments.',
    'Source pages are one-based PDF pages. Source Audit records every distinct PDF, including exclusions; old Papers.zip copies are hash-deduplicated.',
    'These are verified transcriptions of author-reported claims, not a verification of execution, data integrity or reproducibility. Caveats remain explicit.',
    'Published row count measures model/protocol variants, not independent papers. ExplaNET/DFP-Net and hybrid SSRN/journal versions are consolidated.',
    'Local DFDC Baseline contains only completed real-data local runs; synthetic tests and the pre-existing 97.115% image result are excluded.'
]}
out=ROOT/'../research/analyses/dfdc_literature.json'
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'experiment_families':len(included),'unique_pdfs':len(audit),'included':sorted(included)},indent=2))
