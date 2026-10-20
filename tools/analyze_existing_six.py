"""Curated full-PDF analysis of exactly six existing literature records.

This script reads existing evidence, creates JSON/Markdown, and never edits XLSX.
Scientific judgments are explicitly distinguished from reported measurements.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / '../research/working/existing_six'
originals = {r['paper_id']: r for r in json.loads((WORK/'complete_originals.json').read_text(encoding='utf-8'))}
master = {r['Paper No']: r for r in json.loads((PROJECT/'../research/working/expansion/original_master.json').read_text(encoding='utf-8'))}
SOURCE = PROJECT.parent/'Literature_Analysis.xlsx'
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper()
papers = []


def paper(pid, role, dfdc, architecture, xai, datasets, finding, diagram):
    r = originals[pid]
    assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest() == r['sha256']
    p = {k:r[k] for k in ['paper_id','key','source','sha256','pages']}
    p.update(title=master[pid]['Title'],role=role,independent_dfdc=dfdc,
             architecture=architecture,xai=xai,datasets=datasets,finding=finding,
             diagram=diagram,notes=[],performance=[])
    papers.append(p)
    return p


def note(p, section, item, text, ref, status='Reported', value=None, unit=''):
    p['notes'].append(dict(section=section,item=item,text=text,reference=ref,status=status,value=value,unit=unit))


def result(p, dataset, protocol, model, ref, accuracy=None, precision=None, recall=None,
           f1=None, auc=None, other='', status='Reported', dfdc=False, proposed=False):
    p['performance'].append(dict(dataset=dataset,protocol=protocol,model=model,reference=ref,
        accuracy=accuracy,precision=precision,recall=recall,f1=f1,auc=auc,other=other,
        status=status,dfdc=dfdc,proposed=proposed))


p=paper(2,'DFDC experimental baseline','Yes - reported original experiment',
    'Xception + attention-augmented Xception + attention-augmented EfficientNet-B3; decision-level ensemble.',
    'Trainable attention pooling and crop/drop augmentation; post-hoc Grad-CAM; human DFT/DCT inspection.',
    'Full DFDC; Celeb-DF; WildDeepfake',
    'Use DFDC Table 2: 92.20% accuracy / 0.9752 AUC. The 93.64% result is Celeb-DF.',
    'Video -> sample 10% of frames -> RetinaFace -> align/crop 320x320\n'
    '                   +-> Xception ---------------------+\n'
    '                   +-> Xception + attention pooling --+-> weighted decisions\n'
    '                   +-> EfficientNet-B3 + attention ---+   -> average frames -> verdict\n'
    'Training: attention map -> crop/zoom OR drop -> same-label augmented input\n'
    'Interpretation: Grad-CAM -> thresholded ROI -> DFT / 8x8 DCT -> human review')
note(p,'Architecture','Backbones and branches','Three independent classifier branches: ordinary Xception, Xception with weakly supervised attention, and EfficientNet-B3 with weakly supervised attention. EfficientNet is specifically B3; this is not a three-backbone feature-concatenation network. No recurrent temporal backbone is specified.','pp. 7-8, Sections 3.2-3.3')
note(p,'Architecture','Feature extraction','Xception uses depthwise separable convolutions and residual connections. B3 uses compound depth/width/resolution scaling. Each produces feature maps F with spatial H x W and N channels. The paper describes randomly initialized convolution filters; it does not establish a pretrained checkpoint for the actual experiments.','pp. 7-8, Sections 3.2.1-3.3.1')
note(p,'Architecture','Trainable attention','A convolution / batch-normalization / nonlinear block maps F to M attention maps A. M is a hyperparameter with no numerical value supplied. Attention is learned using real/fake supervision rather than facial-part segmentation labels.','p. 8, Eq. 1, Section 3.3.1')
note(p,'Architecture','Attention feature pooling','For each attention map, elementwise multiplication with feature maps followed by global average pooling produces a facial-segment feature. The M segment features form the facial segment matrix used for classification. The paper describes this as bilinear attention pooling.','pp. 7-8, Fig. 5, Section 3.3.2')
note(p,'Architecture','Classifier and ensemble','The branches produce real/fake decisions. Their probabilities are combined with a certainty-related weighted average; exact numerical ensemble weights, calibration rule and classifier-head dimensions are not supplied. Frame probabilities are averaged for an overall video decision.','pp. 8-9, Sections 3.2.3 and 4.1')
note(p,'Explainability','Grad-CAM implementation','Backpropagate the class decision into feature and attention maps, gradient-weight and combine maps, apply ReLU, resize and overlay. Unlike the trainable attention block, this attribution is post-hoc. The account alternates between loss gradients and prediction gradients without a precise implementation specification.','p. 9, Section 3.5, Fig. 7; p. 10, Fig. 8')
note(p,'Explainability','Forensic workflow','Threshold the explanatory heatmap, crop the influential region, inspect its 2-D Fourier spectrum, and compute DCT coefficients in nonoverlapping 8x8 blocks. The 63 AC coefficients are examined through Laplace-scale statistics beta. These outputs support human forensic reasoning; no additional learned forensic classifier or measured final human-assisted accuracy is reported.','pp. 11-12, Section 4.3, Figs. 10-11')
note(p,'Explainability','Effect on prediction','Attention pooling and attention-guided augmentation affect training/classification. Grad-CAM, DFT and DCT inspection explain or review a decision. The paper does not demonstrate a quantitative classification improvement caused by the post-hoc forensic stage.','pp. 8-12, Sections 3.3-4.3','Interpretation')
note(p,'Dataset','DFDC release and corpus size','The full DFDC corpus is used, not DFDC Preview. Table 1 lists 128,154 videos and 104,500 unique fake videos; these are corpus descriptions, not verified counts actually sampled for training. The held-out DFDC evaluation is described as 5,000 ten-second clips with distractors/noise.','p. 5, Table 1 and Section 3.1; p. 9, Section 4.1')
note(p,'Dataset','Sampling and preprocessing','Sample 10% of each video\'s constituent frames, detect faces with RetinaFace, crop around landmarks and use an affine transform to align. Aligned crops are 320x320; face mean/variance normalization is described. The exact temporal index policy and handling of multiple faces/missed detections are not specified.','pp. 6-7, Section 3.1, Fig. 4')
note(p,'Dataset','Split and class distribution','Actual training-frame total, real/fake sampled counts, validation size, split manifest, identity/source grouping and seeds are Not Reported. A frame-based evaluation statement appears on pp. 5-6, while p. 9 says probabilities are averaged over frames; the precise unit underlying the tabulated metrics is not fully resolved.','pp. 5-10; complete PDF pp. 1-14','Unknown')
note(p,'Dataset','Out-of-distribution testing','Celeb-DF is evaluated without being used for training (Table 3); WildDeepfake is an internet-sourced external test (Table 4). Exact evaluated sample volumes and release/version details for those external evaluations are not enumerated.','pp. 10-11, Tables 3-4')
note(p,'Optimization','Random augmentation','Training randomly applies horizontal flips, Gaussian/ISO noise, motion/Gaussian blur, hue-saturation shifts, brightness/contrast changes, sharpening, embossing and sepia. Application probabilities and parameter ranges are not supplied.','p. 6, Section 3.1')
note(p,'Optimization','Attention cropping and dropping','Cropping thresholds an attention map, finds the smallest bounding box and resizes the selected area to the input size with the original label. Dropping zeros high-attention areas to force other facial cues. Threshold values and scheduling are not reported.','pp. 8-9, Section 3.4, Fig. 6')
note(p,'Optimization','Center loss','Center loss sums squared distances between each segment feature f_k and its center c_k. Centers use a minibatch moving average c_k <- c_k + beta(f_k-c_k). This encourages corresponding attention maps to capture consistent segments. Beta, loss weight and the full combined classification objective are not specified.','p. 8, Eq. 2 and Section 3.3.2')
note(p,'Compute','Published resources','Training GPU model/count, epochs, batch size, training duration, total parameter count, optimizer, learning rate/scheduler, weight decay, dropout rate and mixed precision are Not Reported. No speed claim is converted into a resource benchmark.','complete PDF pp. 1-14','Unknown')
note(p,'Limitations','Numerical inconsistencies','The abstract says 92.4% DFDC accuracy, but Table 2 says 92.20%. Table 2 ensemble TP+TN percentages equal 92.00%, not 92.20%, and all four confusion percentages sum to 99.78%. Retain the reported table accuracy/AUC and flag the discrepancy. Do not derive authoritative F1/precision/recall from these rounded, inconsistent percentages.','p. 1, abstract; p. 10, Table 2','Calculated')
note(p,'Limitations','Generalization boundary','The ensemble decreases from 92.20% DFDC accuracy to 72.73% on WildDeepfake. These are different test distributions and volumes are missing. Grad-CAM examples and spectral plots are mainly qualitative, with no controlled attribution-faithfulness or forensic user-study score.','pp. 10-12, Tables 2-4, Figs. 8-11','Interpretation')
note(p,'Future relevance','Existing mechanisms to retain','Study weakly supervised attention pooling, crop/drop augmentation, decision-level ensembling, frame-probability aggregation and explanation-guided forensic inspection as separate documented mechanisms. Their contribution is not a validated recipe for an untested hybrid.','pp. 6-12, Sections 3-4','Interpretation')
for dataset,protocol,table,rows in [
 ('DFDC full','DFDC 5,000-clip test; aggregation/unit qualification above','p. 10, Table 2',[(.8388,.9107),(.9144,.9724),(.8023,.9160),(.9220,.9752)]),
 ('Celeb-DF','External test, trained on DFDC','p. 10, Table 3',[(.8937,.9668),(.9274,.9784),(.7920,.8808),(.9364,.9841)]),
 ('WildDeepfake','External test, trained on DFDC','p. 11, Table 4',[(.6808,.7800),(.7133,.7823),(.5455,.7526),(.7273,.8079)])]:
    for i,(model,(acc,auc)) in enumerate(zip(['Xception','Xception + Attention','EfficientNet-B3 + Attention','Three-branch ensemble'],rows)):
        result(p,dataset,protocol,model,table,accuracy=acc,auc=auc,dfdc=dataset=='DFDC full',proposed=i==3,
               other='Precision/recall/F1 not reported. Confusion percentages must not be relabeled as sensitivity/specificity.')

p=paper(9,'Methodological reference - FF++','No - FF++ experiments only',
    'ImageNet-pretrained HRNet -> DPNet latent patches -> dynamic prototypes -> maximum similarities -> linear class layer.',
    'Intrinsic prototypes and editable model; post-hoc PRP maps; UMAP and landmark-density views.',
    'FF++ HQ (c23), four manipulation types',
    '91.5% to 91.6% is one 30-prototype refinement example on FF++, not DFDC.',
    'Video -> MTCNN face crop every 10 frames\n'
    '      -> RGB frame + optical flow from next 9 frames (DeepFlow)\n'
    '      -> ImageNet-pretrained HRNet -> 16x16x256 latent representation\n'
    '      -> similarity to class prototypes -> spatial maximum per prototype\n'
    '      -> fully connected class layer -> pristine/manipulated scores\n'
    'Expert views PRP / prototype crops / UMAP / landmarks\n'
    '      -> delete or replace prototype -> cached last-layer refit -> re-evaluate')
note(p,'Architecture','Encoder and temporal input','DPNet is the detector underlying the Python/Qt visual analytics system. A face-crop RGB frame is followed by optical-flow fields from nine consecutive frames. ImageNet-pretrained HRNet encodes each input to 16x16x256. This uses motion information, not an LSTM or a transformer.','pp. 3-4, Network architecture; p. 9, System Implementation')
note(p,'Architecture','Prototype layer','Default training uses 20 pristine and 20 manipulated prototypes, each 1x1x256, representing latent training-video patches. For each prototype compute similarity at every spatial position and retain its maximum. The exact similarity formula is delegated to the cited underlying method rather than restated.','p. 4, Network architecture; p. 9, System Implementation')
note(p,'Architecture','Classification and aggregation','A fully connected class layer combines prototype maximum similarities into pristine/manipulated fragment scores. The interface shows frame/fragment contributions and allows temporal selection and averaged predictions. A fixed global whole-video aggregation specification is not supplied.','pp. 4, 7-8 and 10')
note(p,'Architecture','Attention status','No trainable attention block is specified in this architecture. Similarity maps, PRP relevance maps, prototype visualizations and UMAP are not transformer/self-attention layers.','pp. 3-9','Interpretation')
note(p,'Explainability','Intrinsic explanations','Prototypes are part of the classifier, not a separately fitted surrogate. Matching a patch to real training exemplars and exposing its class-layer contribution explains the decision. Prototypes can duplicate one another, focus on background or cover different facial regions for different classes.','pp. 3-4, 6-8 and 10-12')
note(p,'Explainability','PRP implementation','Final version replaces coarse interpolated similarity maps with Prototypical Relevance Propagation, an extension of layer-wise relevance propagation. It propagates the maximum prototype similarity back to input features. PRP changes the displayed explanation; it does not by itself retrain the classifier.','pp. 6-7, Visualizations; p. 12, system improvements')
note(p,'Explainability','Human refinement affects model','Experts delete or replace latent prototypes and then optimize the final class layer while keeping the encoder intact. Deletion uses cached maximum-similarity scores; replacement uses cached encoder outputs to recompute similarity. This changes future class scores, unlike merely viewing a PRP heatmap.','p. 4, Efficient retraining; pp. 8 and 12')
note(p,'Explainability','Visual and user evaluation','Prototype crops/GIFs, flow fields, UMAP, class contribution plots, a radar plot, confusion matrix/ROC and landmark-density map support analysis. Five forensic experts with over nine years of experience participate in two rounds; first sessions take 90-110 minutes with think-aloud tasks/interviews. This supports usability, not a quantified courtroom-validity guarantee.','pp. 6-12, Evaluation','Reported')
note(p,'Dataset','Actual corpus','Training uses FF++ HQ c=23 and its four original manipulation methods: Deepfakes, Neural Textures, FaceSwap and Face2Face. DFDC occurs as competition/background context and a reference, not as a training/test experiment.','p. 1, Introduction; p. 9, System Implementation; p. 12, Discussion')
note(p,'Dataset','Preprocessing and sampling','Use MTCNN face bounding boxes, sample every k=10 frames, compute k-1 consecutive flow frames with DeepFlow, and export GIFs for inspection. Actual image resolution, extracted sequence count, train/validation/test sample volumes and class distribution are not supplied in this paper.','pp. 8-9, Fig. 4 and System Implementation')
note(p,'Dataset','Evaluation split boundary','The interface evaluates a test set described as independent of detector training. Exact split membership and an additional untouched final holdout after expert-guided refinement are not stated. Repeatedly selecting prototypes using displayed test performance can adapt to that set; this is a methodological risk, not proof that the initial train/test split leaked.','pp. 7-9, performance panel','Interpretation')
note(p,'Optimization','Loss terms and actual weights','Training describes cross-entropy, cluster, separation and diversity losses. Actual weights: lambda_cluster=0.2, lambda_separation=0, lambda_diversity=0.1, lambda_L1=0.001. Separation is described conceptually but is disabled by its zero weight in this implementation.','p. 4, Training the network; p. 9, System Implementation')
note(p,'Optimization','Projection and last-layer optimization','Randomly initialize prototype vectors. After five regular training steps, project each prototype to the nearest latent patch from the training set. After projection, freeze other layers and optimize the class layer with cross-entropy plus L1 to suppress connections to the opposite class. The paper says five steps, not five total epochs.','p. 4, Training the network')
note(p,'Optimization','Missing training details','GPU model/count, optimizer, learning rate/scheduler, total epochs, batch size, full training time, HRNet variant, total parameter count, augmentation beyond sampling/cropping, and class-balancing strategy are Not Reported. Do not fill these from the separate DPNet paper.','complete PDF pp. 1-15','Unknown')
note(p,'Performance','Refinement example configuration','Figure 6 uses 30 prototypes before and after three replacements, whereas the described default is 40. Its accuracy 0.915 -> 0.916 is one illustrative expert refinement, not a comprehensive default-configuration benchmark. No numeric F1 or AUC for this example is reported.','p. 11, Fig. 6; p. 9, System Implementation')
note(p,'Compute','Reported interactive speed','Initial testing/refinement took 60-120 seconds. Cached final version reports combined retraining/testing under one second for deletion and a few seconds for replacement. The hardware and cache size are unspecified, so these timings do not predict RTX 4060 performance.','pp. 11-12')
note(p,'Compute','Prototype parameters','40 prototypes x 256 components = 10,240 learnable prototype scalar values in the default configuration. This excludes HRNet and the class layer; it is not the total network parameter count.','pp. 4 and 9; arithmetic 40 x 256','Calculated',10240,'prototype scalars')
note(p,'Compute','Cache size calculation','16 x 16 x 256 FP32 latent values occupy 262,144 bytes (0.25 MiB) per sequence, before overhead. For an illustrative 100,000 sequences, raw cached embeddings alone require about 24.41 GiB. Forty cached maxima need only 160 bytes/sequence, about 15.26 MiB at that hypothetical count. Those are workload calculations, not published dataset sizes.','p. 4, latent dimensions; analyst FP32/count assumptions','Calculated')
note(p,'Limitations','Interpretability and generalization','Experts report duplicated prototypes, incomplete facial coverage, race/gender representation concerns and difficulty interpreting motion matches. Only five experts and an FF++ setup are studied. Similarity of spatiotemporal patches does not guarantee semantically matched facial evidence.','pp. 10-13')
note(p,'Future relevance','Existing mechanisms to retain','Prototype/class contribution decomposition, PRP, human refinement, projection onto training exemplars, cached last-layer fitting and temporal inspection are documented tools. Their value is interpretable model inspection; this paper supplies no DFDC score for later baseline ranking.','pp. 3-12','Interpretation')
result(p,'FF++ HQ c23','One expert example; 30-prototype configuration','Before prototype replacement','p. 11, Fig. 6',accuracy=.915)
result(p,'FF++ HQ c23','Same example, after three prototype replacements','After prototype replacement','p. 11, Fig. 6',accuracy=.916,proposed=True)

p=paper(15,'Methodological reference - review','No - reviews DFDC',
    'Reviews CNN spatial extraction, RNN/LSTM temporal analysis, CNN-RNN hybrids and audio-visual models; no implemented architecture.',
    'Discusses Grad-CAM, saliency, integrated gradients, LIME and attention; no tested XAI implementation.',
    'Discusses FF++, Celeb-DF v2, DFDC and WildDeepfake',
    'Table I accuracy ranges are generic review comparisons with no matched dataset/protocol.',
    'Conceptual family reviewed, not an implemented model:\n'
    'Video frames -> spatial CNN features -> temporal RNN/LSTM -> class decision\n'
    '                          |\n'
    '                          +-> discussed post-hoc attribution / visualization\n'
    'Other reviewed families include audio-visual processing and transformers.')
note(p,'Architecture','Study type and contribution','This is a narrative review. It surveys spatial CNNs, temporal RNN/LSTMs, hybrids, transformers and multimodal approaches; there is no author-trained detector, fixed tensor flow, selected best backbone or new experimental system to reconstruct.','pp. 1-5')
note(p,'Architecture','Spatial models reviewed','VGG19, ResNet50 and Xception are discussed as spatial encoders. Xception depthwise separable convolutions are explained at family level. The review does not specify layers, feature dimensionality or a pretrained checkpoint used in an experiment.','p. 2, Section III-A')
note(p,'Architecture','Temporal and fusion concepts','CNN-LSTM / CNN-BiLSTM patterns combine per-frame spatial features with temporal recurrence. A reviewed audio-visual framework uses intra/intermodality mixer layers. These descriptions are conceptual accounts of other work, not architectures implemented by the review authors.','pp. 2-3, Sections III-B-D')
note(p,'Architecture','Attention status','Attention and graph-based integration are mentioned as future avenues; attention layers are also discussed as a transparency concept. The paper implements no self-attention, attention pooling or fusion block whose behavior can be tested.','pp. 3-4, Sections III-D and IV-B')
note(p,'Explainability','Methods actually named','The text names integrated gradients, Grad-CAM, saliency maps, LIME, rule-based explanations, trees/surrogates and attention. SHAP is not specified in the complete supplied text despite the older workbook classification. These are discussed, not implemented with recorded hyperparameters.','pp. 3-4; complete PDF pp. 1-7')
note(p,'Explainability','Effect on prediction','Grad-CAM/saliency/gradients and LIME are presented as ways to interpret a detector. Attention is an architectural concept that can affect a model when implemented. The review offers no experiment separating predictive gains from explanatory usefulness or evaluating attribution faithfulness.','pp. 3-4','Interpretation')
note(p,'Dataset','DFDC involvement','DFDC appears as a benchmark containing over 100,000 videos and in discussion of previous RNN studies. There is no downloaded subset, independent training/test split, sample count, real/fake distribution, preprocessing run or frame-sampling schedule used by these authors.','p. 2, Section III-B; p. 5, Section V-B')
note(p,'Performance','Review ranges','Table I gives accuracy ranges: CNN 90-95%, RNN 85-92%, hybrid CNN-RNN 92-97%, transformer 93-98%, handcrafted features 80-88%, GAN-based 88-94%. It does not assign each range to an exact dataset, split or matched experiment. These are review statements, excluded from DFDC Verified.','p. 3, Table I','Reported (review)')
note(p,'Performance','Own metrics','Independent accuracy, precision, recall, F1, AUC, localization, faithfulness, adversarial-robustness and efficiency measurements are not reported. Definitions of metrics do not constitute measured values.','pp. 4-5; complete PDF pp. 1-7','Unknown')
note(p,'Compute','Published resources','No executable proposed model, GPU, epochs, batch, training time, parameter count or measured latency. A reproduction VRAM/time estimate for this review is not meaningful without choosing a separate defined model, which is outside the six-paper analysis.','complete PDF pp. 1-7','Unknown')
note(p,'Optimization','Actual use versus discussion','No optimizer, loss, scheduler, regularization, augmentation or class-balancing procedure was actually run. Adversarial robustness, multimodal data and cross-dataset generalization are discussed as needs, not demonstrated improvements.','pp. 2-5')
note(p,'Limitations','Evidence traceability','Table I does not supply experiment-level provenance for its ranges. Text citations [19] and [20] for LIME/attention on p. 4 exceed the reference list ending at [18] on p. 7. This limits verification of those particular claims; it does not establish misconduct.','pp. 3-4 and 6-7','Interpretation')
note(p,'Future relevance','Use as conceptual taxonomy','Useful for distinguishing spatial, temporal and multimodal cues and for stating why explanation matters. Treat discussions of mechanisms as a reading taxonomy rather than as evidence that a particular hybrid improves DFDC accuracy. No untested combination is proposed here.','pp. 2-5','Interpretation')

p=paper(16,'Methodological reference - review','No - reviews DFDC',
    'Reviews CNNs, RNN/LSTMs, ResNet-BiLSTM concepts, ViT/TimeSformer and audio-visual systems; no trained proposed model.',
    'Architectural attention is discussed; explainability is future work; no implemented post-hoc XAI experiment.',
    'Discusses FF++, DFDC, Celeb-DF and Google DFD',
    'Table I is a review comparison explicitly on FF++; 96.4% is not an original DFDC result.',
    'Conceptual families reviewed, not an implementation:\n'
    'Frames -> CNN spatial encoding -> RNN/BiLSTM temporal analysis -> decision\n'
    'or tokens/frames -> transformer self-attention -> decision\n'
    'or audio + video encoders -> cross-modal comparison -> decision\n'
    'No single trained network or complete experimental data flow is defined.')
note(p,'Architecture','Study type','Narrative review of AI-driven deepfake detection. The article discusses more than 75 publications and supplies a survey-style comparison; it does not train or evaluate a new detector.','pp. 1-5')
note(p,'Architecture','Reviewed backbone families','Xception, EfficientNet and ResNet spatial encoders, recurrent/LSTM temporal analysis, a ResNet-BiLSTM pattern, ViT/TimeSformer global attention and audio-visual synchronization are described. These are separate reviewed families, not layers of an author-built combined architecture.','pp. 2-3, Section III')
note(p,'Architecture','Missing implementable specification','No fixed backbone variant, tensor dimensions, token count, temporal sequence length, modality fusion rule, classifier head, weights or code for an original trained model are supplied. “Hybrid CNN-RNN” is a family label, not a reproducible full model description.','pp. 2-3; complete PDF pp. 1-7','Unknown')
note(p,'Explainability','Attention and XAI distinction','Self-attention is described as a model mechanism in transformers. The article does not implement Grad-CAM, SHAP, LIME, prototype explanation or a measured XAI evaluation. Explainability and interpretability are emphasized as future requirements.','pp. 2-3, Section III-D; p. 5, future directions')
note(p,'Dataset','Actual use','DFDC is described as a corpus of over 100,000 clips, alongside FF++, Celeb-DF and Google DFD. These are benchmark descriptions. There are no author-selected training/test videos, independent class counts, frames or preprocessing operations.','p. 3, Section IV-A')
note(p,'Performance','Table I attribution','The table is explicitly titled accuracy of selected models on FaceForensics++: XceptionNet 95.2%, Capsule-Forensics 93.5%, RNN approach 89-91%, hybrid CNN-RNN 96.4%. These are review-reported figures without a matched split/compression setting, not independent results of this article and not DFDC measurements.','p. 3, Table I; p. 4, Fig. 1','Reported (review)')
note(p,'Performance','Own metrics','No independent accuracy, F1, AUC, precision/recall, cross-dataset result or attack measurement. Metric definitions and research/publication distribution plots are not evidence of a trained detector experiment.','pp. 3-4','Unknown')
note(p,'Compute','Published resources','No experiment-specific GPU, epochs, batch size, training duration, parameter count or VRAM. Hardware feasibility for an undefined reviewed family is not quantifiable as a paper reproduction.','complete PDF pp. 1-7','Unknown')
note(p,'Optimization','Discussion only','Adversarial training, self-supervised learning, domain adaptation and ensembles are reviewed as approaches. None is specified with an executed optimization schedule, objective or balancing protocol by these authors.','p. 3, Section III-F')
note(p,'Limitations','Traceability within the PDF','Several supporting references concern unrelated fields: for example, reference [70] is a skincare recommendation system although cited in a spatial deepfake paragraph, and [84] is brain-tumor imaging. The review gives no systematic search/inclusion protocol or source-specific conditions for Table I. Treat numerical comparisons cautiously; no external publication-quality accusation is inferred.','pp. 2-3 and 6-7','Interpretation')
note(p,'Limitations','Generalization claims','The text discusses unseen manipulations, compression, adversarial robustness and resource limits, but provides no direct experiment showing their resolution. Statements that multimodal fusion improves robustness remain survey claims.','pp. 3-5','Interpretation')
note(p,'Future relevance','Use as methodological context','Use its spatial/temporal/global/audio-visual taxonomy and its discussion of cross-dataset evaluation as context. It supplies no verified DFDC baseline or evidence selecting an exact architecture for the RTX 4060.','pp. 2-5','Interpretation')

p=paper(35,'DFDC experimental report; protocol incomplete','Yes - reported original experiment',
    'ResNet50 + unspecified ViT + PRNU/JPEG/frequency descriptors -> concatenation -> trainable FC/dropout -> multi-head binary output.',
    'Forensic cues enter classification; Grad-CAM/proxy and forensic overlays interpret outputs; conditional IoU assessment.',
    'FF++; Celeb-DF v2; DFDC; separate PoC/robustness sets',
    'DFDC 79% / 0.77 F1 / 0.82 AUC. FF++ 97% must remain separate; architecture and timing have unresolved details.',
    'Video -> frames at 25 fps -> MTCNN alignment -> 224x224 [0,1] / JPEG QF 50,75,100\n'
    '         +-> PRNU-like residual / JPEG blocks / DCT-spectral features --+\n'
    '         +-> ResNet50 spatial representation -------------------------+\n'
    '         +-> unspecified ViT representation --------------------------+\n'
    '         -> concatenate -> trainable FC + dropout -> multi-head output -> real/fake\n'
    '         -> post-hoc Grad-CAM and forensic overlays -> analyst\n'
    'Temporal tokenization, fusion dimensions and head details remain unspecified.')
note(p,'Architecture','Backbones','ResNet50 extracts local pixel artifacts; a Vision Transformer extracts global relationships. The ViT size/patches/depth/heads, pretrained initialization and actual temporal modeling are not specified. The prose claims temporal coherence but does not define cross-frame tokens, ordering or aggregation.','p. 3, Sections III-C-D')
note(p,'Architecture','Forensic feature branch','Extract PRNU/noise residual cues, JPEG quantization/block artifacts and DCT/spectral descriptors. Feature dimensions, extractor filters, normalization and whether a camera-reference PRNU template is required are not specified. The figure labels the residual “PRNU-like” and its spectrum FFT magnitude.','p. 3, Section III-C; p. 5, Fig. 3')
note(p,'Architecture','Fusion and classification','The most concrete definition is concatenation of independently extracted representations, then a trainable fully connected layer with dropout, followed by a “multi-head” classification block giving real/manipulated probability. The text also calls it more than simple concatenation; no additional joint-attention mechanism or mathematical fusion objective is supplied. Head count and dimensions are unknown.','p. 3, Section III-D, Fig. 1')
note(p,'Explainability','Prediction versus explanation','Forensic features participate in the decision because they are fused with deep features. Grad-CAM and forensic heatmaps are displayed after the decision for human interpretation. There is no documented explanation-derived training loss or evidence that viewing a heatmap itself improves classification.','pp. 3 and 7-9')
note(p,'Explainability','Proxy-map qualification','Figures label CNN feature/transformer attention maps as proxies (p. 5) and Grad-CAM as a proxy (p. 7). The target CNN layer and actual transformer-attention extraction are not documented. Those illustrations alone do not establish faithful learned attention or backpropagated attribution.','p. 5, Fig. 3; p. 7, Fig. 9','Interpretation')
note(p,'Explainability','IoU experiment','Normalize maps, retain top 20% activations, compare with manipulation masks; IoU >= 0.3 is deemed meaningful. 410 of 500 correctly classified fake samples from FF++/Celeb-DF v2 meet the criterion. 82% is a conditional alignment rate, not mean IoU, DFDC accuracy or an all-input explanation success rate. Mask provenance and sampling details are insufficiently documented.','p. 8, Eq. 6 and Section IV-F')
note(p,'Dataset','DFDC involvement and volumes','The article describes DFDC as exceeding 100,000 clips and reports its own Table V experiment. Actual videos/frames used, DFDC subset/release details, identities, real/fake counts and a split manifest are not provided. The corpus description must not be used as the actual training volume.','pp. 2-3, Section III-A/E; p. 6, Table V')
note(p,'Dataset','Preprocessing','Extract frames at 25 fps, align with MTCNN, resize to 224x224 and normalize to [0,1]. JPEG quality factors 50, 75 and 100 are used to retain/test compression cues. Operation order and stored crop selection are not fully specified.','p. 3, Section III-B; p. 4, Fig. 2')
note(p,'Dataset','Split and evaluation protocol','The paper says stratified 80/10/10 train/validation/test for each dataset, plus separate cross-dataset evaluation. It does not establish whether splits are source-video/identity disjoint, whether metrics are frame- or video-level, or how frame predictions are aggregated. Stratification alone does not mean balanced classes.','pp. 3 and 6','Interpretation')
note(p,'Dataset','Separate PoC','A streaming pilot uses 50 clips from public repositories and locally generated swaps. It is not identified as a DFDC test set. Its reported confusion counts are TP22/TN24/FP2/FN2; those imply 24 fakes and 26 real clips, whereas the text calls the set balanced.','pp. 8-9, Section IV-H','Calculated')
note(p,'Optimization','Optimizer and scheduler','Adam with initial learning rate 1e-4 and cosine annealing is reported. Dropout regularizes the fusion layer. Loss function, dropout probability, scheduler horizon, minimum LR, weight decay, stopping rule, pretrained/frozen layers and seed are not reported.','pp. 3 and 5, Sections III-E and IV-C')
note(p,'Optimization','Augmentation and balancing','Horizontal flipping, color jittering and Gaussian noise are used. The paper says Celeb-DF recall improves 0.78 -> 0.81 with augmentation. No augmentation probabilities or class-balancing loss/oversampling procedure are specified.','pp. 3 and 6')
note(p,'Compute','Training resources','Training is described only as NVIDIA GPUs with mixed precision, allegedly reducing memory about 40% and epoch time about 30%; absolute baselines, GPU model/count and training duration are missing. The T4 is specifically named for the deployment pilot, not established as the training GPU.','pp. 3 and 6')
note(p,'Compute','Epochs and batch size','Figure 4 shows a curve through epoch 20. A textual total-epoch schedule, training batch size and parameter count are Not Reported. Treat “20 plotted epochs” as figure evidence, not a verified complete training configuration.','p. 5, Fig. 4; complete PDF pp. 1-11','Unknown')
note(p,'Compute','Published deployment','Single NVIDIA T4 16 GB, Intel Xeon Silver CPU and 32 GB RAM; Python 3.10, PyTorch/OpenCV, Flask/gRPC and Redis/Celery. Pilot sampling is 5-10 fps, with timing measured at 8 fps and 224x224. The paper reports 2.17 s per ten-second clip, 120 fps average and 11.5 videos/min.','pp. 3 and 8, Table IX')
note(p,'Limitations','Timing reconciliation','At 8 fps, a ten-second clip supplies 80 sampled frames. 80/2.17 = 36.87 sampled frames/s and 60/2.17 = 27.65 clips/min, not the reported 120 fps / 11.5 clips/min. Counting all 250 original frames gives 115.21 frame-equivalents/s and may partly explain 120, but does not reconcile 11.5. The reported latency is retained with this qualification.','p. 8, Table IX; arithmetic from stated duration and sampling','Calculated')
note(p,'Limitations','Missing implementation and validation','Unspecified ViT/temporal module, forensic descriptor dimensions, loss, fusion head and data volumes prevent exact reproduction. No runs/seeds/intervals or experiment artifact links are supplied. A frame-level split could leak content if implemented without source grouping; this is a risk, not an established occurrence.','pp. 2-9; complete PDF pp. 1-11','Interpretation')
note(p,'Limitations','Other methodological cautions','Figure 4 training loss goes below zero while the loss objective is unspecified. Proxy illustrations and incompletely documented manipulation masks limit assessment of the 82% explanation claim. Table VIII compares other studies without matched protocol and uses 0.96 FF++ F1 as an overall headline. These are reporting concerns, not proof of fabricated experiments.','pp. 5 and 7-8, Figs. 3-4, Tables VII-VIII','Interpretation')
note(p,'Future relevance','Existing mechanisms to retain','The paper motivates fusion of explicit forensic cues with local/global learned representations, JPEG-aware preprocessing, dropout, cosine scheduling, robustness evaluation and separate explanation assessment. These are mechanisms to understand; the paper is not a complete implementable blueprint for a new hybrid.','pp. 3-8','Interpretation')
for dataset,table,rows in [
 ('FF++','p. 6, Table III',[(.88,.87,.81,.84,.89),(.95,.94,.93,.94,.96),(.96,.95,.94,.95,.97),(.97,.96,.95,.96,.98)]),
 ('Celeb-DF v2','p. 5, Table II; p. 6, Table IV',[(.70,.68,.63,.66,.71),(.76,.74,.70,.72,.79),(.79,.77,.75,.76,.82),(.84,.83,.81,.82,.87)]),
 ('DFDC, release/subset unspecified','p. 6, Table V',[(.67,.65,.60,.62,.68),(.72,.71,.69,.70,.74),(.74,.73,.71,.72,.77),(.79,.78,.76,.77,.82)])]:
    for i,(model,vals) in enumerate(zip(['Forensic only','ResNet50 only','ViT only','ResNet50 + ViT + forensic fusion'],rows)):
        result(p,dataset,'Reported stratified 80/10/10; sample unit/grouping unspecified',model,table,*vals,dfdc=dataset.startswith('DFDC'),proposed=i==3)
result(p,'Celeb-DF v2','Train FF++, test Celeb-DF v2; separate cross-dataset result','Hybrid','p. 6, training paragraph',auc=.83,proposed=True)
for qf,vals in [(100,(.97,.96,.95,.96,.98)),(75,(.94,.93,.91,.92,.95)),(50,(.90,.88,.86,.87,.91))]:
    result(p,'FF++',f'Compression QF={qf}','Hybrid','p. 6, Table VI',*vals,proposed=True)
result(p,'Dataset not allocated','FGSM epsilon=0.01; not a verified DFDC-specific result','Hybrid','p. 7, Table VII and prose',.85,.83,.82,.83,.84,proposed=True)
result(p,'Unseen diffusion manipulations; corpus unnamed','Not seen during training; source/split unspecified','Hybrid','p. 7, Fig. 8 and prose',f1=.79,auc=.83,proposed=True)
result(p,'50-clip streaming pilot; not DFDC','Public/local-swap pilot','Hybrid','p. 9, Section IV-H',.92,.92,.92,.92,other='Reported rounded metrics; counts TP22/TN24/FP2/FN2.',proposed=True)

p=paper(45,'Methodological reference - FF++ XAI','No - FF++ experiments only',
    'ImageNet1K-pretrained EfficientNetV2 multiclass detector; separate explainer and NES perturbation evaluation.',
    'Grad-CAM++, RISE, SHAP, LIME, SOBOL; top-SLIC-region attacks test explanation faithfulness.',
    'FF++ real + DF/F2F/FS/NT; fake-frame XAI sets',
    'LIME ranks well under this particular attack-based test. Lower post-attack accuracy is not a new detector improvement.',
    'Video -> 1 frame/s -> RetinaFace -> enlarge box x1.3 -> crop\n'
    '      -> ImageNet1K EfficientNetV2 -> multiclass real/manipulation decision\n'
    'Correct fake -> explainer heatmap -> SLIC (~50 regions) -> top 1/2/3 regions\n'
    '            -> masked NES attack -> re-score unchanged detector\n'
    '            -> retained detection accuracy / prediction-drop sufficiency\n'
    'Explainers do not retrain or become attention layers in the detector.')
note(p,'Architecture','Detector','An ImageNet1K-pretrained EfficientNetV2 from timm is fine-tuned for multiclass classification / deepfake attribution. EfficientNetV2 uses MBConv and fused-MBConv. The exact V2 variant, input resolution and classifier-head dimensions are not stated. Do not import the unrelated EfficientNet-B7 winning ensemble mentioned in background.','pp. 3 and 5, Sections 3.1 and 4.1')
note(p,'Architecture','Class structure and feature flow','Face crops feed one EfficientNetV2 detector and a multiclass cross-entropy objective. Four fake categories and pristine content are described; five classes is a reasonable inference from that setup, but the exact output tensor shape is not explicitly listed. No temporal backbone, feature fusion or added attention block is implemented.','pp. 2-5','Interpretation')
note(p,'Explainability','Grad-CAM++','Uses positive higher-order partial-derivative weighting rather than ordinary Grad-CAM. The implementation averages results over all convolutional 2-D layers, rather than simply choosing only the last one. This is a post-hoc attribution method.','p. 3, Section 3.2; p. 5, implementation settings')
note(p,'Explainability','RISE','Random binary masks perturb the input; predictions weight masks and their aggregation forms the explanation. Uses 4,000 masks with other parameters at defaults. No detector retraining is involved.','pp. 3 and 5')
note(p,'Explainability','SHAP','Shapley-inspired additive feature attribution is approximated using 2,000 evaluations and a blurring masker with kernel size 128. It explains the existing detector; the article does not turn SHAP into a classification branch.','pp. 3 and 5')
note(p,'Explainability','LIME','Segment input using SLIC targeting 50 superpixels, apply 2,000 random segment-mask perturbations, fit a local interpretable linear model, then rank segment coefficients. It is a local surrogate, not a trainable attention mechanism in the detector.','pp. 3-5')
note(p,'Explainability','SOBOL','Uses quasi-Monte Carlo masks and output-variance sensitivity / total-order Sobol indices. Grid size is 8 and number of designs 32; other settings are left at defaults.','pp. 3-5')
note(p,'Explainability','Quantitative pipeline','For a correctly detected fake, overlay its explanation on SLIC regions and average pixel scores per region; LIME already supplies segment scores. Attack the top-1, top-2 and top-3 regions with NES. Lower retained detector accuracy and larger original-to-attacked prediction drop indicate more influential selected regions under this framework.','p. 4, Section 3.3, Algorithm 1')
note(p,'Explainability','Purpose and limitations','Attributions do not change the trained classifier. Deliberately perturbed evaluation images can change its output. Attack success is a proxy for explanation influence, not automatically overlap with authentic manipulation masks, calibrated forensic evidence or detector robustness on clean data.','pp. 4-7','Interpretation')
note(p,'Dataset','Actual corpus and split','Only FF++ experiments: 1,000 pristine and 4,000 fake videos, 1,000 per DF/F2F/FS/NT class. The paper states train/validation/test divisions of 720/140/140 videos; it does not enumerate extracted frame totals or all per-class split counts. DFDC is mentioned as context for other models, not an experiment.','p. 3, Section 3.1; p. 5, Section 4.1')
note(p,'Dataset','Frame processing','Detector training/test crops are obtained at one frame per second using RetinaFace, with bounding boxes enlarged by 1.3 for contextual background. Crop resolution, FF++ compression level, missed-face policy, random seed and exact training class balance after cropping are not stated.','p. 5, Section 4.1')
note(p,'Dataset','XAI evaluation volumes','Take 127 test videos per fake category and 10 frames/video: 1,270 images per category, 5,080 across four categories (calculated total). Framework comparison Tables 3-4 uses a separate random 150 images/category, or 600 total. These are not training sizes or DFDC samples.','pp. 5-6; arithmetic 4 x 127 x 10 and 4 x 150','Calculated')
note(p,'Optimization','Actual detector training','30 epochs, AdamW, learning rate 5e-5, weight decay 0.1, multiclass cross-entropy. Augmentations: RandomErasing, RandomResizedCrop, AugMix. Stochastic depth uses drop-path rate 0.4. No training scheduler, batch size, class weights, total parameters, duration or AMP mode is reported.','p. 5, Section 4.1')
note(p,'Optimization','NES evaluation settings','Maximum 50 iterations, learning rate 1/255, distortion limit 16/255, search variance 0.001, and 40 samples. Early stop when the detector predicts real. These are evaluation attack parameters, not detector-training optimization or adversarial training.','pp. 4-5, Algorithm 1 / Section 4.1')
note(p,'Compute','Published hardware','Experiments use NVIDIA RTX 4090 GPU cards; the number of cards, their peak allocation, batch sizes and per-run duration are not supplied. The public reproduction-code URL printed in the paper is https://github.com/IDT-ITI/XAI-Deepfakes . It was not fetched or executed for this analysis.','p. 5, Section 4.1')
note(p,'Compute','Query workload bound','Under the intended two-sided NES interpretation: 2 x 40 x 50 = 4,000 detector evaluations per attack, plus stopping checks. Across 5,080 images, five methods and three top-k settings, the ceiling is 304,800,000 attack evaluations before early stopping, caching/batching or explanation generation. This is a calculated workload bound, not elapsed time.','pp. 4-5, Algorithm 1 and settings; analyst two-sided-NES assumption','Calculated',304800000,'forward evaluations')
note(p,'Limitations','Printed algorithm inconsistency','Visual inspection confirms that Algorithm 1 lines 8 and 9 both use x_adv + sigma*u while one term is added and one subtracted. For deterministic outputs they cancel, yielding zero estimated gradient. This appears to be a printing error; do not silently repair it or assume the implementation follows a corrected formula. The position of clipping also does not fully specify projection of cumulative perturbations.','p. 4, Algorithm 1, lines 8-11','Interpretation')
note(p,'Limitations','Conditioning and comparability','The framework is framed around correctly detected fakes, yet Table 1 original accuracies are below one; the denominator/filtering detail is not fully explained. No real-class accuracy, overall multiclass accuracy, F1 or ROC-AUC is supplied. Results are per fake type and cannot be treated as balanced binary whole-dataset accuracy.','pp. 4-5, Table 1','Interpretation')
note(p,'Limitations','Generalization of XAI ranking','Only one detector family and FF++ are evaluated. Tables 3-4 show explanation rankings change when the evaluation framework changes. SHAP exceeds LIME for FS top-3 in Tables 1-2. A broad claim that LIME is always best is unsupported. Top-k segments can have different areas, and uncertainty/rerun statistics are not reported.','pp. 5-7, Tables 1-4','Interpretation')
note(p,'Future relevance','Existing mechanisms to retain','Useful mechanisms are multiclass detector training with augmentation/AdamW/stochastic depth, complementary attribution methods, controlled-region perturbation testing, and separate clean-detection/explanation metrics. The framework is an XAI evaluation reference, not evidence of a DFDC detector baseline or a new hybrid.','pp. 3-7','Interpretation')
for subtype,acc in [('DF',.978),('F2F',.977),('FS',.982),('NT',.924)]:
    result(p,'FF++ '+subtype,'Original image set; per fake-type accuracy','EfficientNetV2','p. 5, Table 1',accuracy=acc,proposed=True)
attack_rows = {
 'Grad-CAM++':[[.781,.644,.571],[.864,.798,.737],[.887,.808,.728],[.601,.481,.432]],
 'RISE':[[.877,.766,.686],[.843,.710,.622],[.896,.809,.734],[.783,.637,.513]],
 'SHAP':[[.813,.609,.450],[.846,.739,.637],[.876,.702,.543],[.686,.497,.344]],
 'LIME':[[.735,.440,.245],[.803,.633,.484],[.864,.698,.559],[.579,.340,.197]],
 'SOBOL':[[.750,.591,.490],[.816,.653,.512],[.874,.703,.574],[.621,.417,.313]],
}
sufficiency_rows={
 'Grad-CAM++':[[.148,.253,.310],[.069,.115,.162],[.063,.113,.160],[.194,.251,.280]],
 'RISE':[[.087,.162,.219],[.091,.173,.223],[.060,.114,.157],[.115,.204,.273]],
 'SHAP':[[.137,.300,.402],[.092,.158,.222],[.073,.181,.269],[.167,.282,.357]],
 'LIME':[[.195,.408,.539],[.121,.238,.334],[.087,.189,.262],[.233,.363,.431]],
 'SOBOL':[[.166,.277,.352],[.108,.212,.296],[.078,.180,.259],[.198,.302,.362]],
}
p['xai_attack_table']=attack_rows
p['xai_sufficiency_table']=sufficiency_rows

hardware = [
 dict(paper_id=2,published_gpu='Not Reported',epochs='Not Reported',batch='Not Reported',parameters='Not Reported',published_time='Not Reported',
      estimate='Conditionally feasible with one branch resident at a time. At the reported 320x320 resolution, AMP and a trial microbatch of 1-4 are plausible starting assumptions, not measured fit guarantees. Simultaneous three-branch training adds activation and optimizer memory; sequential inference avoids holding all branch activations.',
      bottleneck='Face extraction, several training passes, crop/drop views and explanation gradients. Exact peak VRAM/time cannot be predicted without attention dimensions and a complete training schedule.',ref='P2 pp. 6-9; missing resources: full pp. 1-14'),
 dict(paper_id=9,published_gpu='Not Reported',epochs='Not Reported; projection after 5 regular steps',batch='Not Reported',parameters='Calculated 10,240 prototype scalars for 40 x 256; total unknown',published_time='<1 s deletion / a few seconds replacement; hardware unspecified',
      estimate='Inference and cached final-layer refinement are plausible on 8 GB, conditional on HRNet variant and input resolution. Full encoder training fit is uncertain. Begin any later capacity check with one temporal sample; no batch fit is claimed here.',
      bottleneck='Optical-flow preprocessing and host/disk embedding cache. 100,000 hypothetical FP32 latent grids occupy 24.41 GiB, exceeding the 16 GB RAM context and consuming most of the 30 GiB additional storage budget.',ref='P9 pp. 3-4, 9 and 11-12'),
 dict(paper_id=15,published_gpu='Not applicable - review',epochs='Not applicable',batch='Not applicable',parameters='No implemented model',published_time='No experiment',estimate='No paper-specific training/inference workload exists to fit. A GPU-memory estimate would require inventing a detector configuration.',bottleneck='Missing executable architecture and experiment; use conceptual taxonomy only.',ref='P15 complete pp. 1-7'),
 dict(paper_id=16,published_gpu='Not applicable - review',epochs='Not applicable',batch='Not applicable',parameters='No implemented model',published_time='No experiment',estimate='No paper-specific reproduction resource estimate is possible. Reviewed families vary widely; no architecture is selected in this analysis.',bottleneck='Review numbers have no shared training/protocol definition.',ref='P16 complete pp. 1-7'),
 dict(paper_id=35,published_gpu='Training: NVIDIA unspecified. Pilot: 1 x T4 16 GB',epochs='20 plotted epochs; full schedule unknown',batch='Not Reported',parameters='Not Reported; ViT variant unknown',published_time='Pilot: 2.17 s/10-s clip; throughput inconsistent. Training time unknown',
      estimate='Inference at 224x224 may fit with a single clip/frame microbatch; joint training of ResNet50 + ViT depends strongly on unspecified ViT size and temporal tokens. A trial microbatch of 1-2 with AMP is an estimate, not a verified reproduction. The 16 GB T4/32 GB host pilot does not establish fit on an 8 GB/16 GB laptop.',
      bottleneck='Joint branch activations, forensic extraction and XAI gradients. No conversion of reported T4 throughput to RTX 4060 speed is defensible.',ref='P35 pp. 3, 5-6 and 8, Table IX'),
 dict(paper_id=45,published_gpu='RTX 4090 GPU cards; count unspecified',epochs=30,batch='Not Reported',parameters='EfficientNetV2 variant/total unknown',published_time='Not Reported',
      estimate='Detector inference and small-batch explanation generation are plausible at moderate resolution, but exact fit is unknown because variant/resolution are missing. An illustrative later check at 224/256 and batch 1-4 is a changed assumption, not the original protocol. Serial mask microbatches reduce peak memory; the complete attack sweep remains expensive.',
      bottleneck='Thousands of forward evaluations per explanation/attack; 8 GB is not the only constraint. Full worst-case NES ceiling is 304.8 million forward evaluations before explanations and early-stop savings.',ref='P45 pp. 3-6, Algorithm 1 / Section 4.1'),
]

assert [p['paper_id'] for p in papers]==[2,9,15,16,35,45]
dfdc=[dict(paper_id=p['paper_id'],**r) for p in papers for r in p['performance'] if r['dfdc']]
assert {r['paper_id'] for r in dfdc}=={2,35} and len(dfdc)==8
output=dict(source_workbook=str(SOURCE),source_sha256=source_hash,papers=papers,hardware=hardware,dfdc_performance=dfdc,
            evidence_types=['Reported','Reported (review)','Calculated','Estimated','Unknown','Interpretation'],
            read_scope='All 63 physical PDF pages across the six original source files, with representative tables/figures visually inspected. No external publication discovery or code execution.',
            user_hardware={'gpu':'RTX 4060 Laptop','vram_gib':8,'ram_gib_approx':16,'additional_storage_gib':30})
(WORK/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')


def fmt(v):
    return 'Not Reported' if v is None else f'{v:.4f}'


lines=['# Detailed analysis of six existing papers','',
       'Scope: **Paper IDs 2, 9, 15, 16, 35 and 45 only**. All 63 physical PDF pages were read. This report and the separate nine-sheet analysis workbook do not change the authoritative literature workbook or add publications. No dataset, model checkpoint or source code was downloaded, implemented or trained.','',
       '**Finding:** Papers **2 and 35** report their own DFDC experiments. Papers **9 and 45** perform FF++ experiments and are methodological references for this task. Papers **15 and 16** are reviews and supply no independent detector experiment. “Verified” here means a value was found in the original PDF and attributed to its reported protocol; it does not mean an experiment was rerun or its results independently reproduced.','',
       'Evidence labels: **Reported** is stated in the original paper; **Reported (review)** is a secondary comparison in a review; **Calculated** is transparent arithmetic from stated values; **Estimated** is an analyst hardware scenario; **Unknown** means not supplied; **Interpretation** is a source-grounded assessment. Missing F1/AUC values are not zeros. References use physical PDF pages, even where printed article pagination differs.','',
       '## DFDC-only reported performance','',
       '| Paper | Model | Accuracy | Precision | Recall | F1 | AUC | Original evidence |',
       '|---|---|---:|---:|---:|---:|---:|---|']
for r in dfdc:
    lines.append(f"| {r['paper_id']} | {r['model']} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['auc'])} | P{r['paper_id']}, {r['reference']} |")
lines += ['', 'All metric values above are fractions in [0,1]. Paper 2 describes a full-DFDC 5,000-clip test and frame averaging but leaves some reporting-unit details unclear. Paper 35 describes an 80/10/10 split without actual DFDC volumes, release/subset or frame/video grouping. **Do not rank the two papers as a matched benchmark or average their results.** Both rows for proposed systems and their within-paper baselines are retained; eight experiment rows represent only two papers.','']
for p in papers:
    pid=p['paper_id'];src=p['source'].replace('\\','/')
    lines += [f"## Paper {pid}: {p['title']}",'',f"**Classification:** {p['role']}. **Independent DFDC:** {p['independent_dfdc']}.",'',
              f"Original PDF: [{Path(p['source']).name}](</{src}>) ({p['pages']} pages). SHA-256: `{p['sha256']}`.",'',
              '### Technical workflow','', '```text',p['diagram'],'```','',
              'The diagram summarizes the source-defined blocks; unspecified tensors, layers and training settings remain unspecified.']
    for category in ['Architecture','Explainability','Dataset','Performance','Compute','Optimization','Limitations','Future relevance']:
        notes=[n for n in p['notes'] if n['section']==category]
        if notes:
            lines += ['',f'### {category}','']
            for n in notes:
                lines += [f"**{n['item']} [{n['status']}].** {n['text']} (P{pid}, PDF {n['reference']}).",'']
        if category=='Performance' and p['performance']:
            if not notes:lines += ['', '### Performance','']
            lines += ['Values are fractions. Each row has its own dataset and protocol.','',
                      '| Dataset | Protocol | Model | Accuracy | Precision | Recall | F1 | AUC | Evidence |',
                      '|---|---|---|---:|---:|---:|---:|---:|---|']
            for r in p['performance']:
                lines.append(f"| {r['dataset']} | {r['protocol']} | {r['model']} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['auc'])} | P{pid}, {r['reference']} |")
            if pid==45:
                lines += ['', 'The following are **retained post-attack accuracies**, not clean detection improvements. For each method/type the tuple is top-1 / top-2 / top-3 attacked SLIC regions. Lower is interpreted as stronger explanation influence under this framework (P45, PDF p. 5, Table 1).','',
                          '| Explainer | DF | F2F | FS | NT |','|---|---|---|---|---|']
                for method,vals in p['xai_attack_table'].items():lines.append('| '+method+' | '+' | '.join(' / '.join(f'{x:.3f}' for x in triplet) for triplet in vals)+' |')
                lines += ['', 'Prediction-drop sufficiency scores are distinct from accuracy; higher indicates greater effect. Tuples again use top-1 / top-2 / top-3 (P45, PDF p. 6, Table 2).','',
                          '| Explainer | DF | F2F | FS | NT |','|---|---|---|---|---|']
                for method,vals in p['xai_sufficiency_table'].items():lines.append('| '+method+' | '+' | '.join(' / '.join(f'{x:.3f}' for x in triplet) for triplet in vals)+' |')
    h=next(h for h in hardware if h['paper_id']==pid)
    lines += ['', '### RTX 4060 8 GB feasibility estimate','',
              f"**Published:** GPU: {h['published_gpu']}. Epochs: {h['epochs']}. Batch: {h['batch']}. Parameters: {h['parameters']}. Time: {h['published_time']}. (Original PDF: {h['ref']}).",'',
              f"**Estimated, not measured on this laptop:** {h['estimate']}",'',
              f"**Constraint:** {h['bottleneck']} Published inputs are cited above; feasibility assessments do not substitute for missing original settings.",'']
lines += ['## Hardware calculations and boundaries','',
          'The hardware comparison uses the user-provided RTX 4060 Laptop GPU with 8 GB VRAM, about 16 GB RAM, and 30 GiB additional D: storage. Small microbatch suggestions are capacity-check assumptions, not published settings, guaranteed fits or measured timings. No architectural combination is invented.','',
          '- **P9 cache calculation:** 16 x 16 x 256 x 4 bytes = 262,144 bytes = 0.25 MiB per latent grid. At a hypothetical 100,000 grids this is 24.414 GiB before overhead. Forty FP32 maximum-similarity scores per grid need 160 bytes, or 15.259 MiB for that hypothetical count. The paper reports the dimensions, not that sample count or storage footprint (P9, PDF p. 4).',
          '- **P45 workload bound:** 2 x 40 x 50 = 4,000 forward evaluations/attack under the intended two-sided interpretation; 5,080 x 5 methods x 3 region settings x 4,000 = 304.8 million evaluations, excluding explanations/stopping checks. At an illustrative 100 evaluations/s this would be 846.7 hours before early-stop savings. The 100/s rate is an editable illustration, not a measured RTX 4060 throughput. The printed pseudocode issue must be resolved before treating this as an executable recipe (P45, PDF pp. 4-5).',
          '- **P35 throughput reconciliation:** 10 s x 8 sampled frames/s = 80 sampled frames; 80 / 2.17 = 36.87 sampled frames/s; 60 / 2.17 = 27.65 clips/min. The paper instead lists 120 fps and 11.5 videos/min. Its end-to-end claim is retained as reported, with the inconsistency exposed (P35, PDF p. 8, Table IX).','',
          'No missing total parameter count, GPU count, sampled training volume or training duration has been invented. Exact peak VRAM and training time remain unknown where backbone variants, sequence lengths, loss/head details or batches are absent.','',
          '## Preservation and reproducibility record','',
          f'Authoritative workbook read-only fingerprint: `{source_hash}`. Original path: `{SOURCE}`. The original workbook was neither saved nor replaced. Each original PDF was freshly parsed and its SHA-256 matched the existing source inventory. The separate analysis workbook has exactly nine sheets: Summary, DFDC Verified, Hardware Feasibility, 2, 9, 15, 16, 35, 45.','',
          'All interpretation and feasibility estimates are restricted to mechanisms already described in these six papers. No new publication discovery, model design, data acquisition or experiment is part of this analysis.','']
(PROJECT/'../research/reports/Existing_Six_Papers_Deep_Analysis.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'papers':len(papers),'pages_read':sum(p['pages'] for p in papers),'dfdc_papers':[2,35],'dfdc_rows':len(dfdc),'notes':sum(len(p['notes']) for p in papers),'original_sha256':source_hash},indent=2))
