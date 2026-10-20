"""Record the independently reviewed decisions; do not modify the workbook.

The judgments below are curated evidence assessments, not automatic inclusion
rules.  Page numbers are physical PDF pages, including repository cover pages.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
WORK = PROJECT / '../research/working/expansion'
load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
metadata = {r['key']: r for r in load(WORK / 'bibliographic_records.json')}
metadata['pdf_075']['crossref'] = load(WORK / 'convnext_crossref.json')['message']
pdfs = {r['key']: r for r in load(WORK / 'candidate_pdf_inspections.json')}
master = load(WORK / 'original_master.json')
validation = load(PROJECT / '../research/analyses/DFDC_Literature_Expansion_Validation.json')

IJCB_ACCEPTED = 'https://iapr.org/archives/ijcb2024/index.html%3Fp=1732.html'
IJCB_CFP = 'https://iapr.org/archives/ijcb2024/index.html%3Fp=222.html'
IJCB_PROCEEDINGS = 'https://iapr.org/archives/ijcb2024/index.html%3Fp=2804.html'
TBIOM = 'https://ieee-biometrics.org/publications/t-biom/'
TBIOM_AUTHORS = 'https://ieee-biometrics.org/publications/t-biom/author-instructions/'
ACCESS_REVIEW = 'https://ieeeaccess.ieee.org/reviewers/reviewer-guidelines/'
THESIS = 'https://digitalcommons.murraystate.edu/honorstheses/288/'

curated = [
    {
        'key': 'pdf_059', 'decision': 'Needs Manual Review',
        'peer_review': 'Final Image and Vision Computing research article corroborated by the publisher page and Crossref, not an unreviewed preprint. Individual review reports were not obtained.',
        'peer_review_urls': ['https://www.sciencedirect.com/science/article/pii/S0262885625002847'],
        'dfdc': 'Original secondary DFDC Preview evaluation in Table 4, PDF p. 8: accuracy 91.7%, precision 90.9%, recall 91.2%, F1 91.0%. No DFDC AUC reported. This is distinct from the approximately 99% image-classification headline.',
        'metric_evidence': [{'page': 8, 'table': '4', 'dataset': 'DFDC Preview', 'accuracy': .917, 'precision': .909, 'recall': .912, 'f1': .91}],
        'methodology': 'Fine-tuned VGG16 on 256×256 faces, followed by post-hoc LIME. Main training benchmark is the 140K FFHQ/StyleGAN real/fake image corpus with a 70/15/15 split. LIME is not trainable architectural attention.',
        'reason': 'DFDC is a limited secondary check; its actual sample volumes, independent split, video/frame unit and aggregation are insufficiently specified. Substantial DFDC centrality is not established. Free full text was also not located at an authorized public source.',
        'access': 'Publisher offers institutional access/purchase; OpenAlex records closed access with no repository full text. No verified free PDF URL.',
        'authorized_pdf_url': None,
    },
    {
        'key': 'pdf_074', 'decision': 'Needs Manual Review',
        'peer_review': 'Final ICSCN 2025 IEEE proceedings record and the article proceedings header agree. A separate conference review-policy record was not established; proceedings publication alone is not treated as conclusive proof of every acceptance gate.',
        'peer_review_urls': ['https://ieeexplore.ieee.org/document/11308255/'],
        'dfdc': 'Table II, PDF p. 7, reports DFDC Preview accuracy 94.28% at 100 frames/video. The AUC cell is printed as 0.95%, while surrounding prose uses 0.95–0.96 as fractions. The audit retains that unit ambiguity.',
        'metric_evidence': [{'page': 7, 'table': 'II', 'dataset': 'DFDC Preview', 'accuracy': .9428, 'auc_as_printed': '0.95% (prose: 0.95–0.96)'}],
        'methodology': 'ResNeXt-50 visual features, an unspecified audio CNN and spatiotemporal transformer fusion; self/cross attention and post-hoc Grad-CAM are distinct. Reports 112×112 inputs, 100 frames/video and an RTX 3090. A generic 6,000-video 70/30 split is not tied to a reproducible DFDC allocation.',
        'reason': 'DFDC appears only in the short result table, with no DFDC-specific split, class counts or complete transformer specification. Centrality and protocol quality remain ambiguous. No verified free full text found.',
        'access': 'OpenAlex closed; no authorized repository or author full-text version located. Publisher record is not itself an open PDF.',
        'authorized_pdf_url': None,
    },
    {
        'key': 'pdf_075', 'decision': 'Qualified but Inaccessible',
        'peer_review': 'Exact title appears in the official IJCB 2024 accepted-paper list; the official CFP describes reviewed submissions and IEEE Xplore publication, corroborated by final DOI/proceedings metadata.',
        'peer_review_urls': [IJCB_ACCEPTED, IJCB_CFP, 'https://ieeexplore.ieee.org/document/10744484/'],
        'dfdc': 'Substantial DFDC Preview benchmark: Table 1, PDF p. 6, accuracy 90.87%, AUC 93.62% (0.9362). Table 3, p. 7, independently reports FF++→DFDC-P accuracy 75.42% and Celeb-DF→DFDC-P 60.00%. These are separate protocols.',
        'metric_evidence': [{'page': 6, 'table': '1', 'dataset': 'DFDC Preview intra-dataset', 'accuracy': .9087, 'auc': .9362}, {'page': 7, 'table': '3', 'dataset': 'FF++→DFDC Preview', 'accuracy': .7542}, {'page': 7, 'table': '3', 'dataset': 'Celeb-DF→DFDC Preview', 'accuracy': .60}],
        'methodology': 'MTCNN facial crops resized to 224×224; modified ConvNeXt with LeakyReLU and prototype learning inspired by ProtoPNet, followed by similarity-based classification. Prototype explanations are intrinsic; no added attention mechanism is established. Clear intra/cross-corpus comparisons and contribution. Actual sampled frames, source-group split, GPU, batch size and epoch count are not sufficiently reported; these remain limitations rather than inferred values.',
        'reason': 'Research relevance, peer review and understandable methodology pass the literature inclusion threshold, with reproducibility limits recorded. Authorized free full-text access fails the mandatory access gate.',
        'access': 'Official IJCB proceedings archive explicitly requires a password; it is not freely downloadable. OpenAlex records closed access and no repository full text. Author-lab/CV and exact-title manuscript searches did not establish a free article PDF. Publisher access did not return a usable free PDF. No password or access control was bypassed.',
        'access_urls': [IJCB_PROCEEDINGS, 'https://msplab.uettaxila.edu.pk/wp-content/uploads/2025/07/ali-javed-cv-2025.pdf'],
        'authorized_pdf_url': None,
    },
    {
        'key': 'pdf_077', 'decision': 'Rejected',
        'title': 'Deepfakes on Trial: Developing a High-Accuracy, Court-Admissible AI Pipeline for Deepfake Detection in Corporate Fraud Litigation',
        'authors': ['Aiden J. Green'], 'year': 2025, 'doi': None,
        'venue': 'Murray State University, Honors College Theses, no. 288',
        'publication_url': THESIS,
        'peer_review': 'The authoritative university record identifies an honors thesis and its examining committee. No independently verified peer-reviewed journal/conference version was located. The journal wording in the author affiliation does not establish a separate journal publication.',
        'peer_review_urls': [THESIS],
        'dfdc': 'Original substantial full-DFDC pipeline is described. PDF p. 42 reports cross-validation accuracy 90.54%, AUC 0.9285, F1 0.7048; p. 54 reports held-out accuracy 91.72%, AUC about 0.94 and F1 about 0.705. Cross-validation and held-out outcomes must remain separate.',
        'metric_evidence': [{'page': 42, 'figure': '14', 'dataset': 'full DFDC, cross-validation', 'accuracy': .9054, 'auc': .9285, 'f1': .7048}, {'page': 54, 'dataset': 'full DFDC, held-out', 'accuracy': .9172, 'auc': .94, 'f1': .705}],
        'methodology': 'Five sampled frames/video; EfficientNet-B7 embeddings combined with handcrafted visual/audio forensic features and an MLP. Reports an 80/10/10 split and SMOTE. Source-video/identity grouping and metric averaging need care; reported dataset/class totals are not perfectly consistent.',
        'reason': 'Fails the required independently established publication peer-review gate. Thesis committee review is not substituted for the requested journal/conference review. The existing workbook containing other theses does not waive the new-candidate rule.',
        'access': 'University repository provides a legitimate free complete thesis under CC BY-NC-ND 4.0. No new download because the publication gate fails.',
        'authorized_pdf_url': 'https://digitalcommons.murraystate.edu/cgi/viewcontent.cgi?article=1293&context=honorstheses',
    },
    {
        'key': 'pdf_078', 'decision': 'Rejected',
        'peer_review': 'Final IEEE IITCEE 2026 proceedings publication corroborated by Crossref and the article. The correct publication year is 2026, not the earlier audit’s 2025. A separate article review report was not obtained.',
        'peer_review_urls': ['https://ieeexplore.ieee.org/document/11394363/'],
        'dfdc': 'PDF p. 5 states DFDC accuracy 94.8% and AUROC 0.975. Figure 5 instead shows 920+930 correct out of 2,000, which gives 92.5%. Figure 3 labels both ROC curves AUC=1.00, conflicting with the prose.',
        'metric_evidence': [{'page': 5, 'figure': '4/prose', 'dataset': 'DFDC release unspecified', 'claimed_accuracy': .948}, {'page': 5, 'figure': '3/prose', 'claimed_auc': .975, 'plot_auc': 1.0}, {'page': 5, 'figure': '5', 'confusion_counts': [920, 80, 70, 930], 'accuracy_computed_from_counts': .925}],
        'methodology': '3D-CNN spatiotemporal detection combined with an autoencoder reconstruction branch, Grad-CAM and saliency. Reports 32-frame 224×224 windows; backbone variant, fusion weights, DFDC volumes and train/test allocation are not clearly defined. Dual Xeon is named; no training GPU model is established.',
        'reason': 'Fails research-quality gate: directly conflicting DFDC accuracy/AUC evidence plus an underspecified evaluation protocol. The figures were visually checked; this is an inclusion judgment, not an allegation of misconduct.',
        'access': 'OpenAlex closed; no legitimate free full text located. The quality failure independently excludes the paper.',
        'authorized_pdf_url': None,
    },
    {
        'key': 'pdf_090', 'decision': 'Qualified but Inaccessible',
        'peer_review': 'Final T-BIOM 6(4), 486–497 (2024) DOI record agrees with the article. IEEE Biometrics explicitly identifies T-BIOM as peer reviewed and publishes its editorial/reviewer process.',
        'peer_review_urls': [TBIOM, TBIOM_AUTHORS, 'https://ieeexplore.ieee.org/document/10542403/'],
        'dfdc': 'DFDC Preview is one of three central corpora. Table III, PDF p. 8: accuracy 92.53%, AUC 0.93, F1 0.927. Table IV, same page: FF++→DFDC-P 78.6% accuracy; Celeb-DF→DFDC-P 70.04%. These are not intra-dataset metrics.',
        'metric_evidence': [{'page': 8, 'table': 'III', 'dataset': 'DFDC Preview intra-dataset', 'accuracy': .9253, 'auc': .93, 'f1': .927}, {'page': 8, 'table': 'IV', 'dataset': 'FF++→DFDC Preview', 'accuracy': .786}, {'page': 8, 'table': 'IV', 'dataset': 'Celeb-DF→DFDC Preview', 'accuracy': .7004}],
        'methodology': 'MTCNN 224×224 face crops, a modified DenseNet121 feature extractor, learnable real/fake prototypes, similarity scores and weighted softmax. Prototype clustering/separation objectives support intrinsic explanations; Grad-CAM is post-hoc. Actual sampled frame counts, split fraction, GPU and training duration are not sufficiently reported. Contribution and independent cross-corpus evaluation are understandable despite these reproducibility limits.',
        'reason': 'Passes the research criteria at the literature inclusion threshold, but no authorized free complete PDF was verified. Not integrated.',
        'access': 'OpenAlex closed with no repository full text. The official accepted-manuscript URL returns a login HTML page rather than a PDF (HTTP 200 is not proof of PDF access). Exact-title author/repository searches did not establish another free version.',
        'access_urls': ['https://ieeexplore.ieee.org/ielam/8423754/10767127/10542403-aam.pdf'],
        'authorized_pdf_url': None,
        'related_versions': [{'key': 'pdf_083', 'title': 'DFP-Net: An explainable and trustworthy framework for detecting deepfakes using interpretable prototypes', 'year': 2023, 'doi': '10.1109/IJCB57857.2023.10448842', 'url': 'https://doi.org/10.1109/IJCB57857.2023.10448842', 'assessment': 'Same authors, closely matching prototype architecture, and identical DFDC intra/cross-corpus accuracy values in the supplied papers. Consolidate as a version family for this task; journal adds F1/analysis. Not proof of plagiarism, and not an extra candidate or an existing-workbook duplicate.'}],
    },
    {
        'key': 'pdf_093', 'decision': 'Rejected',
        'peer_review': 'IEEE ICC-ROBINS 2024 proceedings record, title, authors and article proceedings header agree. Publication credibility alone does not resolve the experimental contradictions.',
        'peer_review_urls': ['https://ieeexplore.ieee.org/document/10533980/'],
        'dfdc': 'PDF p. 5 claims 95% DFDC accuracy; p. 6 gives updated aggregate 96% without a dataset-specific mapping. Table 2, p. 6, lists 280,000 DFDC training samples plus 40,000 testing samples, while the prose describes over 280,000 videos without accounting for extracted frames. Crucially, p. 6 also explicitly describes the presented results as hypothetical and requiring actual evaluation.',
        'metric_evidence': [{'page': 5, 'dataset': 'DFDC release unspecified', 'claimed_accuracy': .95}, {'page': 6, 'table': '2/prose', 'dataset': 'DFDC/FakeAVCeleb, mapping unspecified', 'claimed_aggregate_accuracy': .96, 'original_experiment_verified': False}],
        'methodology': 'Describes multimodal face, visual, audio and temporal features with alternative Facenet/ArcFace/VGGFace and MLP/LSTM/RNN choices rather than one fully specified evaluated implementation. No defensible independent DFDC test protocol can be reconstructed.',
        'reason': 'Fails genuine original-experiment and research-quality gates. The paper’s own hypothetical-results statement prevents treating the numerical claims as verified original DFDC measurements.',
        'access': 'OpenAlex closed; no authorized free article PDF located. Access is not the sole exclusion reason.',
        'authorized_pdf_url': None,
    },
    {
        'key': 'pdf_103', 'decision': 'Rejected',
        'peer_review': 'Final IEEE Access volume 14, 131–162 (2026) record and article agree. IEEE Access specifies at least two independent reviewers. DOI contains 2025 and the article states online publication on 29 December 2025; use final volume year 2026.',
        'peer_review_urls': [ACCESS_REVIEW, 'https://ieeexplore.ieee.org/document/11316647/'],
        'dfdc': 'Original author-selected full-DFDC subset: 2,500 videos (1,500 fake/1,000 real), 50,000 frames, about 27,000 face samples, then 80/20 face-sample train/test (21,600/5,400); source-video grouping is not reported (PDF pp. 9–10).',
        'metric_evidence': [{'page': 15, 'dataset': 'author-selected full-DFDC subset', 'model': 'DenseNet201 features + Random Forest', 'claimed_accuracy': .991, 'claimed_precision': .9718, 'claimed_recall': .984, 'claimed_f1': .991, 'confusion_counts': {'tp': 3168, 'tn': 2109, 'fp': 52, 'fn': 95}, 'accuracy_computed_from_counts': (3168+2109)/(3168+2109+52+95)}],
        'methodology': 'Five CNN feature extractors are paired separately with five classical classifiers; this is not evidence of a single five-CNN ensemble. LIME/SHAP are post-hoc. Reports CPU i7-1355U, batch 64 and 100 epochs; no GPU is established.',
        'reason': 'Fails research-quality gate. On p. 15 the Random Forest confusion counts sum to 5,424 rather than the stated 5,400 test faces and imply 97.29%, not 99.10% accuracy. Stated F1=99.1% exceeds both corresponding precision=97.18% and recall=98.4%; it cannot be their harmonic mean. Table 7 (p. 25) further differs from the prose. Page 10 says standardization statistics use all 27,000 samples; implemented as described, this leaks test-distribution information. This is an inference from the stated procedure, not a code-level verification. Frame/face splitting without documented source grouping creates an additional leakage risk, not proof of label leakage.',
        'access': 'IEEE Access article is gold OA; Crossref records a CC BY-NC-ND 4.0 version-of-record license and OpenAlex identifies an OA publisher location. Publisher PDF retrieval was not validated after the quality failure; no new download. An OA label does not cure the methodological exclusion.',
        'authorized_pdf_url': None,
        'authorized_oa_landing_url': 'https://ieeexplore.ieee.org/document/11316647/',
    },
    {
        'key': 'pdf_104', 'decision': 'Needs Manual Review',
        'peer_review': 'Final IEEE I-SMAC 2025 proceedings record and article header agree; DOI deposit occurred later than the 2025 conference. A separate conference review-policy record was not established.',
        'peer_review_urls': ['https://ieeexplore.ieee.org/document/11597656/'],
        'dfdc': 'Table 3, PDF p. 4, reports DFDC AUC 92.7% (0.927), baseline CNN 89.3% and ViT-B/16 90.6%. The separate 98.1% AUC is FaceForensics++, not DFDC. No DFDC accuracy is established.',
        'metric_evidence': [{'page': 4, 'table': '3', 'dataset': 'DFDC release unspecified', 'auc': .927}],
        'methodology': 'Artifact/face features, semantic verification using GPT-4V/LLaVA descriptions, FaceNet-based identity checks, and a logistic meta-classifier with calibration. The artifact backbone is presented through alternatives such as Xception/ConvNeXt rather than one fixed reproducible model.',
        'reason': 'DFDC is one small table row among six benchmarks. Release, sample volume, split, training/evaluation unit, semantic prompts/configuration and compute are insufficiently documented for confident substantial-DFDC inclusion. Quality and centrality require review; no free full text was located.',
        'access': 'OpenAlex closed with no repository full text. No verified authorized free PDF URL.',
        'authorized_pdf_url': None,
    },
]


def normalized(title):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', title).lower())


rows = []
for row in curated:
    key = row['key']
    if key in metadata:
        cr = metadata[key]['crossref']
        row.update({
            'title': cr['title'][0],
            'authors': [(a.get('given', '') + ' ' + a.get('family', '')).strip() for a in cr['author']],
            'year': cr['published']['date-parts'][0][0],
            'doi': metadata[key]['doi'], 'venue': cr['container-title'][0].replace('\ufffd\xa0', ' '),
            'publication_url': cr['resource']['primary']['URL'],
            'crossref_url': 'https://api.crossref.org/works/' + metadata[key]['doi'],
            'openalex_access': metadata[key]['openalex']['open_access'],
            'openalex_is_retracted': metadata[key]['openalex'].get('is_retracted'),
            'crossref_update_to': cr.get('update-to', []),
        })
    pdf = pdfs[key]
    row['supplied_pdf'] = {k: pdf[k] for k in ['source', 'sha256', 'bytes', 'pages', 'all_pages_parsed', 'sparse_text_pages']}
    matches = [r['Paper No'] for r in master if normalized(r['Title']) == normalized(row['title'])]
    assert not matches, (key, matches)
    near = sorted(master, key=lambda r: SequenceMatcher(None, normalized(row['title']), normalized(r['Title'])).ratio(), reverse=True)[:3]
    row['deduplication'] = {'exact_normalized_title_matches': matches, 'nearest_existing_title_ids': [r['Paper No'] for r in near],
                            'outcome': 'No matching contribution found among the 58 original records after title, metadata, architecture and experiment checks; nearest titles are not sufficient evidence of duplication.'}
    row['new_paper_id'] = None
    row['new_download_path'] = None
    rows.append(row)

counts = Counter(r['decision'] for r in rows)
assert counts == {'Rejected': 4, 'Needs Manual Review': 3, 'Qualified but Inaccessible': 2}
assert validation['structural_arithmetic_status'] == 'PASS'
result = {
    'review_completed_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Only nine previously included, not-indexed candidate publications; no broad literature expansion or import of experimental variant rows.',
    'counts': {k: counts[k] for k in ['Accepted', 'Rejected', 'Duplicate', 'Needs Manual Review', 'Qualified but Inaccessible']},
    'new_paper_ids': [], 'accepted_pdf_downloads': [], 'source_workbook_changed': False,
    'decision_precedence': 'Reject definite failed mandatory gates; manual review for ambiguous research evidence; qualified-but-inaccessible only when the research criteria pass and free access is the remaining failure.',
    'access_search_limit': 'No free full text located means not verified in this audit, not proof that no authorized copy exists anywhere. Local archive possession does not establish OA.',
    'correction_retraction_limit': 'Eight DOI records had no OpenAlex retraction flag or Crossref update-to correction/retraction records in the retrieved snapshots. This is not an exhaustive investigation or a scientific endorsement. Quality exclusions rely on the supplied article evidence, not venue accusations.',
    'candidates': rows,
}
out = PROJECT / '../research/analyses/DFDC_Literature_Expansion_Decisions.json'
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

lines = [
    '# DFDC Literature Expansion Audit', '',
    f"Review completed: {result['review_completed_utc']} (UTC).", '',
    '**Outcome: no additions.** All nine previously unindexed candidates were independently screened: **0 Accepted, 4 Rejected, 0 Duplicate, 3 Needs Manual Review, 2 Qualified but Inaccessible**. The authoritative workbook remains **58 records, IDs 1–58, with no new IDs and zero sheets changed**. This follows the explicit instruction to preserve the original if no candidate satisfies every gate.', '',
    '## Scope and evidence', '',
    'The prior `DFDC_Research_Comparison.xlsx` and its source register supplied candidate identifiers, not acceptance evidence. Only the nine previously Included / Not indexed publications were reconsidered. Experimental variants were not imported as papers; no unrelated search, model implementation, training, checkpoint acquisition or dataset download occurred.', '',
    'Bibliographic identity was checked against publisher/proceedings records, Crossref and the supplied complete papers; access checks also used OpenAlex and targeted author/institutional searches. A DOI or a local copy alone was never sufficient. PDF page references below are physical pages, including repository covers. Reported measurements are distinguished from calculations and judgments. No new free full-text download was authorized by all gates, so no PDF was added to `papers/`.', '',
              'All task outputs and backups are on D:. Conservative cumulative phase storage, including existing results/tools/tests/docs from the preceding research phase and all literature backups, was about **42.91 MiB** at final review, below the **30 GiB** allocation. Existing input archives, source PDFs and the pre-existing environment are not charged as new downloads. No model or dataset payload was acquired.', '',
    '## Candidate decisions', '',
    '| Audit ID | Publication | Year | Decision |',
    '|---|---|---:|---|',
]
for r in rows:
    lines.append(f"| {r['key']} | [{r['title']}]({r['publication_url']}) | {r['year']} | {r['decision']} |")
for r in rows:
    lines.extend(['', f"### {r['key']} — {r['title']}", '',
                  f"**Authors:** {'; '.join(r['authors'])}. **Venue:** {r['venue']}. **Year:** {r['year']}. "
                  + (f"**DOI:** [{r['doi']}](https://doi.org/{r['doi']})." if r['doi'] else '**DOI:** not established.'), '',
                  '**Publication/peer review:** ' + r['peer_review'] + ' ' + ' '.join(f'[Primary evidence {j+1}]({url}).' for j, url in enumerate(r['peer_review_urls'])), '',
                  '**Original DFDC evidence:** ' + r['dfdc'], '',
                  '**Method and reporting limits:** ' + r['methodology'], '',
                  '**Access:** ' + r['access']])
    if r.get('access_urls'):
        lines.append(' '.join(f'[Access check {j+1}]({url}).' for j, url in enumerate(r['access_urls'])))
    if r.get('authorized_pdf_url'):
        lines.append(f"Official free PDF link: [repository PDF]({r['authorized_pdf_url']}); not newly downloaded because the peer-review gate failed.")
    lines.extend(['', f"**Decision: {r['decision']}.** {r['reason']}"])

lines.extend(['', '## PDF inventory and access provenance', '',
              'Accepted download inventory: **empty**. No existing PDF was overwritten or renamed. The 58 loose PDFs in the existing paper archive were left in place. The nine supplied candidates were freshly read from the existing archive, hashed and parsed with strict PDF parsing; all pages parsed, and methodology/results text was present. Representative title/result pages were also rendered and inspected, including the contradictory figures. This validates the supplied documents for evidence review; it does **not** certify them as newly acquired authorized OA copies.', '',
              f"Existing source archive: `{ROOT / 'sources/Additional_Papers.zip'}`. In the following inventory, paths are ZIP member paths under `Paywal papers/`, not newly created loose PDFs. Full archive-plus-member paths and byte sizes are in [the decision inventory](../../research/analyses/DFDC_Literature_Expansion_Decisions.json).", '',
              '| Audit ID | Pages | SHA-256 | Existing archive member |', '|---|---:|---|---|'])
for r in rows:
    pdf = r['supplied_pdf']
    lines.append(f"| {r['key']} | {pdf['pages']} | `{pdf['sha256']}` | `{pdf['source'].split('!', 1)[1]}` |")
lines.extend(['', 'The 104-page thesis contains 12 sparse/blank appendix pages; all 104 pages parsed, and the substantive thesis, methodology, results and references are present. Sparse pages were not confused with a truncated download.', '',
              'OpenAlex classified seven DOI candidates as closed and the IEEE Access article as gold OA. The university thesis is independently free through its repository. For the closed papers, “not located” is a bounded search result, not proof that no lawful manuscript exists anywhere. Official IEEE similarity-checking/staging links were not treated as public-access permission. No authentication, password or publisher challenge was bypassed. Direct manuscript/login and metadata snapshots are retained in `../research/working/expansion/`.', '',
              'Eight DOI records showed no OpenAlex retraction flag or Crossref `update-to` notice in the retrieved snapshots. This is a limited metadata check, not a guarantee that all corrections have been found. No exclusion is based solely on citation count or prestige, and no claim of predatory publication, fabrication or plagiarism is made.', '',
              '## Duplicates and versions', '',
              'No candidate is already represented by a matching contribution among the existing 58 records. DOI/publication identity, normalized titles, authors/year, architecture and experiment descriptions were checked; near-title resemblance was not treated as identity. In particular, the VGG16/LIME candidate is different from existing Paper 6’s Xception-based comparison.', '',
              'ExplaNET is a related version family with the earlier [DFP-Net, IJCB 2023](https://doi.org/10.1109/IJCB57857.2023.10448842) (`pdf_083`): same authors, closely matching architecture and identical key DFDC accuracy results; the journal adds F1/analysis. Retain the journal version if access is later resolved; do not count the conference/journal results twice. This related version is outside the nine candidate count and outside the existing 58.', '',
              'The original workbook already contains five repeated normalized-title pairs: **8/42, 9/43, 10/44, 11/45 and 12/46**, leaving 53 distinct normalized titles across 58 records. They were flagged, not removed or renumbered. Candidate Duplicate count remains zero. See the [separate correction log](DFDC_Literature_Correction_Log.md).', '',
              '## Backup and workbook protection', '',
              f"Authoritative workbook: `{validation['source']}`.", '',
              f"Controlling post-save backup: `{validation['backup']}`. Created after the user saved and closed Excel; supersedes the earlier shared-read snapshot. Size: **{validation['source_bytes']:,} bytes**.", '',
              f"Original and backup SHA-256: `{validation['sha256']['source']}`.", '',
              'The backup was the separate review copy. No workbook authoring/export/replacement was performed because no paper was Accepted. Read-only inspection avoided openpyxl’s unsupported-extension save path. Native Excel 16.0 opened the backup read-only, rebuilt calculations, and closed without saving; the original and backup remain byte-identical.', '',
              '## Review of all nine worksheets', '',
              '| Worksheet | Verified original state / dependency | Changes |',
              '|---|---|---|',
              '| Master Literature Database | 58 unique stable IDs, 38 columns; table and filter A1:AL59; numeric fractions and Not Reported conventions preserved | None |',
              '| Summary Statistics | 173 formulas; year/type/media distributions, numerical means and explicit dataset/model/XAI incidence recalculated independently | None |',
              '| Dashboard | 139 formulas, 15 native charts; KPIs and chart data references checked | None |',
              '| Dataset Analysis | 21 categories; counts and associated IDs rebuilt from master Dataset(s); headline averages checked arithmetically | None |',
              '| Model Analysis | 50 categories; counts and IDs rebuilt from explicit Prediction Model(s), following existing convention | None |',
              '| Explainable AI Analysis | 13 categories, 11 used; counts and IDs rebuilt from explicit method names | None |',
              '| Research Gap Analysis | 16 categories, 15 used; frequencies independently counted from unique listed IDs; example IDs checked | None |',
              '| Performance Comparison | All 58 records exactly once; metric/model/dataset fields match master; numeric descending order and competition ranks independently checked | None |',
              '| Insights | All original conclusions reviewed against the unchanged master and derived counts; comparability and attribution issues documented separately | None |', '',
              '## Validation results and limits', '',
              f"**{validation['checks_passed']}/{validation['checks_total']} structural/arithmetic/preservation assertions passed.** [Machine-readable validation](../../research/analyses/DFDC_Literature_Expansion_Validation.json).", '',
              f"All {validation['package_parts']} XLSX package parts are byte-identical to the controlling backup: styles, conditional-format extensions, tables/filters, hyperlinks and drawing parts were retained. The workbook preserves {validation['formula_count']} formulas, {validation['chart_count']} charts and {validation['merged_range_count']} merged ranges. Chart reference sheets resolve; native Excel reported no formula errors, and all 525 stored formula caches agree with native recalculation. Nine representative worksheet regions were rendered and visually reviewed without redesign.", '',
              '| Independent quantity | Result |', '|---|---|',
              '| Paper IDs / new IDs | 1–58 / none |',
              '| Year frequencies | 2022: 3; 2023: 2; 2024: 9; 2025: 32; 2026: 12 |',
              '| Exclusive type frequencies | Research Article 39; Review Article 7; Systematic Review 4; Conference Paper 3; Research/Review 2; Thesis 2; Survey 1 |',
              '| Existing grouped research / review totals | 41 / 13; the two mixed Research/Review records count in both groups |',
              f"| Numeric accuracy / F1 / AUC counts | {validation['numeric_metric_counts']['Accuracy']} / {validation['numeric_metric_counts']['F1']} / {validation['numeric_metric_counts']['AUC']} |",
              f"| Mean stored accuracy / F1 / AUC | {validation['summary']['B10']:.10f} / {validation['summary']['B13']:.6f} / {validation['summary']['B14']:.10f} |",
              '| Highest / lowest numeric accuracy | 0.998 / 0.7803; Papers 32 and 40 tie at the highest value |',
              '| Leading dataset counts | DFDC 6; FaceForensics++ 6 |',
              '| Leading explicit model counts | CNN 22; Vision Transformer 9; LSTM 7 |',
              '| Leading XAI counts | Attention 14; Grad-CAM 11; SHAP 7; LIME 6 |',
              '| Leading coded gap | High Computational Cost 28 |', '',
              '**Scope of the pass: preservation and structural/arithmetic checks passed, with existing content warnings.** Dataset/model/XAI category totals were rebuilt from the stored master terms; gap counts were rebuilt from the listed unique IDs, not independently re-coded from every original publication. This is not a clean scientific bill of health for all 58 sources. Selected source-grounded errors and duplicate titles remain, as documented in the correction log. An additional Insights inconsistency was found: B7 claims an extra range-only accuracy paper, but the current Accuracy column has 36 numeric values and 22 Not Reported cells. Paper 27 retains an older range in Other Metrics and already has an exact numeric headline; it is not an additional range-only record.', '',
              'Dataset averages currently reuse each paper’s single headline accuracy for every named dataset. Thus a mathematically correct “DFDC average” may contain an FF++ metric (Paper 35). The performance table mixes datasets, split protocols, media and review-reported figures. Its existing ranks are descriptive, not evidence of DFDC superiority. Neither those averages nor the Insights wording is scientifically certified by this preservation pass.', '',
              '## Promising excluded publications and resolution', '',
              '- **ConvNext-PNet**: obtain an authorized, publicly free full manuscript matching the IJCB paper, then recheck its version and completeness before inclusion.',
              '- **ExplaNET**: obtain a legitimately free accepted/published manuscript; the official manuscript link currently requires login. Preserve the DFP-Net relationship when assigning a future single contribution ID.',
              '- **VGG16/LIME, Combating Deepfakes and Semantic Forensics**: need evidence clarifying DFDC centrality, release, actual sample counts and independent evaluation protocol, as well as free full text and any remaining review-process uncertainty.', '',
              'No access request or message was sent to authors. The workbook was not populated from abstracts. The evidence, exclusion decisions, correction proposals and verification records complete this literature-integrity pass; no accepted-paper integration was warranted.',
])
(PROJECT / '../research/reports/DFDC_Literature_Expansion_Audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'decisions': result['counts'], 'workbook_changed': False, 'new_ids': [], 'decision_inventory': str(out)}, indent=2))
