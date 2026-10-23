"""Rank completed models only on a common validation dataset and evaluation unit."""
import copy
import json
from pathlib import Path
from config import PROJECT_ROOT
from inference.contracts import checkpoint_digest
from inference.unified import available_models, ResearchPredictor
from inference.media import MediaPredictor
from explainability.api import compatible_methods
from inference.demo_candidate import current_image_candidates


ARCHITECTURE_NAMES = {
    'resnet18': 'ResNet-18',
    'efficientnet_b0': 'EfficientNet-B0',
    'deit_tiny_patch16_224': 'DeiT-Tiny',
    'mobilenet_v3_large': 'MobileNet-V3-Large',
}
MODEL_TAGS = {
    'ft': 'continued fine-tuning',
    'v': 'baseline pipeline',
    'f': 'face detection + alignment',
    'b': 'class-balanced sampling',
    'a': 'training augmentation',
    'j': 'JPEG augmentation',
    'img12k': 'trained on 12,000 face images',
    'img100k': 'trained on 100,000 face images',
}


def architecture_name(record):
    """Name constituent modules and distinguish feature fusion from score fusion."""
    config = record['config']
    spec = config.get('model_spec', {})
    backbones = config.get('backbones') or ([spec['backbone']] if spec.get('backbone') else [])
    if record['id'] == 'S0':
        backbones = ['resnet18']
    names = [ARCHITECTURE_NAMES.get(name, name) for name in backbones]
    if spec.get('attention', 'none') != 'none':
        names.append(spec['attention'].upper())
    if spec.get('fusion') == 'frequency':
        names.append('FFT features')
    if config.get('kind') in {'temporal', 'score_fusion'}:
        names.append('LSTM')
    name = ' + '.join(names) or config['model']
    if config.get('kind') == 'score_fusion':
        name += ' (score fusion)'
    return name


def model_tags(record):
    config = record['config']
    augmentation = config.get('augmentation', 'none')
    tags = []
    if config.get('continued_finetuning'):
        tags.append('ft')
    if config.get('preprocess') == 'face':
        tags.append('f')
    if config.get('balance'):
        tags.append('b')
    if config.get('augment') or augmentation not in {'none', None}:
        tags.append('a')
    if augmentation == 'jpeg':
        tags.append('j')
    if not tags:
        tags.append('v')
    count = record.get('training_image_count')
    if count:
        tags.append(f'img{count // 1000}k')
    return tags


def display_name(record):
    """Presentation only: derive tags from the recorded implementation config."""
    return f"{architecture_name(record)} [{', '.join(model_tags(record))}]"


def model_tag_legend(records):
    used = {tag for record in records for tag in model_tags(record)}
    return 'Tags: ' + ' · '.join(f'**{tag}** = {meaning}' for tag, meaning in MODEL_TAGS.items() if tag in used)


def model_description(record):
    config = record['config']
    backbones = config.get('backbones', [])
    if config.get('continued_finetuning'):
        name = architecture_name(record)
        layers = 'final residual stage' if 'resnet18' in backbones else 'final two feature stages'
        return f'{name} continued from its existing image checkpoint; the {layers} and classifier were fine-tuned on 100,000 images. Full-frame RGB input at 224 pixels; fixed threshold 0.5.'
    if config.get('kind') == 'temporal':
        return 'EfficientNet-B0 extracts features from eight sampled frames; a one-layer LSTM classifies the sequence. The final score comes from the LSTM head.'
    if config.get('kind') == 'score_fusion':
        spatial = ('EfficientNet-B0 + DeiT-Tiny features feed a spatial classifier; the LSTM uses EfficientNet-B0 features only.'
                   if len(backbones) > 1 else 'EfficientNet-B0 features feed a spatial classifier and a separate one-layer LSTM classifier.')
        weight = config['spatial_weight']
        return f'{spatial} Their probabilities are combined: {weight:.0%} spatial + {1-weight:.0%} LSTM, before calibration.'
    if len(backbones) > 1:
        return 'EfficientNet-B0 and DeiT-Tiny features are concatenated into one spatial classifier (feature fusion).'
    name = architecture_name(record)
    video = ' Video scores aggregate its eight sampled frame predictions.' if record.get('evaluation_unit') == 'video' else ''
    if 'deit_tiny_patch16_224' in backbones:
        return f'{name} is a vision transformer.' + video
    return f'{name} is a convolutional classifier.' + video


def complete_metrics(metrics):
    result = dict(metrics)
    matrix = result.get('confusion_matrix_real_fake')
    if matrix:
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        result.setdefault('macro_f1', ((2*tn/max(2*tn+fp+fn, 1))+(2*tp/max(2*tp+fp+fn, 1)))/2)
        result.setdefault('real_recall', tn/max(tn+fp, 1))
        result.setdefault('fake_recall', tp/max(tp+fn, 1))
        result.setdefault('samples', tn+fp+fn+tp)
    return result


def family(record):
    identifier = record['id']
    names = record['config'].get('backbones', [])
    if identifier in {'S6', 'S7', 'S8'}:
        return 'Spatial + Temporal Hybrid'
    if len(names) > 1:
        return 'CNN + ViT Hybrid'
    if 'deit_tiny_patch16_224' in names:
        return 'Vision Transformer'
    return 'EfficientNet' if 'efficientnet' in record['config']['model'].lower() else 'ResNet / CNN'


def catalog(modality='Image', include_experimental=False):
    if modality not in {'Image', 'Video'}:
        raise ValueError('Unsupported modality')
    unified = available_models()
    rows = []
    if modality == 'Video':
        for original in unified:
            record = copy.deepcopy(original)
            record.update(source='unified', ranked=True, evaluation_unit='video',
                          evaluation_protocol='DFDC source-grouped validation; normalized8; threshold 0.5',
                          training_dataset='Real vs Fake Faces' if record['id'] == 'S0' else 'Local source-grouped DFDC training videos',
                          evaluation_dataset='Local DFDC validation: 175 videos, 17 source groups')
            if record['id'] == 'S0':
                record['training_image_count'] = 100000
            rows.append(record)
    else:
        base = PROJECT_ROOT/'results/execution_20261019'
        for name in ['E00_validation', 'E01_resnet18', 'E02_efficientnet_b0', 'E03_attention', 'E04_jpeg']:
            path = base/name/'summary.json'
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding='utf-8'))
            checkpoint = Path(summary['checkpoint'])
            if not summary.get('completed') or summary.get('smoke_test') or not checkpoint.is_file():
                continue
            if checkpoint_digest(checkpoint) != summary['checkpoint_sha256']:
                continue
            identifier = 'S0' if name == 'E00_validation' else name
            spec = summary['model']
            architecture = 'Preserved ResNet-18' if identifier == 'S0' else spec['backbone'].replace('_', '-')
            if spec.get('attention', 'none') != 'none':
                architecture += ' + '+spec['attention'].upper()
            if name == 'E04_jpeg':
                architecture += ' + JPEG augmentation'
            rows.append({'id': identifier, 'source': 'unified' if identifier == 'S0' else 'image_transfer',
                         'config': {'model': architecture, 'preprocess': 'full frame', 'backbones': [spec['backbone']], 'model_spec': spec,
                                    'augmentation': summary.get('config', {}).get('augmentation', 'none')},
                         'checkpoint': str(checkpoint), 'checkpoint_sha256': summary['checkpoint_sha256'],
                         'validation': complete_metrics(summary['validation']), 'calibration': {'temperature': 1., 'scope': 'uncalibrated images'},
                         'ranked': True, 'evaluation_unit': 'image',
                         'evaluation_protocol': 'Real vs Fake Faces validation; '+summary['manifest_sha256'],
                         'evaluation_dataset': 'Real vs Fake Faces: common 2,000-image validation subset',
                         'training_dataset': 'Real vs Fake Faces: 100,000 training images' if identifier == 'S0' else 'Real vs Fake Faces: 12,000 training images',
                         'training_image_count': 100000 if identifier == 'S0' else 12000,
                         'limitations': 'Training budgets differ; common evaluation does not make training a controlled comparison.'})
        benchmark_path = PROJECT_ROOT/'results/debug_refinement_20261019/image_transfer_benchmark.json'
        if benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
            source = next((record for record in unified if record['id'] == 'S2'), None)
            predictions = Path(benchmark['predictions_path'])
            if source and benchmark.get('completed') and source['checkpoint_sha256'] == benchmark['checkpoint_sha256'] and predictions.is_file() and checkpoint_digest(predictions) == benchmark['predictions_sha256']:
                record = copy.deepcopy(source)
                record['config']['model'] += ' (DFDC trained)'
                record.update(source='unified',ranked=True,evaluation_unit='image',validation=benchmark['validation'],
                              evaluation_protocol='Real vs Fake Faces validation; '+benchmark['manifest_sha256'],
                              evaluation_dataset='Real vs Fake Faces: common 2,000-image validation subset; native DFDC alignment',
                              training_dataset='Local source-grouped DFDC training videos',calibration={'temperature':1.,'scope':'uncalibrated images'},
                              limitations='Image transfer measured separately; validation does not establish broad GAN generalization.')
                rows.append(record)
        if include_experimental:
            for original in unified:
                if original['id'] in {record['id'] for record in rows}:
                    continue
                record = copy.deepcopy(original)
                record.update(source='unified', ranked=False, evaluation_unit='image', validation={},
                              evaluation_protocol=None, evaluation_dataset='Image transfer not benchmarked',
                              training_dataset='Local source-grouped DFDC training videos')
                rows.append(record)
    if modality == 'Image':
        overrides = {record['id']: record for record in current_image_candidates()}
        rows = [record for record in rows if record['id'] not in overrides]
        rows.extend(overrides.values())
    for record in rows:
        record['family'] = family(record)
        record['validation'] = complete_metrics(record['validation'])
    return sorted(rows, key=ranking_key, reverse=True)


def ranking_key(record):
    metrics = record.get('validation', {}) if record.get('ranked') else {}
    return tuple(metrics.get(key, -1) for key in ['balanced_accuracy', 'macro_f1', 'roc_auc'])


def grouped(records):
    groups = {}
    for record in records:
        groups.setdefault(record['family'], []).append(record)
    result = []
    for name, variants in groups.items():
        variants = sorted(variants, key=ranking_key, reverse=True)
        best = variants[0]
        comparable = [row for row in variants if row['ranked'] and row['evaluation_protocol'] == best['evaluation_protocol']]
        result.append({'family': name, 'variants': variants, 'best': best,
                       'mean_balanced_accuracy': sum(row['validation']['balanced_accuracy'] for row in comparable)/len(comparable) if comparable else None,
                       'comparable_runs': len(comparable)})
    return sorted(result, key=lambda group: ranking_key(group['best']), reverse=True)


def methods_for(record):
    names = record['config'].get('backbones', [])
    return compatible_methods('deit_tiny_patch16_224' not in names or 'efficientnet_b0' in names, 'deit_tiny_patch16_224' in names)


def create_predictor(record):
    return ResearchPredictor(record['id']) if record['source'] == 'unified' else MediaPredictor(record['checkpoint'])
