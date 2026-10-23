"""Publish current image-demo plots from saved scores; never run model inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

from config import PROJECT_ROOT

RUN = PROJECT_ROOT/'results/demo_accuracy_20261021_040741'
OUTPUT = PROJECT_ROOT/'docs/figures/demo_accuracy_20261021'
COLORS = ['#65758b', '#007f80']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def snapshot(run):
    status = read(run/'final_status.json')
    assert status['completed'] and status['test_evaluated']
    sources = [run/'final_status.json', run/'baseline_validation_fp32.json', run/'history.json',
               PROJECT_ROOT/'outputs/training_history.csv',
               PROJECT_ROOT/'results/execution_20261019/E00_test/metrics.json']
    with sources[3].open(newline='', encoding='utf-8') as stream:
        original = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    continuation = read(sources[2])
    history = [{'epoch': int(row['epoch']), 'train_accuracy': row['train_accuracy'],
                'validation_accuracy': row['valid_accuracy'], 'train_loss': row['train_loss'],
                'validation_loss': row['valid_loss']} for row in original]
    history += [{'epoch': row['original_epoch'], 'train_accuracy': row['train_accuracy'],
                 'validation_accuracy': row['validation']['accuracy'], 'train_loss': row['train_loss'],
                 'validation_loss': row['validation']['loss']} for row in continuation]
    metrics = {'original_validation': read(sources[1]), 'continued_validation': status['validation'],
               'original_test': read(sources[4]), 'continued_test': status['test']}
    sources += [run/'deployment.json', PROJECT_ROOT/'results/execution_20261019/E00_validation/summary.json']
    common = {'continued': read(sources[-2])['record']['validation'],
              'original': read(sources[-1])['validation']}
    curves = {}
    for split in ['validation', 'test']:
        path = run/f'candidate_{split}_fp32.csv'
        sources.append(path)
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        truth = np.array([row['label']=='FAKE' for row in rows], dtype=int)
        score = np.array([float(row['fake_probability']) for row in rows])
        expected = status[split]
        assert len(rows) == expected['samples'] == 20000
        assert confusion_matrix(truth, score >= .5, labels=[0, 1]).tolist() == expected['confusion_matrix_real_fake']
        assert np.isclose(np.mean(truth == (score >= .5)), expected['accuracy'])
        assert np.isclose(roc_auc_score(truth, score), expected['roc_auc'])
        fpr, tpr, _ = roc_curve(truth, score)
        precision, recall, _ = precision_recall_curve(truth, score)
        def points(x, y):
            indices = np.unique(np.linspace(0, len(x)-1, min(len(x), 1200), dtype=int))
            return {'x': x[indices].tolist(), 'y': y[indices].tolist()}
        curves[split] = {'roc': points(fpr, tpr), 'precision_recall': points(recall, precision)}
    from inference.catalog import catalog, grouped, display_name
    families = [{'family': group['family'], 'best': display_name(group['best']),
                 'best_balanced_accuracy': group['best']['validation']['balanced_accuracy'],
                 'mean_balanced_accuracy': group['mean_balanced_accuracy'], 'runs': group['comparable_runs']}
                for group in grouped(catalog('Image'))]
    return {'format': 'image_demo_plot_evidence_v1', 'model': 'ResNet-18 [ft, a, img100k]',
            'checkpoint_sha256': status['checkpoint_sha256'], 'selected_epoch': read(run/'selection.json')['epoch'],
            'metrics': metrics, 'common_validation': common, 'history': history, 'curves': curves, 'image_families': families,
            'curve_sampling': 'At most 1,200 points for rendering; metrics verified on all saved scores.',
            'sources': {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}


def plots(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    metrics = data['metrics']
    def save(fig, name):
        for suffix in ['png', 'svg']:
            path = folder/f'{name}.{suffix}'
            fig.savefig(path, dpi=180, facecolor='white')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    x = np.arange(2)
    for i, model in enumerate(['original', 'continued']):
        values = [metrics[f'{model}_{split}'] for split in ['validation', 'test']]
        bars = axes[0].bar(x+(i-.5)*.34, [m['accuracy']*100 for m in values], .34,
                           color=COLORS[i], label=['Original ResNet-18', 'Continued ResNet-18'][i])
        axes[0].bar_label(bars, labels=[f"{m['accuracy']:.3%}" for m in values], padding=4, fontsize=9)
        errors = [int(m['samples']-round(m['accuracy']*m['samples'])) for m in values]
        bars = axes[1].bar(x+(i-.5)*.34, errors, .34, color=COLORS[i])
        axes[1].bar_label(bars, padding=4)
    for ax in axes:
        ax.set_xticks(x, ['Full validation\n20,000 images', 'Held-out test\n20,000 images'])
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    axes[0].set(ylim=(0, 110), ylabel='Accuracy (%)', title='Accuracy on the same evaluation splits')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=2, frameon=False)
    axes[1].set(ylim=(0, 720), ylabel='Misclassified images', title='Fewer classification errors at threshold 0.5')
    fig.suptitle('Image demo: original vs continued ResNet-18', fontsize=15)
    save(fig, 'accuracy_comparison')

    history = data['history']; epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for key, label, color in [('train', 'Training', '#65758b'), ('validation', 'Full validation', '#007f80')]:
        axes[0].plot(epochs, [row[f'{key}_accuracy'] for row in history], 'o-', label=label, color=color)
        axes[1].plot(epochs, [row[f'{key}_loss'] for row in history], 'o-', label=label, color=color)
    for ax in axes:
        ax.axvline(2.5, color='#9b6124', linestyle='--', linewidth=1, label='Continuation starts')
        ax.set(xlabel='Global epoch', xticks=epochs); ax.grid(alpha=.18); ax.legend(frameon=False)
    axes[0].set(ylabel='Accuracy', ylim=(.91, 1.005), title='Observed learning curve')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].plot(data['selected_epoch'], metrics['continued_validation']['accuracy'], '*', markersize=17,
                 markeredgecolor='#242d38', color='#e4a73a', zorder=4)
    axes[1].set(ylabel='Cross-entropy loss', title='Training and validation loss')
    fig.suptitle('Two original epochs + four continuation epochs; star = selected epoch', fontsize=14)
    save(fig, 'training_curves')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), layout='constrained')
    for ax, key, title in zip(axes.flat,
            ['original_validation', 'continued_validation', 'original_test', 'continued_test'],
            ['Original / full validation', 'Continued / full validation', 'Original / recorded held-out test', 'Continued / final held-out test']):
        m = metrics[key]; matrix = np.array(m.get('confusion_matrix_real_fake', m.get('confusion_matrix')))
        image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=10000)
        for (row, column), count in np.ndenumerate(matrix):
            ax.text(column, row, f'{count:,}\n({count/matrix[row].sum():.2%} of class)',
                    ha='center', va='center', color='white' if count > 5000 else '#172330', fontsize=11)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
               xlabel='Predicted class', ylabel='Actual class', title=f"{title}\nAccuracy {m['accuracy']:.3%}")
    fig.colorbar(image, ax=axes, shrink=.7, label='Image count (shared scale)')
    fig.suptitle('Confusion matrices: 10,000 REAL + 10,000 FAKE in each split', fontsize=14)
    save(fig, 'confusion_matrices')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for split, color in [('validation', '#007f80'), ('test', '#a24d82')]:
        m = metrics[f'continued_{split}']; curve = data['curves'][split]
        label = 'Full validation' if split=='validation' else 'Held-out test'
        axes[0].plot(curve['roc']['x'], curve['roc']['y'], color=color, label=f"{label}: AUC {m['roc_auc']:.6f}")
        axes[1].plot(curve['precision_recall']['x'], curve['precision_recall']['y'], color=color,
                     label=f"{label}: PR-AUC {m['pr_auc']:.6f}")
        # Fixed-threshold points connect these curves to the displayed classifier.
        matrix = np.array(m['confusion_matrix_real_fake'])
        axes[0].plot(matrix[0, 1]/matrix[0].sum(), m['recall'], 'o', color=color)
        axes[1].plot(m['recall'], m['precision'], 'o', color=color)
    axes[0].plot([0, 1], [0, 1], '--', color='#9a9a9a', linewidth=1)
    axes[1].axhline(.5, linestyle='--', color='#9a9a9a', linewidth=1)
    axes[0].set(xlabel='False-positive rate (REAL called FAKE)', ylabel='True-positive rate (FAKE recall)', title='ROC: separation across thresholds')
    axes[1].set(xlabel='FAKE recall', ylabel='FAKE precision', title='Precision–recall: FAKE detection trade-off')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.025)); ax.grid(alpha=.18); ax.legend(loc='lower left', frameon=False, fontsize=9)
    for index, key, limits in [(0, 'roc', (0, .05, .95, 1.001)),
                               (1, 'precision_recall', (.95, 1, .95, 1.001))]:
        zoom = axes[index].inset_axes([.38, .30, .55, .43])
        for split, color in [('validation', '#007f80'), ('test', '#a24d82')]:
            curve = data['curves'][split][key]
            zoom.plot(curve['x'], curve['y'], color=color, linewidth=1.3)
        zoom.set(xlim=limits[:2], ylim=limits[2:], title='Detail (zoomed axes)')
        zoom.title.set_fontsize(9); zoom.tick_params(labelsize=8); zoom.grid(alpha=.18)
    fig.suptitle('Continued ResNet-18: curves from saved predictions; dots = fixed threshold 0.5', fontsize=14)
    save(fig, 'roc_precision_recall')


def publish(data, folder):
    base = folder.relative_to(PROJECT_ROOT).as_posix()
    m = data['metrics']
    rows = []
    for key, label in [('original_validation', 'Original / full validation'), ('continued_validation', 'Continued / full validation'),
                       ('original_test', 'Original / recorded held-out test'), ('continued_test', 'Continued / final held-out test')]:
        value = m[key]
        rows.append(f"| {label} | {value['samples']:,} | {value['accuracy']:.3%} | {value['precision']:.3%} | {value['recall']:.3%} | {value.get('f1',value.get('f1_score')):.5f} |")
    descriptions = [
        ('accuracy_comparison', 'Accuracy and error comparison', 'Why: quantify the improvement on equal evaluation splits. The left panel shows accuracy on a zero-based scale; the right panel shows the corresponding number of errors. Each split contains 20,000 images.'),
        ('training_curves', 'Training and validation curves', 'Why: show how continued training changed accuracy and loss. The dashed line separates the original two epochs from four continuation epochs; the star marks the validation-selected epoch. Training accuracy includes stochastic augmentation. No test scores selected the epoch.'),
        ('confusion_matrices', 'Confusion matrices', 'Why: show which class errors remain. Rows are actual labels and columns are predictions; off-diagonal cells are mistakes. Counts and within-class percentages use a shared color scale. The original test matrix is previously recorded evidence.'),
        ('roc_precision_recall', 'ROC and precision–recall', 'Why: demonstrate separation and the precision/recall trade-off across thresholds. FAKE is positive; dots mark the deployed threshold of 0.5. Dashed references show chance ROC and the 50% FAKE prevalence. Insets magnify the near-perfect region with explicitly zoomed axes. These are descriptive plots from saved scores; no threshold was selected on test data.')]
    section = '## Demo image continuation — 21 October 2026\n\n'
    section += f"The active image model is **{data['model']}** (`ft` = continued fine-tuning, `a` = training augmentation, `img100k` = 100,000 training images). Full validation accuracy is **{m['continued_validation']['accuracy']:.3%}** and held-out test accuracy is **{m['continued_test']['accuracy']:.3%}**. Accuracy equals balanced accuracy here because both classes have 10,000 images per split.\n\n"
    section += '| Model / split | Images | Accuracy | FAKE precision | FAKE recall | FAKE F1 |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    original_matrix = m['original_test']['confusion_matrix']
    current_matrix = m['continued_test']['confusion_matrix_real_fake']
    section += f"On the held-out test, missed FAKE images decreased from **{original_matrix[1][0]} to {current_matrix[1][0]}**, while REAL images incorrectly flagged FAKE increased from **{original_matrix[0][1]} to {current_matrix[0][1]}**. The confusion matrices below expose this trade-off.\n\n"
    common = data['common_validation']
    section += f"The original held-out result was recorded before this continuation. The candidate was selected using complete validation results and evaluated once on the held-out test set. The shared 2,000-image comparison cohort is separate: the candidate scores **{common['continued']['accuracy']:.2%}**, versus **{common['original']['accuracy']:.2%}** for the original image model. The UI labels these evaluation scopes explicitly.\n\n"
    for name, title, caption in descriptions:
        section += f'### {title}\n\n![{title}]({base}/{name}.png)\n\n{caption}\n\n'
    section += f'PNG and editable SVG figures, compact metrics, source hashes and regeneration instructions are in [the plot evidence folder]({base}/README.md). The complete local execution report is [here](results/demo_accuracy_20261021_040741/REPORT.md). These within-dataset scores do not establish performance on arbitrary internet photos. Classification heatmaps show model influence, not verified edit boundaries.\n\n'
    section += 'Activation uses `configs/runtime/active_image_model.json`; original weights are preserved. Restore the original demo with `powershell -ExecutionPolicy Bypass -File "D:\\sundarCollege\\Deepfake_Detect\\Deepfakereserch\\rollback_demo.ps1"`. The historical video benchmark below retains its original results. Model weights, rollback backups and the machine-local activation pointer are excluded from Git; a fresh checkout needs these assets provisioned before the improved model can load.\n\n'
    readme = PROJECT_ROOT/'README.md'
    text = readme.read_text(encoding='utf-8')
    start = text.index('## Demo image continuation — 21 October 2026')
    end = text.index('Current debugging, XAI, latency', start)
    text = text[:start]+section+text[end:]
    heading = ('### Current image-family ranking (common 2,000-image validation cohort)'
               if '### Current image-family ranking' in text else
               '### Image-family ranking (common 2,000-image validation cohort)')
    start = text.index(heading); end = text.index('### Video-family ranking', start)
    family_table = '### Current image-family ranking (common 2,000-image validation cohort)\n\nUpdated after the 21 October continuation. Full 20,000-image metrics above are a separate scope.\n\n| Family | Best variant | Best validation BA | Mean validation BA | Comparable runs |\n|---|---|---:|---:|---:|\n'
    for group in data['image_families']:
        family_table += f"| {group['family']} | {group['best']} | {group['best_balanced_accuracy']:.2%} | {group['mean_balanced_accuracy']:.2%} | {group['runs']} |\n"
    text = text[:start]+family_table+'\n'+text[end:]
    text = text.replace('S0 remains the image default and S6 the video default.', 'The continued ResNet-18 is the current image default; S0 remains available as the original image baseline. S6 is the video default.')
    readme.write_text(text, encoding='utf-8')
    notes = '# Image accuracy plot evidence — 21 October 2026\n\n'
    notes += 'These aggregate PNG/SVG plots and `metrics.json` are portable documentation assets. No face images, checkpoints or per-image paths are included. The original experiments and their plots remain unchanged.\n\n'
    for name, title, caption in descriptions:
        notes += f'## {title}\n\n{caption}\n\n[PNG]({name}.png) · [SVG]({name}.svg)\n\n'
    notes += '## Reproduce\n\nFrom the project root, using existing local experiment JSON/CSV logs:\n\n```powershell\n.\\dfenv\\Scripts\\python.exe -B -m tools.plot_demo_accuracy\n```\n\nTo render the same figures from the compact aggregate snapshot without local experiment files:\n\n```powershell\n.\\dfenv\\Scripts\\python.exe -B -m tools.plot_demo_accuracy --snapshot docs/figures/demo_accuracy_20261021/metrics.json\n```\n\nThis reads saved scores only and does not run inference, retrain, or re-evaluate test images. Each saved score CSV is checked against its recorded sample count, confusion matrix, accuracy and ROC-AUC. Curve rendering uses at most 1,200 sampled points; reported metrics come from all 20,000 scores. Source artifact SHA-256 hashes and the selected checkpoint SHA-256 are recorded in `metrics.json`.\n'
    (folder/'README.md').write_text(notes, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--current', action='store_true', help='Both existing current image identities')
    args = parser.parse_args()
    current = args.current or (not args.snapshot and not args.run and (PROJECT_ROOT/'results/current_image_models/evaluation_complete.json').exists())
    run = args.run or (PROJECT_ROOT/'results/current_image_models' if current else RUN)
    data = read(args.snapshot) if args.snapshot else (current_snapshot(run) if current else snapshot(run))
    current = data.get('format') == 'current_image_plot_evidence_v1'
    folder = (args.output or (PROJECT_ROOT/'docs/figures/current_image_models' if current else OUTPUT)).resolve()
    (current_plots if current else plots)(data, folder)
    (folder/'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    if current:
        publish_current(data, folder, run)
    elif '## Current image models' not in (PROJECT_ROOT/'README.md').read_text(encoding='utf-8'):
        publish(data, folder)
    print(f'Regenerated image plots from saved evidence in {folder}')


def current_snapshot(run):
    """Verify all saved scores, then export a portable aggregate plot snapshot."""
    from training.demo_accuracy import image_metrics, sha
    from training.image_continuation import IDENTITIES
    sources, models = {}, {}
    for identity in IDENTITIES:
        folder = run/identity
        status = read(folder/'final_status.json')
        training = read(folder/'training_finished.json')
        training['checkpoint_bytes'] = status['checkpoint_bytes']
        models[identity] = {'name': 'ResNet-18' if identity == IDENTITIES[0] else 'EfficientNet-B0',
                            'training': training, 'baseline_validation': read(folder/'baseline_validation_fp32.json'),
                            'validation': status['validation'], 'test': status['test'],
                            'checkpoint': status['checkpoint'], 'checkpoint_sha256': status['checkpoint_sha256'],
                            'inference_benchmark': read(folder/'inference_benchmark.json'), 'curves': {}}
        for name in ['final_status.json', 'training_finished.json', 'baseline_validation_fp32.json', 'inference_benchmark.json']:
            sources[(folder/name).relative_to(PROJECT_ROOT).as_posix()] = sha(folder/name)
        for split in ['validation', 'test']:
            path = folder/f'candidate_{split}_fp32.csv'
            sources[path.relative_to(PROJECT_ROOT).as_posix()] = sha(path)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            truth = np.array([int(r['label']=='FAKE') for r in rows])
            scores = np.array([float(r['fake_probability']) for r in rows])
            observed = image_metrics(truth, scores)
            expected = status[split]
            assert len(rows) == expected['samples'] == 20000
            assert sources[path.relative_to(PROJECT_ROOT).as_posix()] == expected['predictions_sha256']
            for key, value in observed.items():
                if isinstance(value, (float, int)):
                    assert np.isclose(value, expected[key], rtol=1e-12, atol=1e-12), (identity, split, key)
            assert observed['classwise'] == expected['classwise']
            assert observed['confusion_matrix_real_fake'] == expected['confusion_matrix_real_fake']
            fpr, tpr, _ = roc_curve(truth, scores)
            precision, recall, _ = precision_recall_curve(truth, scores)
            def points(x, y):
                # Full curve points are retained; figures never drive metrics.
                return {'x': x.tolist(), 'y': y.tolist()}
            models[identity]['curves'][split] = {'roc': points(fpr, tpr), 'precision_recall': points(recall, precision)}
    historical_e02 = read(PROJECT_ROOT/'results/execution_20261019/E02_efficientnet_b0/summary.json')
    historical_resnet_path = PROJECT_ROOT/'results/demo_accuracy_20261021_040741/candidate_test_fp32.json'
    historical_resnet_test = read(historical_resnet_path)
    assert historical_resnet_test['samples'] == 20000
    models[IDENTITIES[0]]['historical_test'] = historical_resnet_test
    sources[historical_resnet_path.relative_to(PROJECT_ROOT).as_posix()] = sha(historical_resnet_path)
    from inference.catalog import catalog, grouped, display_name
    families = [{'family': g['family'], 'name': display_name(g['best']), 'best': g['best']['validation']['balanced_accuracy'],
                 'mean': g['mean_balanced_accuracy'], 'runs': g['comparable_runs']} for g in grouped(catalog('Image'))]
    return {'format': 'current_image_plot_evidence_v1', 'models': models, 'sources': sources,
            'historical_e02': {'training_images': historical_e02['training_images'], 'validation_images': historical_e02['validation_images'],
                               'best_epoch': historical_e02['best_epoch'], 'validation': historical_e02['validation']},
            'families': families, 'threshold': .5, 'positive_class': 'FAKE',
            'test_scope': 'Repeated predefined test split, used historically; no test-based selection',
            'rendering': 'Full ROC/PR points; every metric recomputed from all saved prediction scores'}


def current_plots(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    entries = list(data['models'].values())
    colors = ['#007f80', '#975a91']
    def save(fig, name):
        for suffix in ['png', 'svg']:
            path = folder/f'{name}.{suffix}'
            fig.savefig(path, dpi=180, facecolor='white')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        plt.close(fig)
    for index, entry in enumerate(entries):
        training = entry['training']; history = training['history']
        epochs = [r['epoch'] for r in history]
        prefix = 'resnet18' if index == 0 else 'efficientnet_b0'
        for kind, ylabel in [('accuracy', 'Accuracy'), ('loss', 'Cross-entropy loss')]:
            fig, ax = plt.subplots(figsize=(8.5, 4.5), layout='constrained')
            ax.plot(epochs, [r[f'train_{kind}'] for r in history], 'o-', label='Training (augmented)', color='#65758b')
            ax.plot(epochs, [r['validation'][kind] for r in history], 'o-', label='Full validation (20,000)', color=colors[index])
            selected = training['selected_epoch']
            chosen = next((r for r in history if r['epoch'] == selected), None)
            if chosen:
                ax.plot(selected, chosen['validation'][kind], '*', markersize=17, color='#dc9a23', markeredgecolor='#333333', label=f'Selected epoch {selected}')
            else:
                ax.text(.03, .06, 'Selected: original warm start (stage epoch 0)', transform=ax.transAxes, fontsize=9)
            if index == 0:
                for boundary, label in [(2.5, 'October 21 continuation'), (6.5, 'Current continuation')]:
                    ax.axvline(boundary, color='#a3a3a3', linestyle='--', linewidth=1, label=label)
            if training['early_stopping_triggered']:
                ax.axvline(training['end_epoch'], color='#b34d3d', linestyle=':', label='Early stopping')
            ax.set(xlabel='Global epoch' if index == 0 else 'Full-dataset stage epoch (100,000 training images)',
                   ylabel=ylabel, xticks=epochs, title=f"{entry['name']}: observed training and validation {kind}")
            if kind == 'accuracy':
                observed = [r['train_accuracy'] for r in history]+[r['validation']['accuracy'] for r in history]
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_ylim(max(0, min(observed)-.02), 1.005)
                ax.set_ylabel('Accuracy (zoomed axis)')
            else:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            save(fig, prefix+'_'+kind)
    for split in ['validation', 'test']:
        fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
        keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'fake_recall', 'roc_auc']
        x = np.arange(len(keys))
        for index, entry in enumerate(entries):
            values = [entry[split][key] for key in keys]
            bars = ax.bar(x+(index-.5)*.35, values, .35, color=colors[index], label=entry['name'])
            ax.bar_label(bars, labels=[f'{value:.2%}' for value in values], padding=3, fontsize=8, rotation=30)
        ax.set(xticks=x, xticklabels=['Accuracy', 'Balanced accuracy', 'Macro-F1', 'FAKE recall', 'ROC-AUC'],
               ylim=(0, 1.14), ylabel='Metric', title=f"Full {split}: 20,000 images / FP32 / threshold 0.5")
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc='lower right'); ax.grid(axis='y', alpha=.15)
        save(fig, split+'_comparison')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    for index, entry in enumerate(entries):
        for column, split in enumerate(['validation', 'test']):
            ax = axes[index, column]; matrix = np.array(entry[split]['confusion_matrix_real_fake'])
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=10000)
            for (row, col), value in np.ndenumerate(matrix):
                ax.text(col, row, f'{value:,}\n{value/matrix[row].sum():.2%}', ha='center', va='center',
                        color='white' if value > 5000 else '#172330')
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                   xlabel='Predicted', ylabel='Actual', title=f"{entry['name']} / {split}")
    fig.suptitle('Fixed threshold 0.5; shared color scale; counts and within-class percentages')
    save(fig, 'confusion_matrices')
    for curve_type, filename in [('roc', 'roc_curves'), ('precision_recall', 'precision_recall_curves')]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for ax, split in zip(axes, ['validation', 'test']):
            for index, entry in enumerate(entries):
                curve = entry['curves'][split][curve_type]
                metric = entry[split]['roc_auc' if curve_type == 'roc' else 'pr_auc']
                ax.plot(curve['x'], curve['y'], color=colors[index], label=f"{entry['name']}: {metric:.6f}")
                m = entry[split]
                point = (m['false_positive_rate'], m['fake_recall']) if curve_type == 'roc' else (m['fake_recall'], m['precision'])
                ax.plot(*point, 'o', color=colors[index], markersize=4)
            if curve_type == 'roc':
                ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set(xlabel='False-positive rate', ylabel='FAKE recall')
            else:
                ax.axhline(.5, linestyle='--', color='gray'); ax.set(xlabel='FAKE recall', ylabel='FAKE precision')
            ax.set(xlim=(0, 1), ylim=(0, 1.025), title=f'Full {split} / 20,000 images')
            ax.legend(loc='lower left', fontsize=8); ax.grid(alpha=.2)
            zoom = ax.inset_axes([.42, .30, .54, .42])
            for index, entry in enumerate(entries):
                curve = entry['curves'][split][curve_type]
                zoom.plot(curve['x'], curve['y'], color=colors[index], linewidth=1)
            zoom.set(xlim=(0, .05) if curve_type == 'roc' else (.9, 1), ylim=(.9, 1.001), title='Zoomed axes')
            zoom.title.set_fontsize(8); zoom.tick_params(labelsize=7); zoom.grid(alpha=.2)
        fig.suptitle('FAKE positive; dots = fixed threshold 0.5; curves from frozen predictions')
        save(fig, filename)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for row, split in enumerate(['validation', 'test']):
        for column, cls in enumerate(['REAL', 'FAKE']):
            ax = axes[row, column]; x = np.arange(3)
            for index, entry in enumerate(entries):
                values = [entry[split]['classwise'][cls][key] for key in ['precision', 'recall', 'f1']]
                bars = ax.bar(x+(index-.5)*.35, values, .35, color=colors[index], label=entry['name'])
                ax.bar_label(bars, labels=[f'{v:.2%}' for v in values], padding=3, fontsize=8)
            ax.set(xticks=x, xticklabels=['Precision', 'Recall', 'F1'], ylim=(0, 1.12), title=f'{split} / {cls}')
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc='lower right', fontsize=8)
    save(fig, 'classwise_metrics')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    names = [e['name'] for e in entries]
    for ax, key, divisor, title in [(axes[0], 'parameters', 1e6, 'Total parameters (millions)'),
                                      (axes[1], 'checkpoint_bytes', 1024**2, 'Selected checkpoint size (MiB)')]:
        values = [e['training'][key]/divisor for e in entries]
        bars = ax.bar(names, values, color=colors); ax.bar_label(bars, fmt='%.2f', padding=4)
        ax.set(title=title, ylim=(0, max(values)*1.2))
    save(fig, 'model_resources')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    for ax, batch, key, title in [(axes[0], '1', 'ms_per_image', 'Single-image forward (ms)'),
                                 (axes[1], '32', 'ms_per_image', 'Batch-32 amortized latency (ms/image)'),
                                 (axes[2], '32', 'images_per_second', 'Batch-32 throughput (images/s)')]:
        values = [e['inference_benchmark']['batches'][batch][key] for e in entries]
        bars = ax.bar(names, values, color=colors); ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
        ax.set(title=title, ylim=(0, max(values)*1.2)); ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('FP32 resident input; 10 warmups + 30 synchronized repeats; excludes preprocessing and XAI')
    save(fig, 'inference_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, entry in zip(axes, entries):
        values = [entry['baseline_validation']['accuracy'], entry['validation']['accuracy']]
        bars = ax.bar(['Existing checkpoint', 'Selected checkpoint'], values, color=['#65758b', '#007f80'])
        ax.bar_label(bars, labels=[f'{value:.3%}' for value in values], padding=4)
        ax.set(ylim=(0, 1.12), ylabel='Accuracy', title=entry['name']+' / same full validation')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Both baselines and selections evaluated on the same 20,000 validation images')
    save(fig, 'validation_improvement')
    historical = entries[0].get('historical_test')
    if historical:
        fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
        keys = ['accuracy', 'balanced_accuracy', 'fake_recall', 'macro_f1', 'roc_auc']
        x = np.arange(len(keys))
        for index, (label, metric) in enumerate([('Recorded epoch-6 checkpoint', historical),
                                                ('Current frozen checkpoint', entries[0]['test'])]):
            values = [metric[key] for key in keys]
            bars = ax.bar(x+(index-.5)*.35, values, .35, label=label, color=['#65758b', '#007f80'][index])
            ax.bar_label(bars, labels=[f'{v:.3%}' for v in values], padding=3, fontsize=8, rotation=30)
        ax.set(xticks=x, xticklabels=['Accuracy', 'Balanced accuracy', 'FAKE recall', 'Macro-F1', 'ROC-AUC'],
               ylim=(0, 1.14), title='ResNet-18: recorded vs current result on the same 20,000 test images')
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc='lower right')
        save(fig, 'resnet18_test_improvement')


def publish_current(data, folder, run):
    """Refresh only current-image sections; preserve historical video content."""
    folder.mkdir(parents=True, exist_ok=True)
    entries = list(data['models'].items())
    rows = []
    for identity, entry in entries:
        for split in ['validation', 'test']:
            m = entry[split]
            rows.append(f"| {entry['name']} | {split} (20,000) | {m['accuracy']:.3%} | {m['balanced_accuracy']:.3%} | {m['fake_recall']:.3%} | {m['macro_f1']:.6f} | {m['roc_auc']:.6f} |")
    table = '| Model | Split / images | Accuracy | Balanced accuracy | FAKE recall | Macro-F1 | ROC-AUC |\n|---|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    base = folder.relative_to(PROJECT_ROOT).as_posix()
    section = '## Current image models — full-dataset continuation\n\n'
    section += 'The existing `R18_FT_20261021` and `E02_efficientnet_b0` identities resolve to validation-selected checkpoints. Both continuation stages use the predefined 100,000-image training split and FP32 evaluation at 224 pixels, with FAKE positive and threshold 0.5. Historical runs and video checkpoints remain preserved.\n\n'+table+'\n\n'
    for identity, entry in entries:
        t = entry['training']; before = entry['baseline_validation']
        warm = data['historical_e02']
        section += f"**{entry['name']}**: trained epochs {t['start_epoch']}–{t['end_epoch']}, selected epoch {t['selected_epoch']}; early stopping {'triggered' if t['early_stopping_triggered'] else 'did not trigger'}. Full-validation accuracy changed from {before['accuracy']:.3%} to {entry['validation']['accuracy']:.3%}. "
        section += ('ResNet numbering includes the six previous 100k epochs.\n\n' if identity.startswith('R18') else
                    f"The separate warm start used {warm['training_images']:,} training images, selected at warm-start epoch {warm['best_epoch']}; its historical {warm['validation']['accuracy']:.2%} result used only {warm['validation_images']:,} validation images. The before/after comparison here reruns the original E02 checkpoint on all 20,000 validation images.\n\n")
        if t['selected_epoch'] == 0:
            section += 'The original 12k-trained warm-start weights were retained because the completed full-dataset stage did not improve validation selection.\n\n'
    section += 'The predefined test split was used in historical reporting. These repeated test results describe the frozen selections; they are not a newly independent sealed benchmark and did not choose epochs or deployment candidates. Family ranking retains the common 2,000-image validation cohort, separately from full-validation selection.\n\n'
    section += f'![Full-validation model comparison]({base}/validation_comparison.png)\n\n![Training accuracy]({base}/resnet18_accuracy.png)\n\n![EfficientNet full-dataset training]({base}/efficientnet_b0_accuracy.png)\n\n'
    section += f'All accuracy/loss, confusion, ROC/PR, class-wise, resource, latency and before/after charts have PNG and editable SVG versions in [the current figure folder]({base}/README.md). The [executed report](results/current_image_models/REPORT.md) includes model hashes, timings, training settings, and verification.\n\n'
    readme = PROJECT_ROOT/'README.md'
    text = readme.read_text(encoding='utf-8')
    starts = [text.index(marker) for marker in ['## Current image models', '## Demo image continuation — 21 October 2026'] if marker in text]
    if starts:
        start = min(starts); end = text.index('Current debugging, XAI, latency', start)
        text = text[:start]+section+text[end:]
    else:
        position = text.index('\n\n')+2; text = text[:position]+section+text[position:]
    heading = '### Current image-family ranking (common 2,000-image validation cohort)'
    if heading in text:
        start = text.index(heading); end = text.index('### Video-family ranking', start)
        ranking = heading+'\n\nCurrent checkpoint aliases; full-validation selection is reported above. Training budgets differ across historical variants.\n\n| Family | Best current variant | Best validation BA | Mean validation BA | Comparable runs |\n|---|---|---:|---:|---:|\n'
        for group in data['families']:
            ranking += f"| {group['family']} | {group['name']} | {group['best']:.3%} | {group['mean']:.3%} | {group['runs']} |\n"
        text = text[:start]+ranking+'\n'+text[end:]
    text = text.replace('S0 trained on 100,000 images, E01–E04 on 12,000, and S2 on DFDC videos;',
                        'S0 and the current ResNet/E02 models trained on 100,000 images, historical E01/E03/E04 on 12,000, and S2 on DFDC videos;')
    if data['families']:
        recommended = data['families'][0]['name']
        text = text.replace('The continued ResNet-18 is the current image default; S0 remains available as the original image baseline. S6 is the video default.',
                            f'The recommended image model is {recommended} on the common validation ranking; S0 remains available as the original image baseline. S6 is the video default.')
    readme.write_text(text, encoding='utf-8')
    captions = {
        'resnet18_accuracy': 'Training and full-validation accuracy across trustworthy global epochs; selected epoch marked.',
        'resnet18_loss': 'Observed global-epoch losses; stage boundaries and early stopping shown where applicable.',
        'efficientnet_b0_accuracy': 'Only the new 100k-image stage; the earlier 12k-image warm start is excluded from this axis.',
        'efficientnet_b0_loss': 'Training and validation loss over the full-dataset stage.',
        'validation_comparison': 'Two frozen models on the same full 20,000-image validation split.',
        'test_comparison': 'Repeated historically used test split after freezing both selections; no test-based tuning.',
        'confusion_matrices': 'REAL then FAKE row/column order; off-diagonals are errors at threshold 0.5.',
        'roc_curves': 'FAKE-positive ROC from every saved validation/test probability.',
        'precision_recall_curves': 'FAKE-positive precision–recall; legends report average precision.',
        'classwise_metrics': 'Precision, recall and F1 for each class and split.',
        'model_resources': 'Actual parameter counts and selected checkpoint bytes, including metadata.',
        'inference_comparison': 'Identical FP32 resident-input timing protocol with warmup and synchronization.',
        'validation_improvement': 'Both existing and updated checkpoints evaluated on the same full validation split.'}
    if list(data['models'].values())[0].get('historical_test'):
        captions['resnet18_test_improvement'] = 'Recorded epoch-6 versus current frozen ResNet results on the same historically used 20,000 test images; descriptive only, never test-based selection.'
    notes = '# Current image figures\n\nAll metrics are derived from saved predictions. Training stages and comparison scopes are explicit.\n\n'
    for name, caption in captions.items():
        notes += f'- [{name} PNG]({name}.png) · [editable SVG]({name}.svg): {caption}\n'
    notes += '\nRegenerate using `python -B -m tools.plot_demo_accuracy --current`. A portable regeneration uses `--snapshot docs/figures/current_image_models/metrics.json`. No inference or training runs during plotting.\n'
    (folder/'README.md').write_text(notes, encoding='utf-8')
    guide = PROJECT_ROOT/'docs/automatic_gallery.md'
    guide_text = guide.read_text(encoding='utf-8')
    start = guide_text.find('## Current image checkpoint')
    if start >= 0:
        end = guide_text.index('Upload an image or video', start)
        current = '## Current image checkpoints\n\n'
        current += 'The existing ResNet and E02 selector entries now use their validation-selected full-dataset checkpoints. The selector shows full 20,000-image validation accuracy; model details separately expose the common 2,000-image ranking cohort and repeated 20,000-image test result. Both retain all six CNN attribution methods.\n\n'
        current += 'Checkpoint hashes are included in prediction and explanation cache keys. Current training provenance and metrics are in [the execution report](../results/current_image_models/REPORT.md). Historical video results remain unchanged.\n\n'
        guide_text = guide_text[:start]+current+guide_text[end:]
    if data['models']['E02_efficientnet_b0']['training']['selected_epoch'] > 0:
        guide_text = guide_text.replace('| EfficientNet-B0 [a, img12k] | E02_efficientnet_b0 |', '| EfficientNet-B0 [ft, a, img100k] | E02_efficientnet_b0 |')
    guide.write_text(guide_text, encoding='utf-8')


if __name__ == '__main__':
    main()
