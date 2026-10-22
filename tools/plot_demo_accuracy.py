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
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    data = read(args.snapshot) if args.snapshot else snapshot(args.run)
    folder = args.output.resolve()
    plots(data, folder)
    (folder/'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    publish(data, folder)
    print(f'Updated README and four PNG/SVG plots in {folder}')


if __name__ == '__main__':
    main()
